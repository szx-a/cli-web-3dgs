"""编排器：唯一执行流程的地方，CLI 与 Web 都只调用它。

职责边界：
  * 建作业目录、写 job.json（参数快照 + 阶段状态）；
  * 按注册顺序跑阶段，支持跳过已完成阶段、强制重跑单阶段；
  * 把每个阶段的开始/结束/失败都变成事件；
  * 失败默认中止（halt_on_error），可选继续。

它不认识 argparse，也不认识 FastAPI。
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import paths
from ..config import Settings, load as load_settings
from ..errors import StageCancelled, StageError
from ..events import EventBus
from . import registry
from .context import StageContext


def new_job_id(prefix: str = "job") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{stamp}-{short}"


def _safe_name(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", text.strip())
    return text[:60] or "job"


@dataclass
class JobSpec:
    """一个作业的输入描述（CLI 与 GUI 都构造它）。"""

    input_path: str = ""                  # 视频文件 / 图片目录 / 单张图
    images_dir: str = ""                  # 已抽帧或已准备的图片目录（优先于 input_path）
    job_id: str = ""
    title: str = "3DGS 重建"
    stages: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)   # 点路径覆盖，如 {"colmap.max_image_size": 1600}
    force_stages: list[str] = field(default_factory=list)
    halt_on_error: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "images_dir": self.images_dir,
            "job_id": self.job_id,
            "title": self.title,
            "stages": self.stages,
            "params": self.params,
            "force_stages": self.force_stages,
            "halt_on_error": self.halt_on_error,
        }


class Orchestrator:
    def __init__(
        self,
        spec: JobSpec,
        settings: Settings | None = None,
        events: EventBus | None = None,
        cancel_event: threading.Event | None = None,
        dry_run: bool = False,
    ):
        self.spec = spec
        self.settings = settings or load_settings(spec.params)
        self.events = events or EventBus()
        self.cancel_event = cancel_event or threading.Event()
        self.dry_run = dry_run
        self.job_id = spec.job_id or new_job_id()
        self.job_dir = Path(self.settings.get("project.workspace_dir", str(paths.WORKSPACE_DIR))) / self.job_id
        self.results: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ 准备
    def prepare(self) -> StageContext:
        self.job_dir.mkdir(parents=True, exist_ok=True)
        (self.job_dir / "logs").mkdir(exist_ok=True)
        self.events.attach_jsonl(self.job_dir / "logs" / "events.jsonl")
        ctx = StageContext(
            job_id=self.job_id,
            job_dir=self.job_dir,
            settings=self.settings,
            events=self.events,
            params=self.spec.params,
            spec=self.spec.to_dict(),
            dry_run=self.dry_run,
            force=bool(self.spec.force_stages),
            _cancel=self.cancel_event,
        )
        existing = ctx.read_job()
        job = {
            **existing,
            "job_id": self.job_id,
            "title": self.spec.title,
            "spec": self.spec.to_dict(),
            "settings_snapshot": self.settings.data,
            "created_at": existing.get("created_at") or time.time(),
            "updated_at": time.time(),
            "status": "running",
            "stage_order": registry.resolve(self.spec.stages),
            "stages": {**existing.get("stages", {}), **self.results},
        }
        ctx.write_job(job)
        return ctx

    # ------------------------------------------------------------------ 执行
    def run(self) -> dict[str, Any]:
        ctx = self.prepare()
        order = registry.resolve(self.spec.stages)
        self.events.info(
            f"作业 {self.job_id} 启动：{self.spec.title}",
            stage="orchestrator",
            job_id=self.job_id,
            data={"stages": order, "job_dir": str(self.job_dir)},
        )
        if self.dry_run:
            self.events.info(f"[dry-run] 将执行阶段：{' -> '.join(order)}", stage="orchestrator", job_id=self.job_id)

        failed_stage = ""
        for idx, name in enumerate(order):
            stage = registry.get(name)
            ctx.bind_stage(name)
            stage_progress_base = idx / max(len(order), 1)

            # 是否需要跳过
            force = name in self.spec.force_stages
            if not force and not self.dry_run and stage.outputs_present(ctx):
                self.events.info(
                    f"跳过 {stage.title}（产物已存在，可用 --force {name} 重跑）",
                    stage=name,
                    job_id=self.job_id,
                    progress=stage_progress_base,
                )
                self.results[name] = {"status": "skipped", "message": "产物已存在"}
                self._persist(ctx)
                continue

            missing = stage.missing_inputs(ctx)
            if missing and not self.dry_run:
                if stage.optional:
                    self.events.warn(f"跳过 {stage.title}：缺少输入 {missing}", stage=name, job_id=self.job_id)
                    self.results[name] = {"status": "skipped", "message": f"缺少输入 {missing}"}
                    self._persist(ctx)
                    continue
                advice = f"请先完成前置阶段：{', '.join(stage.requires) or '检查输入路径'}"
                self.events.error(f"{stage.title} 缺少输入文件：{missing}", stage=name, job_id=self.job_id)
                self.results[name] = {"status": "failed", "message": f"缺少输入 {missing}"}
                failed_stage = name
                break

            self.events.info(f"▶ 开始 {stage.title}", stage=name, job_id=self.job_id, progress=stage_progress_base)
            self.results[name] = {"status": "running"}
            self._persist(ctx)

            if self.dry_run:
                self.results[name] = {"status": "dry-run", "message": "未真正执行"}
                continue

            try:
                result = stage.execute(ctx)
            except StageCancelled as exc:
                self.events.warn(str(exc), stage=name, job_id=self.job_id)
                self.results[name] = {"status": "cancelled", "message": str(exc)}
                failed_stage = name
                break
            except StageError as exc:
                self.events.error(str(exc), stage=name, job_id=self.job_id)
                if exc.advice:
                    self.events.warn(f"修复建议：{exc.advice}", stage=name, job_id=self.job_id)
                self.results[name] = {"status": "failed", "message": str(exc), "advice": exc.advice}
                failed_stage = name
                if self.spec.halt_on_error:
                    break
            else:
                self.results[name] = result.to_dict()
                self._persist(ctx)
                self.events.success(
                    f"✔ {stage.title} 完成（{result.seconds:.1f}s）"
                    + (f"：{result.message}" if result.message else ""),
                    stage=name,
                    job_id=self.job_id,
                    progress=(idx + 1) / max(len(order), 1),
                    data={"metrics": result.metrics},
                )

        status = "failed" if failed_stage else ("cancelled" if self.cancel_event.is_set() else "done")
        job = ctx.read_job()
        job.update({"status": status, "updated_at": time.time(), "stages": self.results, "finished_at": time.time()})
        ctx.write_job(job)
        level = "error" if status == "failed" else ("warn" if status == "cancelled" else "success")
        self.events.emit(
            f"作业 {self.job_id} 结束：{status}",
            level=level if level != "success" else "success",
            stage="orchestrator",
            job_id=self.job_id,
            progress=1.0,
            data={"status": status, "results": self.results},
        )
        return {"job_id": self.job_id, "status": status, "stages": self.results, "job_dir": str(self.job_dir)}

    def _persist(self, ctx: StageContext) -> None:
        job = ctx.read_job()
        job.update({"stages": self.results, "updated_at": time.time()})
        ctx.write_job(job)


# --------------------------------------------------------------------- 工具

def list_jobs(workspace_dir: Path | None = None) -> list[dict[str, Any]]:
    root = workspace_dir or paths.WORKSPACE_DIR
    out: list[dict[str, Any]] = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        f = d / "job.json"
        if not f.is_file():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        data["job_dir"] = str(d)
        out.append(data)
    return out


def load_job(job_id: str, workspace_dir: Path | None = None) -> dict[str, Any]:
    root = workspace_dir or paths.WORKSPACE_DIR
    f = root / job_id / "job.json"
    if not f.exists():
        raise FileNotFoundError(f"作业不存在：{job_id}")
    data = json.loads(f.read_text(encoding="utf-8"))
    data["job_dir"] = str(root / job_id)
    return data
