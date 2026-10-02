"""阶段上下文：阶段能看到的全部“外部世界”的受控视图。"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..events import EventBus


@dataclass
class StageContext:
    job_id: str
    job_dir: Path
    settings: Settings
    events: EventBus
    params: dict[str, Any] = field(default_factory=dict)   # 本阶段参数覆盖
    spec: dict[str, Any] = field(default_factory=dict)     # 作业级参数（输入、阶段选择等）
    dry_run: bool = False
    force: bool = False
    _cancel: threading.Event = field(default_factory=threading.Event)
    _stage: str = ""

    # --- 取消 ---
    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def bind_stage(self, stage: str) -> "StageContext":
        self._stage = stage
        return self

    # --- 事件（自动带 job_id / stage） ---
    def emit(self, message: str, level: str = "info", progress: float | None = None, **data: Any):
        return self.events.emit(
            message, level=level, stage=self._stage, job_id=self.job_id, progress=progress, **data
        )

    def info(self, message: str, progress: float | None = None, **data: Any):
        return self.emit(message, "info", progress, **data)

    def warn(self, message: str, **data: Any):
        return self.emit(message, "warn", **data)

    def error(self, message: str, **data: Any):
        return self.emit(message, "error", **data)

    def success(self, message: str, **data: Any):
        return self.emit(message, "success", **data)

    # --- 路径 ---
    def stage_dir(self, name: str | None = None) -> Path:
        d = self.job_dir / (name or self._stage)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def logs_dir(self) -> Path:
        d = self.job_dir / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- 配置读取（阶段参数 > 全局配置） ---
    def cfg(self, key: str, default: Any = None) -> Any:
        stage = self._stage
        if stage and "." not in key:
            stage_params = self.params.get(stage)
            if isinstance(stage_params, dict) and key in stage_params:
                return stage_params[key]
        if key in self.params and "." in key:
            return self.params[key]
        return self.settings.get(key, default)

    # --- 作业清单 ---
    def read_job(self) -> dict[str, Any]:
        f = self.job_dir / "job.json"
        if not f.exists():
            return {}
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def write_job(self, data: dict[str, Any]) -> None:
        f = self.job_dir / "job.json"
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(f)

    def update_job(self, **kw: Any) -> dict[str, Any]:
        data = self.read_job()
        data.update(kw)
        self.write_job(data)
        return data
