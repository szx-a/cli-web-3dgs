"""阶段基类与阶段间契约。

架构红线（见 docs/01）：
  * 阶段之间只通过磁盘产物 + manifest 通信，不传内存对象；
  * 每个阶段必须声明 inputs/outputs，编排器据此判断“能否跳过/能否续跑”；
  * 阶段内部不得知道 CLI 或 Web 的存在，只通过 ctx.events 汇报。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable

from ..errors import StageError

if TYPE_CHECKING:  # 避免循环导入
    from .context import StageContext


@dataclass
class Artifact:
    """一个阶段产物：相对作业目录的路径 + 人读描述。"""

    path: str
    desc: str = ""
    optional: bool = False

    def abs(self, job_dir: Path) -> Path:
        return job_dir / self.path


@dataclass
class StageResult:
    status: str = "ok"              # ok | skipped | failed | cancelled
    artifacts: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "message": self.message,
            "seconds": round(self.seconds, 2),
        }


class Stage:
    """所有阶段的基类。子类必须覆盖 name / title，并实现 run()。"""

    name: ClassVar[str] = ""
    title: ClassVar[str] = ""
    description: ClassVar[str] = ""
    requires: ClassVar[tuple[str, ...]] = ()      # 依赖的前置阶段
    optional: ClassVar[bool] = False              # 允许被跳过而不报错
    inputs: ClassVar[tuple[Artifact, ...]] = ()
    outputs: ClassVar[tuple[Artifact, ...]] = ()
    # 是否可以用 GUI 单独重跑（产物可被下游继续使用）
    rerunnable: ClassVar[bool] = True

    # --- 生命周期 ---
    def preflight(self, ctx: "StageContext") -> None:
        """前置检查：外部工具、凭据、输入完整性。默认校验输入存在。"""

    def run(self, ctx: "StageContext") -> StageResult:  # pragma: no cover - 抽象
        raise NotImplementedError

    # --- 编排器调用入口 ---
    def execute(self, ctx: "StageContext") -> StageResult:
        started = time.time()
        try:
            if ctx.cancel_requested:
                from ..errors import StageCancelled

                raise StageCancelled(f"[{self.name}] 已取消")
            self.preflight(ctx)
            result = self.run(ctx)
            result.seconds = time.time() - started
            return result
        except Exception as exc:  # 统一收口，交给编排器决定是否继续
            if isinstance(exc, (StageError,)):
                raise
            raise StageError(self.name, f"{type(exc).__name__}: {exc}") from exc

    # --- 工具 ---
    def missing_inputs(self, ctx: "StageContext") -> list[str]:
        missing = []
        for art in self.inputs:
            p = art.abs(ctx.job_dir)
            if not p.exists():
                missing.append(art.path)
        return missing

    def outputs_present(self, ctx: "StageContext") -> bool:
        if not self.outputs:
            return False
        for art in self.outputs:
            if art.optional:
                continue
            if not art.abs(ctx.job_dir).exists():
                return False
        return True

    def declared_artifacts(self) -> Iterable[str]:
        for art in self.outputs:
            yield art.path
