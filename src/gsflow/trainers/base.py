"""训练后端抽象：gsplat（默认） / official（原版 3DGS 仓库）。

训练阶段只认 `TrainBackend.run()`，因此以后想接 LiteGS / faster-gaussian-splatting
之类的加速实现，只需要写一个新的 backend，不改流程。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..pipeline.context import StageContext


class TrainBackend(Protocol):
    name: str

    def available(self, ctx: StageContext) -> tuple[bool, str]:
        """返回 (是否可用, 不可用原因/修复建议)。"""

    def run(self, ctx: StageContext, data_dir: Path, out_dir: Path, iterations: int) -> dict:
        """执行训练，返回指标字典。"""


def pick(name: str, ctx: StageContext):
    key = (name or "gsplat").lower()
    if key == "gsplat":
        from .gsplat_backend import GsplatBackend

        return GsplatBackend()
    if key in ("official", "inria", "graphdeco"):
        from .official_trainer import OfficialBackend

        return OfficialBackend()
    if key == "external":
        from .official_trainer import ExternalBackend

        return ExternalBackend()
    raise ValueError(f"未知训练后端 '{name}'，可用：gsplat / official / external")
