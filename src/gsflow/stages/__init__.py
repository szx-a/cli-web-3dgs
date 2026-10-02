"""阶段注册：新增阶段只要在这里 import + register 即可，CLI 与 GUI 自动同步。"""

from __future__ import annotations

from ..pipeline import registry
from .colmap import ColmapStage
from .export import ExportStage
from .frames import FramesStage
from .mask import MaskStage
from .metrics import MetricsStage
from .train import TrainStage

_REGISTERED = False


def register_all() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    for stage in (FramesStage(), MaskStage(), ColmapStage(), TrainStage(), ExportStage(), MetricsStage()):
        registry.register(stage)
    _REGISTERED = True


register_all()

__all__ = [
    "register_all",
    "FramesStage",
    "MaskStage",
    "ColmapStage",
    "TrainStage",
    "ExportStage",
    "MetricsStage",
]
