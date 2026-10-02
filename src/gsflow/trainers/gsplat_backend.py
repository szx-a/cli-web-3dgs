"""训练后端：gsplat（默认）。"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..errors import StageError
from ..pipeline.context import StageContext
from .gsplat_trainer import TrainConfig, train


def _has_nvcc() -> bool:
    """检测 nvcc：PATH -> CUDA_HOME/CUDA_PATH 下的 bin。"""
    if shutil.which("nvcc"):
        return True
    for key in ("CUDA_HOME", "CUDA_PATH"):
        home = os.environ.get(key)
        if home and (Path(home) / "bin" / "nvcc.exe").exists():
            return True
    for root in (
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    ):
        base = Path(root)
        if base.exists() and any((d / "bin" / "nvcc.exe").exists() for d in base.iterdir() if d.is_dir()):
            return True
    return False


class GsplatBackend:
    name = "gsplat"

    def available(self, ctx: StageContext) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "未安装 torch：pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        try:
            import gsplat  # noqa: F401
        except ImportError:
            return False, "未安装 gsplat：pip install gsplat"
        try:
            import torch

            if not torch.cuda.is_available():
                return False, "torch 看不到 CUDA 设备：确认装的是 cu128+ 轮子"
        except Exception as exc:  # pragma: no cover
            return False, f"CUDA 自检失败：{exc}"

        # gsplat 的 CUDA 内核是 JIT 编译的，必须有 nvcc。
        # 没有 nvcc 时 gsplat 只会打印一句 "No CUDA toolkit found. gsplat will be disabled."，
        # 然后在调用时抛 'NoneType' object has no attribute 'CameraModelType' —— 极难排查，
        # 所以这里提前拦下来，给出明确的修复动作。
        if not _has_nvcc():
            return False, (
                "gsplat 需要 nvcc 现场编译 CUDA 内核，但当前找不到 CUDA Toolkit。"
                "解决：先跑 `scripts\\download.cmd cuda` 下载安装包（约 3.3GB），"
                "再以管理员身份运行 `powershell -ExecutionPolicy Bypass -File scripts\\install_cuda.ps1`。"
            )
        return True, ""

    def run(self, ctx: StageContext, data_dir: Path, out_dir: Path, iterations: int) -> dict:
        ok, why = self.available(ctx)
        if not ok:
            raise StageError("train", f"训练后端不可用：{why}", "先运行 `gsflow doctor` 查看待办清单")
        cfg = TrainConfig(
            data_dir=data_dir,
            out_dir=out_dir,
            iterations=iterations,
            resolution=int(ctx.cfg("resolution", 1600) or 0),
            sh_degree=int(ctx.cfg("sh_degree", 3)),
            data_device=str(ctx.cfg("data_device", "cuda")),
            cap_max=int(ctx.cfg("cap_max", 1_500_000) or 0),
            eval=bool(ctx.cfg("eval", True)),
            seed=int(ctx.cfg("seed", 42) or 42),
        )
        return train(cfg, ctx)
