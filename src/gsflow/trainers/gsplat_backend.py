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


def _build_cfg(ctx: StageContext, data_dir: Path, out_dir: Path, iterations: int) -> TrainConfig:
    """把 `[train]` 配置注入 `TrainConfig`。

    ⚠️ 这里有两个**曾经真实造成严重问题**的写法，务必不要再引入：

    1) **绝不要写 `ctx.cfg(k, d) or d`**。`or` 会把合法的 `0` / `False` 当成"缺省"：
       配置里 `opacity_reset_interval = 0` 表示"关闭不透明度重置"，
       而 `int(0 or 3000)` 得 3000 —— 于是"已关闭重置"这个修复**从未生效**，
       训练一直每 3000 步重置一次，末次重置状态被存成最终检查点，不透明度塌陷成薄雾。
       `ctx.cfg(k, default)` 本身已经是"键存在就用配置值、否则用 default"，
       多加 `or` 只会把 0/False 吃掉。这里用 `None` 当哨兵，只判断"是否存在"。

    2) **按 dataclass 字段循环注入**，不要手写一长串 `name=ctx.cfg("name", ...)`。
       手写会漏字段：之前 36 个字段里只注入了 19 个，`opacity_reg` / `scale_reg` /
       `min_opacity_final` / `lr_opacities` 等**改了配置完全不生效且不报错**。
       现在自动覆盖所有字段，新增字段无需再改这里。
    """
    import dataclasses
    import typing

    # 由调用方显式给定、不来自配置的字段
    explicit = {"data_dir": data_dir, "out_dir": out_dir, "iterations": iterations}

    try:
        hints = typing.get_type_hints(TrainConfig)
    except Exception:  # noqa: BLE001
        hints = {}

    def cast(value: object, tp: object) -> object:
        """按字段声明类型转换（配置值可能来自 TOML，也可能是 --set 传进来的字符串）。"""
        try:
            if tp is bool:
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "true", "yes", "on")
                return bool(value)
            if tp is int:
                return int(float(value))  # 兼容 "3000" / 3000.0
            if tp is float:
                return float(value)
            if tp is str:
                return str(value)
        except (TypeError, ValueError):
            return value
        return value

    kwargs: dict[str, object] = {}
    for f in dataclasses.fields(TrainConfig):
        if f.name in explicit:
            kwargs[f.name] = explicit[f.name]
            continue
        raw = ctx.cfg(f.name, None)          # 只判断"有没有"，不用 `or`
        if raw is None:
            kwargs[f.name] = f.default        # 让 dataclass 默认值继续兜底
        else:
            kwargs[f.name] = cast(raw, hints.get(f.name))
    return TrainConfig(**kwargs)  # type: ignore[arg-type]


class GsplatBackend:
    name = "gsplat"

    def available(self, ctx: StageContext) -> tuple[bool, str]:
        try:
            import torch
        except ImportError:
            return False, "未安装 torch：pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        try:
            import gsplat  # noqa: F401
        except ImportError:
            return False, "未安装 gsplat：pip install gsplat"
        try:
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
                "解决（**不需要管理员权限**，推荐）："
                "先 `scripts\\download.cmd cuda` 下载安装包，再跑 `scripts\\build_portable_cuda.cmd` "
                "自动解包成 tools\\cuda（约 2 分钟）。"
                "之后用 `scripts\\gsflow_gpu.cmd` 跑训练/导出（它会自动加载 MSVC + 便携 CUDA 环境）。"
            )
        return True, ""

    def run(self, ctx: StageContext, data_dir: Path, out_dir: Path, iterations: int) -> dict:
        ok, why = self.available(ctx)
        if not ok:
            raise StageError("train", f"训练后端不可用：{why}", "先运行 `gsflow doctor` 查看待办清单")
        cfg = _build_cfg(ctx, data_dir, out_dir, iterations)
        return train(cfg, ctx)
