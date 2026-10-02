"""快速环境速览：torch / CUDA / gsplat / 分割后端 / 外部工具 / 磁盘。

用法：env\\venv\\Scripts\\python.exe scripts\\status.py
"""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 必须在 import ultralytics 之前重定向它的配置目录（否则会写 AppData 被拒）
try:
    from gsflow import paths as _paths

    _paths.apply_runtime_env()
except Exception:
    pass

PACKAGES = ["torch", "torchvision", "gsplat", "ultralytics", "rembg", "onnxruntime",
            "fastapi", "uvicorn", "numpy", "PIL", "plyfile", "transformers"]


def main() -> int:
    print("=" * 62)
    print("gsflow 环境速览")
    print("=" * 62)

    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "")
            if not ver:
                try:
                    from importlib.metadata import version as dv

                    ver = dv("pillow" if name == "PIL" else name)
                except Exception:
                    ver = "?"
            print(f"  [OK]   {name:<14} {ver}")
        except Exception as exc:
            print(f"  [MISS] {name:<14} ({type(exc).__name__})")

    print("-" * 62)
    try:
        import torch

        print(f"  CUDA 可用      : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            print(f"  GPU            : {torch.cuda.get_device_name(0)}")
            print(f"  算力           : sm_{cap[0]}{cap[1]}")
            print(f"  torch cuda 版本: {torch.version.cuda}")
            print(f"  内核架构列表   : {torch.cuda.get_arch_list()}")
    except Exception as exc:
        print(f"  torch 检查失败 : {exc}")

    nvcc = shutil.which("nvcc")
    print(f"  nvcc           : {nvcc or '未找到（gsplat 训练需要它）'}")

    print("-" * 62)
    from gsflow.tools import locate

    for tool in ("ffmpeg", "ffprobe", "colmap"):
        info = locate.find_tool(tool, "", required=False)
        print(f"  {tool:<8}: {'OK  ' + info.path if info else '缺失'}")

    print("-" * 62)
    from gsflow.paths import WORKSPACE_DIR

    jobs = sorted((p for p in WORKSPACE_DIR.iterdir() if p.is_dir()), reverse=True) if WORKSPACE_DIR.exists() else []
    print(f"  作业目录: {WORKSPACE_DIR}（{len(jobs)} 个作业）")
    for j in jobs[:5]:
        print(f"    - {j.name}")

    print("-" * 62)
    for letter in ("C", "D", "F"):
        root = Path(f"{letter}:\\")
        if root.exists():
            try:
                u = shutil.disk_usage(root)
                print(f"  {letter}: 可用 {u.free/1024**3:6.1f} GB / {u.total/1024**3:6.1f} GB")
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
