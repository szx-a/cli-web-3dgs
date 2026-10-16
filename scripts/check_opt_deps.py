"""检查本轮新增优化所需的依赖与本地权重是否就绪。

用户方案里 LPIPS 需要 torchvision 的 VGG/AlexNet 权重；本项目**网络受限**
（raw.githubusercontent.com SSL 失败、hf 直连不稳），所以必须先确认权重是否已在本地，
否则"实现了 LPIPS 但训练时下载失败"会变成静默退化。

用法: python scripts/check_opt_deps.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    print("=== 依赖 ===")
    for m in ("torch", "torchvision", "lpips", "numpy", "PIL"):
        try:
            mod = __import__(m)
            print(f"  {m:12s} OK    {getattr(mod, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {m:12s} MISS  {type(e).__name__}: {str(e)[:70]}")

    print("\n=== 缓存目录（必须都在工作区内，否则沙箱会拒绝写）===")
    for k in ("TORCH_HOME", "HF_HOME", "TMPDIR", "TEMP"):
        print(f"  {k:12s} = {os.environ.get(k, '(未设置)')}")

    print("\n=== 本地已缓存的 torchvision 权重 ===")
    th = Path(os.environ.get("TORCH_HOME", "")) if os.environ.get("TORCH_HOME") else None
    found = []
    if th and th.exists():
        for p in th.rglob("*.pth"):
            found.append(p)
    if found:
        for p in found[:10]:
            print(f"  {p.relative_to(th)}   {p.stat().st_size / 1048576:.1f} MB")
    else:
        print("  （无）—— LPIPS 首次使用会尝试下载 vgg16/alexnet 权重；")
        print("     若下载失败，应**优雅退化为只用 L1+SSIM**，而不是中断训练。")

    print("\n=== 结论 ===")
    has_lpips = True
    try:
        __import__("torchvision")
    except Exception:  # noqa: BLE001
        has_lpips = False
    if has_lpips:
        print("  torchvision 可用：LPIPS 可以基于它实现（无需额外包）")
    else:
        print("  **缺 torchvision**：LPIPS 需先安装，或改用「结构相似度 + 梯度域损失」的轻量替代")
    return 0


if __name__ == "__main__":
    sys.exit(main())
