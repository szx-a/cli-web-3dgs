"""最小复现：检查 gsplat 的 rasterization 是否把梯度回传到 means。

背景：训练时 `means.grad` 恒为 0，导致自适应密度控制（依赖位置梯度）完全不触发，
模型只有初始稀疏点。本脚本直接构造一次前向+反向，逐个参数检查梯度。

用法：scripts\\gsflow_gpu.cmd   （先加载 MSVC/便携 CUDA 环境）
      env\\venv\\Scripts\\python.exe scripts\\probe_gsplat_grad.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow import paths  # noqa: E402

paths.apply_runtime_env()

import torch  # noqa: E402
from gsplat import rasterization  # noqa: E402


def main() -> int:
    dev = "cuda"
    torch.manual_seed(0)
    n = 64
    H = W = 64

    means = torch.nn.Parameter((torch.rand(n, 3, device=dev) - 0.5) * 0.6)
    scales = torch.nn.Parameter(torch.full((n, 3), -2.0, device=dev))
    quats = torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]] * n, device=dev))
    opac = torch.nn.Parameter(torch.zeros(n, device=dev))
    shs = torch.nn.Parameter(torch.rand(n, 1, 3, device=dev))

    vm = torch.eye(4, device=dev)[None]
    K = torch.tensor([[[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1.0]]], device=dev)
    gt = torch.rand(H, W, 3, device=dev)

    colors, alphas, info = rasterization(
        means=means, quats=quats, scales=torch.exp(scales),
        opacities=torch.sigmoid(opac), colors=shs,
        viewmats=vm, Ks=K, width=W, height=H,
        sh_degree=0, packed=False, render_mode="RGB",
    )
    print("前向输出:", colors.shape, "alpha 非零像素:", int((alphas > 0).sum()))

    loss = (colors[0] - gt).abs().mean()
    loss.backward()

    print("\n各参数梯度：")
    for name, p in (("means", means), ("scales", scales), ("quats", quats),
                    ("opacities", opac), ("shs", shs)):
        g = p.grad
        if g is None:
            print(f"  {name:<10} grad = None  ← 梯度没回传")
        else:
            print(f"  {name:<10} grad.abs().sum() = {float(g.abs().sum()):.6e}")

    print("\nrasterization 返回的 info 键:", list(info.keys()) if isinstance(info, dict) else type(info))

    print("\n结论：")
    if means.grad is not None and float(means.grad.abs().sum()) > 0:
        print("  means 有梯度 → 密度控制应该能工作；若训练里仍为 0，问题在我们的训练循环")
    else:
        print("  means 没有梯度 → gsplat 的 backward 不回传位置梯度（需改用其它分数控制密度）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
