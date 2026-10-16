"""探明 gsplat `rasterization()` 返回的 `meta` 里有什么，并寻找**正确的利用率指标**。

背景（用户方案 B 的实现失败）：
我用 `|∂L/∂opacity|` 累积作为"利用率"，结果把 **95% 的高斯剪掉**、PSNR 崩到 21.86。
原因推断：梯度小 ≠ 没用，而可能表示"**已经拟合好了**"。所以这个指标方向可能是反的。

正确的"利用率"应当回答：**这个高斯到底有没有被哪个像素需要？**
候选：
  1. 每高斯的**可见计数**（官方密度控制用的 `count`：它被投影进多少个视角的视锥）；
  2. 每高斯的**累积 α 权重**（渲染时它对像素的实际贡献之和）。

本脚本只做**只读探明**：打印 meta 的键与形状、均值协方差形状，
并验证一个便宜的代理指标 —— 用额外渲染通道取出"被渲染出的每高斯可见性"。

用法: scripts\\gpu_python.cmd scripts\\probe_utilization_metric.py --job op-none
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="op-none")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import torch
    from gsplat import rasterization

    from gsflow.gsdata.shio import read_gaussians_full
    from gsflow.stages.export import ExportStage

    job = ROOT / "workspace" / args.job
    g = read_gaussians_full(ExportStage._find_ply(job))
    dev = "cuda"
    n = g["n"]
    print(f"作业 {args.job}：高斯 {n}，SH 阶数 {g['sh_degree']}")

    means = torch.as_tensor(g["xyz"], device=dev)
    quats = torch.as_tensor(g["quats"], device=dev)
    scales = torch.as_tensor(g["scales"], device=dev)
    opac = torch.as_tensor(g["opacity"], device=dev).reshape(-1).clamp(0, 1)
    shs = torch.as_tensor(g["shs"], device=dev)

    # 用第一个训练相机
    data = json.loads((job / "04_train" / "metrics.json").read_text(encoding="utf-8"))
    from gsflow.trainers.gsplat_trainer import load_colmap_data
    d = load_colmap_data(job / "04_train" / "data", downscale=1,
                         white_background=False, flip_camera=False)
    ks = torch.from_numpy(d["ks"]).float()
    c2ws = torch.from_numpy(d["c2ws"]).float()
    vm = torch.linalg.inv(c2ws)[0][None].to(dev)
    K = ks[0][None].to(dev)
    H = W = 64

    with torch.no_grad():
        out = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
            viewmats=vm, Ks=K, width=W, height=H, sh_degree=int(g["sh_degree"]),
            near_plane=0.01, far_plane=1e10, packed=False, render_mode="RGB",
        )
    colors, alphas, meta = out
    print(f"\ncolors {tuple(colors.shape)}  alphas {tuple(alphas.shape)}  meta 类型 {type(meta)}")
    if isinstance(meta, dict):
        for k, v in meta.items():
            shape = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
            extra = ""
            if hasattr(v, "shape") and v.numel() and v.dtype.is_floating_point:
                extra = f"  mean={float(v.mean()):.3e}"
            print(f"  meta[{k!r}] = {shape}{extra}")

    # ---- 便宜的代理：用额外通道取"每高斯被渲染出的贡献" ----
    # 思路：`colors` 的通道数可 >3。若把第 i 个高斯的值设为 1、其余为 0，
    # 渲染出的通道就是该高斯在各像素上的权重 —— 但那需要每个高斯渲一次，太贵。
    # 这里验证**另一个**可行代理：把 colors 当成"每高斯一个标量"来传入，
    # 看 gsplat 是否支持 [N, K, C] 且 K>1（每高斯多通道），
    # 从而可以一次渲染出"低不透明度高斯的权重分布"。
    try:
        test_cols = torch.zeros(n, 1, 4, device=dev)
        test_cols[:, 0, :3] = 0.5
        test_cols[:, 0, 3] = 1.0
        with torch.no_grad():
            c2, a2, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opac, colors=test_cols,
                viewmats=vm, Ks=K, width=W, height=H, sh_degree=0,
                near_plane=0.01, far_plane=1e10, packed=False, render_mode="RGB",
            )
        print(f"\n每高斯 4 通道渲染: colors {tuple(c2.shape)}"
              f"  ⇒ gsplat 支持每高斯多通道 = {c2.shape[-1] == 4}")
    except Exception as e:  # noqa: BLE001
        print(f"\n多通道测试失败：{type(e).__name__}: {str(e)[:100]}")

    print("\n结论要点：")
    print("  * 若 meta 里有 per-gaussian 的计数/权重 ⇒ 直接用官方语义的利用率（最稳）。")
    print("  * 若没有 ⇒ 用'每高斯多通道渲染'在**若干视角**上累加它的 α 权重。")
    print("    这比'梯度大小'正确：梯度小可能只是**已经拟合好了**，")
    print("    而权重为 0 才说明它真的没被任何像素需要。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
