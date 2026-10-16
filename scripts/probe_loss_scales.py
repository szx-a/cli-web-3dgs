"""测量训练损失各分量的**真实量级**，用来校准正则项权重（而不是抄论文的数字）。

动机（用户指出"数学上的放缩过于暴力就越难逼近渐近线"）：
2DGS 论文里深度畸变损失 λd=1000（DTU）/ 0（Mip-NeRF 360 室外），
而我们实现的是**深度图 TV/二阶差分**形式的代理，权重只有 0.05。
两者尺度完全不同 —— **论文的 1000 和我们的 0.05 无法直接比较**。

正确做法是先测出：在**本项目的损失口径**下，各分量的原始数值是多少、
与 RGB 主损失的比值是多少，再据此定权重。本脚本就是做这件事。

它会：
  1. 用真实数据跑若干步（默认 30 步），记录 L1 / (1-SSIM) / depth / normal 的原始值；
  2. 计算每个正则项相对 RGB 损失的比值；
  3. 反推"若要让正则项占比达到 X%，权重应设多少"。

用法: scripts\\gpu_python.cmd scripts\\probe_loss_scales.py --job ab-composite --steps 30
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
    ap.add_argument("--job", default="ab-composite")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import torch
    from gsplat import rasterization

    from gsflow.trainers.gsplat_trainer import (
        TrainConfig, _init_scales_nn, _load_images, depth_distortion_loss,
        depth_to_normal, gaussian_normals, load_colmap_data, normal_consistency_loss,
        ssim_simple,
    )

    job = ROOT / "workspace" / args.job
    data_dir = job / "04_train" / "data"
    metrics = json.loads((job / "04_train" / "metrics.json").read_text(encoding="utf-8"))
    ply = Path(metrics["best_ply"])
    from plyfile import PlyData

    print(f"作业 {args.job}  用已训练模型 {ply.parent.name} 评估各损失分量量级")
    print("（用已训练模型是为了得到接近收敛时的量级；训练早期量级更大）\n")

    data = load_colmap_data(data_dir, downscale=1, white_background=False, flip_camera=False)
    imgs, masks = _load_images(data_dir, data["names"], None)
    ks = torch.from_numpy(data["ks"]).float()
    c2ws = torch.from_numpy(data["c2ws"]).float()
    viewmats = torch.linalg.inv(c2ws)

    d = PlyData.read(str(ply)).elements[0].data
    dev = torch.device("cuda")
    means = torch.tensor(np.stack([d["x"], d["y"], d["z"]], 1), dtype=torch.float32, device=dev)
    scales = torch.tensor(np.exp(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], 1)),
                          dtype=torch.float32, device=dev)
    q = np.stack([d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"]], 1)
    quats = torch.tensor(q, dtype=torch.float32, device=dev)
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
    op = 1.0 / (1.0 + np.exp(-np.asarray(d["opacity"], dtype=np.float64)))
    opacities = torch.tensor(op, dtype=torch.float32, device=dev)
    shs = torch.tensor(np.stack([np.asarray(d[f"f_dc_{k}"], dtype=np.float64) for k in range(3)], 1),
                       dtype=torch.float32, device=dev).unsqueeze(1)

    rows = []
    n = min(args.steps, len(imgs))
    for i in range(0, n, max(1, n // 8)):
        gt = imgs[i].to(dev)
        H, W = gt.shape[:2]
        with torch.no_grad():
            colors, alphas, meta = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=shs, viewmats=viewmats[i][None].to(dev), Ks=ks[i][None].to(dev),
                width=W, height=H, sh_degree=0, near_plane=0.01, far_plane=1e10,
                packed=False, render_mode="RGB+ED",
            )
        rgb = colors[0][..., :3].clamp(0, 1)
        depth = colors[0][..., 3]
        l1 = float((rgb - gt).abs().mean())
        ssim_v = float(ssim_simple(rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        # 与训练一致：深度正则需要额外一次渲染；这里用同一次 ED 渲染的深度做量级估计
        d_tv = float(depth_distortion_loss(depth[None], alphas[0][..., 0][None], "tv"))
        rows.append((i, l1, 1.0 - ssim_v, d_tv))

    print(f"{'view':>5} {'L1':>10} {'1-SSIM':>10} {'depth_tv':>12} {'depth/L1':>10} {'depth/(1-ssim)':>15}")
    for i, l1, s, dtv in rows:
        print(f"{i:>5} {l1:>10.5f} {s:>10.5f} {dtv:>12.6f} {dtv / max(l1, 1e-9):>10.4f} {dtv / max(s, 1e-9):>15.4f}")

    m_l1 = float(np.mean([r[1] for r in rows]))
    m_s = float(np.mean([r[2] for r in rows]))
    m_d = float(np.mean([r[3] for r in rows]))
    rgb_loss = 0.8 * m_l1 + 0.2 * m_s      # 默认 ssim_weight=0.2
    print(f"\n平均：L1={m_l1:.5f}  1-SSIM={m_s:.5f}  depth_tv={m_d:.6f}")
    print(f"RGB 主损失（0.8·L1 + 0.2·(1-SSIM)）= {rgb_loss:.5f}")
    print(f"\n若希望 depth 项占 RGB 损失的 p%：权重 λ = p% · RGB / depth_tv")
    for p in (0.01, 0.05, 0.1, 0.2):
        print(f"  p={p * 100:>4.1f}%  ->  λd ≈ {p * rgb_loss / max(m_d, 1e-12):>10.2f}")
    print("\n对照：2DGS 论文给的是 **λd=1000（DTU）/ 0（Mip-NeRF 360 室外）**，")
    print("      但它作用于 Σ ωiωj(di−dj)²（未归一化），与本项目的 TV 代理**尺度不同**，")
    print("      不能直接照搬；应以本表算出的 λ 为起点做受控扫描。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
