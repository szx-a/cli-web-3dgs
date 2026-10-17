"""审计致密化判据：我们用的"3D 梯度"与官方"2D 屏幕梯度/count"差多少。

**这是本轮最重要的结构性问题之一。**

官方 3DGS / gsplat `DefaultStrategy` 的加密判据：
    grads = meta["means2d"].grad           # [C, N, 2] 屏幕空间梯度
    grads[..., 0] *= width / 2 * n_cameras # 换算到 [-1,1] 屏幕空间
    grads = grads.norm(dim=-1)             # [C, N]
    grads = grad2d / count.clamp_min(1)    # 除以"被看到次数"
    is_grad_high = grads > grow_grad2d     # 阈值 2e-4（官方）

本项目用的是：
    means.grad.norm(dim=-1)                # 3D **世界空间**梯度
    > densify_grad_threshold (2e-4)

两者**量纲完全不同**（一个是"像素/屏幕归一化空间"，一个是"世界单位"），
所以 `2e-4` 在本项目里并不是"官方阈值"，只是**碰巧能用**的一个数。

本脚本在真实训练中同时采集两种判据，回答：
  1. 我们的 3D 梯度分布是什么量级？有多少高斯超过 2e-4？
  2. 官方口径（means2d/count）下的分布是什么量级？
  3. 两种判据选出的"需要加密"的高斯集合**重合度**有多高？
若有官方 strategy 可用，直接对比其触发数量。

用法: scripts\\gpu_python.cmd scripts\\probe_densify_criterion.py --job final-v1
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
    ap.add_argument("--job", default="final-v1")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--checkpoint", default="", help="用哪个 iteration 的 ply（默认 best）")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import torch
    from gsplat import rasterization

    from gsflow.gsdata.shio import read_gaussians_full
    from gsflow.trainers.gsplat_trainer import _load_images, load_colmap_data, ssim_simple

    job = ROOT / "workspace" / args.job
    metrics = json.loads((job / "04_train" / "metrics.json").read_text(encoding="utf-8"))
    ply = Path(args.checkpoint) if args.checkpoint else Path(metrics["best_ply"])
    print(f"作业 {args.job}  检查点 {ply.parent.name}")

    data = load_colmap_data(job / "04_train" / "data", downscale=1,
                            white_background=False, flip_camera=False)
    imgs, masks = _load_images(job / "04_train" / "data", data["names"], None)
    ks = torch.from_numpy(data["ks"]).float()
    c2ws = torch.from_numpy(data["c2ws"]).float()
    viewmats = torch.linalg.inv(c2ws)

    g = read_gaussians_full(ply)
    dev = "cuda"
    means = torch.nn.Parameter(torch.as_tensor(g["xyz"], device=dev))
    quats = torch.as_tensor(g["quats"], device=dev)
    scales = torch.as_tensor(g["scales"], device=dev)
    opac = torch.as_tensor(g["opacity"], device=dev).reshape(-1).clamp(0, 1)
    shs = torch.as_tensor(g["shs"], device=dev)
    n = int(means.shape[0])
    print(f"高斯 {n}，SH 阶数 {g['sh_degree']}")

    grad3d = torch.zeros(n, device=dev)      # 本项目口径：3D 梯度累积
    grad2d = torch.zeros(n, device=dev)      # 官方口径：屏幕梯度累积
    count = torch.zeros(n, device=dev)       # 被看到次数
    lam_d = 0.0                              # 只测 RGB 项的梯度，避免正则干扰

    steps = min(args.steps, len(imgs))
    for i in range(steps):
        gt = imgs[i].to(dev)
        H, W = gt.shape[:2]
        colors, alphas, meta = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
            viewmats=viewmats[i][None].to(dev), Ks=ks[i][None].to(dev),
            width=W, height=H, sh_degree=int(g["sh_degree"]),
            near_plane=0.01, far_plane=1e10, packed=False, render_mode="RGB",
        )
        # 与训练一致的损失
        l1 = (colors[0] - gt).abs().mean()
        s = ssim_simple(colors[0].permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        loss = 0.8 * l1 + 0.2 * (1.0 - s)
        if meta and meta.get("means2d") is not None:
            meta["means2d"].retain_grad()
        loss.backward()

        with torch.no_grad():
            if means.grad is not None:
                grad3d += means.grad.norm(dim=-1)
            m2 = meta.get("means2d") if isinstance(meta, dict) else None
            if m2 is not None and m2.grad is not None:
                gg = m2.grad.detach().clone()
                # 官方第一步：换算到 [-1,1] 屏幕空间
                gg[..., 0] *= (W / 2.0)
                grad2d += gg.norm(dim=-1).reshape(-1)
                # "被看到次数" ≈ 该高斯是否落在视锥内（用 radii>0 或投影在画布内）
                radii = meta.get("radii")
                if radii is not None:
                    vis = (radii.reshape(-1, 2) > 0).any(dim=-1).float()
                    count += vis
            means.grad = None
            if m2 is not None:
                m2.grad = None

    # 官方口径：除以可见次数
    cnt = count.clamp_min(1.0)
    g2d = (grad2d / cnt)
    thr3d = 2e-4
    thr2d = 8e-4        # 官方 absgrad 建议值；非 absgrad 时 2e-4
    sel3d = grad3d > thr3d * steps
    sel2d = g2d > thr2d
    print(f"\n=== 两种判据的分布（{steps} 步累积）===")
    print(f"  本项目 3D 梯度：p50={grad3d.median():.3e} p90={grad3d.quantile(0.9):.3e} "
          f"p99={grad3d.quantile(0.99):.3e} max={grad3d.max():.3e}")
    print(f"  官方 2D 口径  ：p50={g2d.median():.3e} p90={g2d.quantile(0.9):.3e} "
          f"p99={g2d.quantile(0.99):.3e} max={g2d.max():.3e}")
    print(f"\n  3D 判据选中的高斯：{int(sel3d.sum())} / {n}（{(sel3d.float().mean()*100):.1f}%）")
    print(f"  2D 判据选中的高斯：{int(sel2d.sum())} / {n}（{(sel2d.float().mean()*100):.1f}%）")
    inter = int((sel3d & sel2d).sum())
    union = int((sel3d | sel2d).sum())
    print(f"  两种判据的重合：{inter} 个（IoU={inter/max(union,1):.3f}）")
    print(f"\n  可见次数分布：p50={count.median():.0f} p90={count.quantile(0.9):.0f} "
          f"max={count.max():.0f}（共 {steps} 个视角）")
    print("\n解读：")
    print("  * 若两者 IoU 很低 ⇒ 3D 梯度选出的高斯与官方完全不同 ⇒ 加密位置是错的。")
    print("  * 3D 梯度大但 2D 梯度小 = 在世界空间里位置敏感、但屏幕上看不出差别")
    print("    ⇒ 加密对这个视角**无收益**（正是'高斯多但细节不变'的成因）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
