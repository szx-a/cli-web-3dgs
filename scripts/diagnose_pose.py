"""诊断：用 COLMAP 稀疏点（已知正确的位置与颜色）直接渲染，
对比真值图像，判断**相机位姿约定**是否正确。

思路：稀疏点来自 COLMAP，位置与颜色都是"真值"。如果位姿约定正确，
把它们渲染出来应当与真值图像**几何对齐**（棋盘落在同一个位置）。
若位姿错，点云会整体偏移/镜像/翻转。

用法：scripts\\gpu_python.cmd scripts\\diagnose_pose.py --data <去畸变数据目录>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow import paths  # noqa: E402

paths.apply_runtime_env()

from gsplat import rasterization  # noqa: E402

from gsflow.trainers.gsplat_trainer import _load_images, load_colmap_data  # noqa: E402


def render(pts, cols, c2w, K, w, h, radius):
    """把点云当极小高斯渲染出来。"""
    viewmat = torch.linalg.inv(c2w)[None].cuda()
    n = pts.shape[0]
    means = torch.nn.Parameter(pts)
    scales = torch.nn.Parameter(torch.full((n, 3), np.log(radius), device="cuda"))
    quats = torch.nn.Parameter(torch.zeros(n, 4, device="cuda"))
    quats.data[:, 0] = 1.0
    opac = torch.nn.Parameter(torch.full((n, 1), 4.0, device="cuda"))
    shs = torch.zeros(n, 1, 3, device="cuda")
    shs[:, 0, :] = (cols - 0.5) / 0.28209479177387814
    with torch.no_grad():
        colors, alphas, _ = rasterization(
            means=means, quats=quats, scales=torch.exp(scales),
            opacities=torch.sigmoid(opac).squeeze(-1), colors=shs,
            viewmats=viewmat, Ks=K[None].cuda(),
            width=w, height=h, sh_degree=0, packed=False, render_mode="RGB",
        )
    return colors[0], alphas[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "env" / "diagnose_pose"))
    args = ap.parse_args()

    data_dir = Path(args.data)
    imgs, masks = _load_images(data_dir, sorted(x.name for x in (data_dir / "images").iterdir()), 400)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    print(f"{'约定':<16}{'alpha 和':>12}{'可见像素':>10}{'PSNR(dB)':>10}")
    print("-" * 48)
    results = []
    for flip in (False, True):
        d = load_colmap_data(data_dir, flip_camera=flip)
        names = d["names"]
        imgs, _ = _load_images(data_dir, names, 400)
        c2ws = torch.as_tensor(d["c2ws"], dtype=torch.float32)
        Ks = torch.as_tensor(d["ks"], dtype=torch.float32)
        pts = torch.as_tensor(d["points"], dtype=torch.float32).cuda()
        cols = torch.as_tensor(d["colors"], dtype=torch.float32).cuda() / 255.0

        i = args.view
        gt = imgs[i].cuda()
        col, alpha = render(pts, cols, c2ws[i], Ks[i], gt.shape[1], gt.shape[0], 0.03)
        mse = float(((col - gt) ** 2).mean())
        psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
        tag = "不翻转" if not flip else "翻 y"
        print(f"{tag:<16}{float(alpha.sum()):>12.1f}{int((alpha > 0.01).sum()):>10}{psnr:>10.2f}")
        results.append((tag, psnr))
        a_np = (alpha.clamp(0, 1).cpu().numpy().reshape(gt.shape[0], gt.shape[1]) * 255).astype(np.uint8)
        strip = np.concatenate([
            (col.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
            (gt.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
            np.stack([a_np] * 3, axis=-1),
        ], axis=1)
        Image.fromarray(strip).save(out_dir / f"{tag.replace('+', 'z')}_render_gt_alpha.png")

    best = max(results, key=lambda r: r[1])
    print(f"\n更优的约定：{best[0]}（PSNR {best[1]:.2f} dB）")
    print(f"对比图已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
