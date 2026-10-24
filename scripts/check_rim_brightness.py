"""量化"轮廓白边"：轮廓内 2px 环带亮度 − 内部亮度。

## 为什么需要这个指标

用户反复反馈"模型有白边/发光"，但**所有掩码内指标都看不见它**：
那一圈在紧掩码之外，或在松掩码内被当成物体。实测（2026-10-24）：

* 旧模型在紧掩码上 `psnr_masked` **26.48 dB / SSIM 0.9615**（很好看）；
* 但渲染图上轮廓有一圈接近纯白的亮边；
* 根因：训练掩码偏大（`rim_ratio` 0.97，掩码边缘其实是背景 0.89 vs 内部 0.5）。

所以要一个**不依赖真值、不依赖掩码**的指标：只看渲染本身。
定义：`alpha>0.5` 为实心；腐蚀 2px 得内部；两者的差为轮廓环；
`rim_bright = mean(lum[rim]) - mean(lum[inner])`。

健康值应接近 0（轮廓不该比内部亮）；>0.10 就是肉眼可见的白边。

用法: python scripts/check_rim_brightness.py --jobs img3-final,ab-tight-g2d-r1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="img3-final")
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--band", type=int, default=2)
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env

    apply_runtime_env()
    from PIL import Image
    from scipy import ndimage as nd

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.gsdata.shio import read_gaussians_full, render_view

    print(f"=== 轮廓亮边检查（轮廓内 {args.band}px 环 vs 内部）===")
    print(f"  {'作业':<24}{'rim_bright':>12}{'rim 占比':>10}{'判读':>10}")
    for job in [j.strip() for j in args.jobs.split(",") if j.strip()]:
        jd = ROOT / "workspace" / job
        plys = list((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"))
        if not plys:
            print(f"  {job:<24}  没有训练产物")
            continue
        ply = max(plys, key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))
        g = read_gaussians_full(ply)
        data = Path(args.data_dir) if args.data_dir else jd / "04_train" / "data"
        if not (data / "sparse").exists():
            print(f"  {job:<24}  找不到 sparse/")
            continue
        rec = max((read_model(c) for c in sorted(
            {p.parent for p in (data / "sparse").rglob("cameras.bin")})),
            key=lambda r: len(r.images))
        imgs = sorted(rec.images, key=lambda im: im.name)
        picks = np.linspace(0, len(imgs) - 1, args.views).astype(int).tolist()
        vals, fracs = [], []
        for k in picks:
            im = imgs[k]
            cam = rec.cameras[im.camera_id]
            W, H = cam.width, cam.height
            R, c = im.cam_to_world()
            R = np.asarray(R, dtype=np.float64)
            c = np.asarray(c, dtype=np.float64)
            K = np.asarray(cam.intrinsics(), dtype=np.float64).copy()
            w2c = np.eye(4)
            w2c[:3, :3] = R.T
            w2c[:3, 3] = -R.T @ c
            colors, alphas = render_view(g, w2c, K, W, H)
            col = np.asarray(colors.detach().cpu().numpy()
                             if hasattr(colors, "detach") else colors)[..., :3]
            a = np.asarray(alphas.detach().cpu().numpy()
                           if hasattr(alphas, "detach") else alphas)[..., 0]
            solid = a > 0.5
            if not solid.any():
                continue
            inner = nd.binary_erosion(solid, iterations=args.band)
            rim = solid & ~inner
            if not (inner.any() and rim.any()):
                continue
            lum = np.clip(col, 0, 1).mean(2)
            vals.append(float(lum[rim].mean() - lum[inner].mean()))
            fracs.append(float(rim.sum() / max(solid.sum(), 1)))
        if not vals:
            print(f"  {job:<24}  没有可用视角")
            continue
        v = float(np.mean(vals))
        f = float(np.mean(fracs))
        verdict = ("**有白边**" if v > 0.10 else "轻微" if v > 0.05 else "正常")
        print(f"  {job:<24}{v:>12.3f}{f*100:>9.1f}%{verdict:>10}")
    print("\n  判读：rim_bright 应接近 0；>0.10 即肉眼可见的轮廓白边/发光。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
