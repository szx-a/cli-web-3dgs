"""把"我的渲染"与"真值"并排存成 PNG，用于判断 13 dB vs 21 dB 的差异来源。

## 背景

训练器报告 `psnr_masked = 20.97 dB`，而 `diag_per_view_psnr.py` 逐视角算出 **13.37 dB**。
差 7.6 dB 必须解释清楚 —— 否则我所有"自由视角"诊断都建立在错误渲染上。

两种可能：
1. **我的渲染确实错了**（SH 用法 / 颜色空间 / 合成方式 / 缩放）；
2. 训练器的 20.97 与本脚本测的不是同一件事（例如 GT 被处理过）。

并排图能一眼看出是哪一类：镜像、颜色偏移、尺度错位、或纯噪声。

用法: python scripts/probe_render_vs_gt.py --job img3-final --view 0
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
    ap.add_argument("--job", default="img3-final")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env

    apply_runtime_env()
    from PIL import Image

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.gsdata.shio import read_gaussians_full, render_view

    jd = ROOT / "workspace" / args.job
    data = jd / "04_train" / "data"
    g = read_gaussians_full(
        sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"), key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0))[-1])
    rec = max((read_model(c) for c in sorted(
        {p.parent for p in (data / "sparse").rglob("cameras.bin")})),
        key=lambda r: len(r.images))
    imgs = sorted(rec.images, key=lambda im: im.name)
    im = imgs[args.view]
    cam = rec.cameras[im.camera_id]

    W = cam.width if args.width <= 0 else min(args.width, cam.width)
    H = int(W * cam.height / cam.width)
    R, c = im.cam_to_world()
    R = np.asarray(R, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    K = np.asarray(cam.intrinsics(), dtype=np.float64).copy()
    K[0, :] *= W / cam.width
    K[1, :] *= H / cam.height
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ c
    colors, alphas = render_view(g, w2c, K, W, H)
    col = np.asarray(colors.detach().cpu().numpy()
                     if hasattr(colors, "detach") else colors)[..., :3]
    a = np.asarray(alphas.detach().cpu().numpy()
                   if hasattr(alphas, "detach") else alphas)[..., 0]

    gt_p = None
    for cand in (data / "images" / im.name,
                 data / "images" / (Path(im.name).stem + ".png")):
        if cand.exists():
            gt_p = cand
            break
    with Image.open(gt_p) as gi:
        gt = np.asarray(gi.convert("RGB").resize((W, H), Image.BILINEAR),
                        dtype=np.float64) / 255.0

    # 三种合成方式都算一遍，看哪个与训练器口径一致
    variants = {
        "raw": np.clip(col, 0, 1),
        "over_black": np.clip(col + 0.0 * (1 - a[..., None]), 0, 1),
        "over_white": np.clip(col + 1.0 * (1 - a[..., None]), 0, 1),
        "unpremul": np.clip(col / np.maximum(a[..., None], 1e-6), 0, 1),
    }
    mp = None
    for cand in (data / "masks" / (im.name + ".png"),
                 data / "masks" / (Path(im.name).stem + ".png")):
        if cand.exists():
            mp = cand
            break
    with Image.open(mp) as mi:
        m = np.asarray(mi.convert("L").resize((W, H), Image.NEAREST)) > 127

    print(f"=== {args.job} 视角 {im.name}（{W}x{H}）===")
    for name, v in variants.items():
        d = (v - gt) ** 2
        ps_m = 10 * np.log10(1 / max(float(d[m].mean()), 1e-10))
        ps_all = 10 * np.log10(1 / max(float(d.mean()), 1e-10))
        print(f"  {name:<11} PSNR 掩码内 {ps_m:6.2f} dB   全图 {ps_all:6.2f} dB")
    print(f"  alpha 中位 {np.median(a):.3f}；渲染亮度均值 {col[m].mean():.3f}；"
          f"真值亮度均值 {gt[m].mean():.3f}")

    # 存并排图：上排 渲染 / 真值；下排 差异 ×4
    diff = np.clip(np.abs(variants["raw"] - gt) * 4, 0, 1)
    grid = np.concatenate([
        np.concatenate([variants["raw"], gt], axis=1),
        np.concatenate([diff, np.stack([a] * 3, -1)], axis=1),
    ], axis=0)
    out = Path(args.out) if args.out else (
        ROOT / "env" / "tmp" / f"render_vs_gt_{args.job}_{args.view}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((grid * 255).astype(np.uint8)).save(out)
    print(f"  已写出并排图：{out}（上：渲染 | 真值；下：差异×4 | alpha）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
