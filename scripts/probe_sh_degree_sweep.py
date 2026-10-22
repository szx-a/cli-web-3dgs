"""逐阶 SH 消融：判断"过曝 + 彩色镶边"是不是高阶 SH 造成的。

## 为什么

`probe_render_vs_gt.py` 显示：从 PLY 渲染出的图**过曝、带彩虹镶边**，而真值是正常灰色，
且训练器自己从**内存模型**渲染的 preview 是对的。差 7.6 dB 必须定位到具体环节。

读/写两端的 `f_rest` 布局都核对过（通道主序，一致）。所以逐阶截断 SH，看 PSNR 随阶数怎么变：

* 若 `sh_degree=0`（只用 DC）就正常 ⇒ **高阶 SH 系数有问题**（布局/尺度/符号）；
* 若 0 阶也过曝 ⇒ 问题在 DC 换算或颜色空间，与 SH 无关。

用法: python scripts/probe_sh_degree_sweep.py --job img3-final --view 0
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
    ap.add_argument("--width", type=int, default=456)
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
    print(f"  PLY sh_degree = {g.get('sh_degree')}   shs.shape = "
          f"{np.asarray(g['shs']).shape}")

    rec = max((read_model(c) for c in sorted(
        {p.parent for p in (data / "sparse").rglob("cameras.bin")})),
        key=lambda r: len(r.images))
    imgs = sorted(rec.images, key=lambda im: im.name)
    im = imgs[args.view]
    cam = rec.cameras[im.camera_id]
    W = min(args.width, cam.width)
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

    for cand in (data / "images" / im.name,
                 data / "images" / (Path(im.name).stem + ".png")):
        if cand.exists():
            gt_p = cand
            break
    with Image.open(gt_p) as gi:
        gt = np.asarray(gi.convert("RGB").resize((W, H), Image.BILINEAR),
                        dtype=np.float64) / 255.0
    for cand in (data / "masks" / (im.name + ".png"),
                 data / "masks" / (Path(im.name).stem + ".png")):
        if cand.exists():
            mp = cand
            break
    with Image.open(mp) as mi:
        m = np.asarray(mi.convert("L").resize((W, H), Image.NEAREST)) > 127

    print(f"  视角 {im.name}（{W}x{H}），真值亮度均值 {gt[m].mean():.3f}")
    print(f"  {'阶数':<6}{'PSNR(mask)':>12}{'渲染亮度':>10}{'亮度偏差':>10}")
    for deg in (0, 1, 2, 3):
        nb = (deg + 1) ** 2
        gg = dict(g)
        shs = np.asarray(g["shs"])[:, :nb, :].copy()
        gg["shs"] = shs
        gg["sh_degree"] = deg
        colors, alphas = render_view(gg, w2c, K, W, H)
        col = np.asarray(colors.detach().cpu().numpy()
                         if hasattr(colors, "detach") else colors)[..., :3]
        a = np.asarray(alphas.detach().cpu().numpy()
                       if hasattr(alphas, "detach") else alphas)[..., 0]
        v = np.clip(col, 0, 1)
        ps = 10 * np.log10(1 / max(float(((v - gt) ** 2)[m].mean()), 1e-10))
        print(f"  {deg:<6}{ps:>12.2f}{v[m].mean():>10.3f}"
              f"{v[m].mean()-gt[m].mean():>+10.3f}   alpha中位 {np.median(a):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
