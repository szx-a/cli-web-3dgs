"""逐视角掩码内 PSNR：判断误差是**系统性**还是**局部**。

## 为什么先测这个

我们的总量指标是：掩码内 PSNR **20.97 dB**、掩码内 SSIM **0.9477**。
"SSIM 高、PSNR 低"通常意味着**整体结构对、局部有错位**（细薄结构对不齐）。
但"错位"可能来自两种完全不同的原因，修法完全不同：

| 现象 | 含义 | 该做什么 |
|---|---|---|
| 各视角都在 20~22 dB（方差小）| **系统性**偏差（位姿整体偏、焦距/主点不准、颜色偏移）| 相机位姿优化 / 内参精化 |
| 少数视角明显低、多数很高 | 局部失败（遮挡、动态物体、掩码错）| 查那几张图 / 掩码 |

本脚本按视角分别渲染并算 PSNR（掩码内），给出分布与最差/最好视角。

用法: python scripts/diag_per_view_psnr.py --job img3-final --views 12
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
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--data-dir", default="")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env

    apply_runtime_env()
    if sys.modules.get("gsplat.csrc") is None:
        print("gsplat 内核未注册，跳过")
        return 0
    try:
        from gsplat.cuda._backend import _C as _k

        if _k is None:
            print("gsplat 内核未就绪，跳过")
            return 0
    except Exception:  # noqa: BLE001
        print("gsplat 不可用，跳过")
        return 0

    from PIL import Image

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.gsdata.shio import read_gaussians_full, render_view

    jd = ROOT / "workspace" / args.job
    data = Path(args.data_dir) if args.data_dir else jd / "04_train" / "data"
    plys = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"), key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0))
    if not plys:
        print("没有训练产物")
        return 1
    g = read_gaussians_full(plys[-1])

    roots = [data / "sparse", jd / "03_colmap" / "sparse"]
    cands = [p.parent for r in roots if r.exists() for p in r.rglob("cameras.bin")]
    if not cands:
        print("没有相机位姿")
        return 1
    rec = max((read_model(c) for c in sorted(set(cands))), key=lambda r: len(r.images))
    imgs = sorted(rec.images, key=lambda im: im.name)
    picks = np.linspace(0, len(imgs) - 1, args.views).astype(int).tolist()

    cam0 = rec.cameras[imgs[picks[0]].camera_id]
    W = cam0.width if args.width <= 0 else min(args.width, cam0.width)
    H = int(W * cam0.height / max(cam0.width, 1))

    print(f"=== 逐视角掩码内 PSNR：{args.job}（{len(picks)} 个视角）===")
    vals, names = [], []
    for k in picks:
        im = imgs[k]
        cam = rec.cameras[im.camera_id]
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
        # 真值（同一视角，来自训练数据 images/）
        gt_p = None
        for cand in (data / "images" / im.name,
                     data / "images" / (Path(im.name).stem + ".png")):
            if cand.exists():
                gt_p = cand
                break
        if gt_p is None:
            continue
        with Image.open(gt_p) as gi:
            gt = np.asarray(gi.convert("RGB").resize((W, H), Image.BILINEAR),
                            dtype=np.float64) / 255.0
        # 掩码内
        mp = None
        for cand in (data / "masks" / (im.name + ".png"),
                     data / "masks" / (Path(im.name).stem + ".png")):
            if cand.exists():
                mp = cand
                break
        if mp is None:
            continue
        with Image.open(mp) as mi:
            m = np.asarray(mi.convert("L").resize((W, H), Image.NEAREST)) > 127
        d = (col - gt) ** 2
        mse = float(d[m].mean()) if m.any() else float(d.mean())
        psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
        vals.append(psnr)
        names.append(im.name)
        print(f"  {im.name:<14} PSNR(掩码内) = {psnr:6.2f} dB   "
              f"覆盖 {float((a[m] > 0.5).mean())*100:5.1f}%")

    if vals:
        v = np.array(vals)
        print(f"\n  均值 {v.mean():.2f} dB   标准差 {v.std(ddof=0):.2f} dB"
              f"   min {v.min():.2f}（{names[int(v.argmin())]}）"
              f"   max {v.max():.2f}（{names[int(v.argmax())]}）")
        spread = float(v.max() - v.min())
        if v.std(ddof=0) < 1.0 and spread < 3.0:
            print("  判读：**各视角接近 ⇒ 系统性偏差**（位姿/内参/颜色），"
                  "优先考虑相机位姿优化；靠调高斯参数收益有限。")
        else:
            print("  判读：**视角间差异明显 ⇒ 局部失败**，先查最差的那几张"
                  "（遮挡/动态物体/掩码错），别急上全局优化。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
