"""掩码内实体覆盖：模型到底把物体画全了没有。

## 为什么必须单独测

`psnr_masked` / `ssim_masked` 是**整幅掩码区域**的平均误差。一个只画了中央核心、
周围留空的模型，仍可能因为核心又大又平滑而拿到不错的平均值 —— 但它显式地漏掉了
四周结构（屋檐、翼角）。

这里直接统计：**掩码内 alpha>0.5 的像素占比**（= 物体被实体覆盖的比例），
以及掩码内 alpha<0.1 的占比（= 该有东西却什么也没有的"空洞"）。

实测参照（img3-final）：
* 掩码前景约占画面 57.8%；
* 若覆盖 ≈ 100% ⇒ 模型把物体画全了；
* 若明显 < 90% ⇒ 只画了核心（正是"网格是碎片"的根源）。

用法: python scripts/check_mask_coverage.py --job img3-final --views 4
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
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--width", type=int, default=456)
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
    plys = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"))
    if not plys:
        print("没有训练产物")
        return 1
    g = read_gaussians_full(plys[-1])
    data = Path(args.data_dir) if args.data_dir else jd / "04_train" / "data"
    roots = [data / "sparse", jd / "03_colmap" / "sparse"]
    cands = [p.parent for r in roots if r.exists() for p in r.rglob("cameras.bin")]
    rec = max((read_model(c) for c in sorted(set(cands))), key=lambda r: len(r.images))
    mdir = data / "masks"
    imgs = sorted(rec.images, key=lambda im: im.name)
    picks = np.linspace(0, len(imgs) - 1, args.views).astype(int).tolist()
    cam0 = rec.cameras[imgs[picks[0]].camera_id]
    W = min(args.width, cam0.width)
    H = int(W * cam0.height / max(cam0.width, 1))

    print(f"=== 掩码内实体覆盖：{args.job} ===")
    covs, holes = [], []
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
        a = np.asarray(alphas.detach().cpu().numpy()
                       if hasattr(alphas, "detach") else alphas)[..., 0]
        mp = None
        for cand in (mdir / (im.name + ".png"), mdir / (Path(im.name).stem + ".png")):
            if cand.exists():
                mp = cand
                break
        if mp is None:
            print(f"  {im.name}: 没有掩码")
            continue
        with Image.open(mp) as mm:
            m = np.asarray(mm.convert("L").resize((W, H), Image.NEAREST)) > 127
        if not m.any():
            continue
        cov = float((a[m] > 0.5).mean())
        hole = float((a[m] < 0.1).mean())
        covs.append(cov)
        holes.append(hole)
        print(f"  {im.name:<14} 掩码内覆盖 {cov*100:5.1f}%   "
              f"空洞(alpha<0.1) {hole*100:5.1f}%")
    if covs:
        mc = float(np.mean(covs))
        mh = float(np.mean(holes))
        print(f"\n  平均覆盖 {mc*100:.1f}%   平均空洞 {mh*100:.1f}%")
        ok = mc > 0.90 and mh < 0.10
        print(f"  [{'OK  ' if ok else 'FAIL'}] 掩码内覆盖 > 90% 且空洞 < 10%"
              f"   （覆盖 {mc*100:.1f}% / 空洞 {mh*100:.1f}%）")
        print("  判读: " + ("通过" if ok else "**不通过：模型没把物体画全**"))
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
