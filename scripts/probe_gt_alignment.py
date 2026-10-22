"""渲染与真值的**朝向/配对**检验：GT 做 4 种变换，看哪种 PSNR 最高。

## 为什么

同一条渲染链，在 200 步模型上与训练器自报指标吻合（14.5 vs 15.5 dB），
到 30k 却相差 7~9 dB。若是**配对错**（拿错图 / 上下翻转 / 左右镜像），
低 PSNR 会一直存在且随细节增多而急剧恶化 —— 与观察一致。

判据：把 GT 分别做 原样 / 左右镜像 / 上下翻转 / 180°，
看 PSNR 是否在某个变换下**明显更高**。

用法: python scripts/probe_gt_alignment.py --job img3-final --views 4
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
    ap.add_argument("--data-dir", default="")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env

    apply_runtime_env()
    from PIL import Image

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.gsdata.shio import read_gaussians_full, render_view

    jd = ROOT / "workspace" / args.job
    data = Path(args.data_dir) if args.data_dir else jd / "04_train" / "data"
    g = read_gaussians_full(
        sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"), key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0))[-1])
    rec = max((read_model(c) for c in sorted(
        {p.parent for p in (data / "sparse").rglob("cameras.bin")})),
        key=lambda r: len(r.images))
    imgs = sorted(rec.images, key=lambda im: im.name)
    picks = np.linspace(0, len(imgs) - 1, args.views).astype(int).tolist()

    transforms = {
        "原样": lambda a: a,
        "左右镜像": lambda a: a[:, ::-1],
        "上下翻转": lambda a: a[::-1, :],
        "180°": lambda a: a[::-1, ::-1],
    }
    print(f"=== GT 变换检验：{args.job} ===")
    print(f"  {'视角':<14}" + "".join(f"{k:>10}" for k in transforms))
    best_counts = dict.fromkeys(transforms, 0)
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
        colors, _ = render_view(g, w2c, K, W, H)
        col = np.clip(np.asarray(
            colors.detach().cpu().numpy() if hasattr(colors, "detach") else colors
        )[..., :3], 0, 1)
        gp = None
        for cand in (data / "images" / im.name,
                     data / "images" / (Path(im.name).stem + ".png")):
            if cand.exists():
                gp = cand
                break
        if gp is None:
            continue
        with Image.open(gp) as gi:
            gt0 = np.asarray(gi.convert("RGB").resize((W, H), Image.LANCZOS),
                             dtype=np.float64) / 255.0
        mp = None
        for cand in (data / "masks" / (im.name + ".png"),
                     data / "masks" / (Path(im.name).stem + ".png")):
            if cand.exists():
                mp = cand
                break
        with Image.open(mp) as mi:
            m = np.asarray(mi.convert("L").resize((W, H), Image.NEAREST)) > 127
        vals = {}
        for name, fn in transforms.items():
            gt = fn(gt0)
            d = ((col - gt) ** 2)[m].mean()
            vals[name] = 10 * np.log10(1 / max(float(d), 1e-10))
        best = max(vals, key=vals.get)
        best_counts[best] += 1
        print(f"  {im.name:<14}" + "".join(f"{vals[k]:>10.2f}" for k in transforms)
              + f"   best={best}")
    print("\n  各变换成为最优的次数:", {k: v for k, v in best_counts.items() if v})
    print("  若'原样'总是最优 ⇒ 配对正确，问题不在朝向；")
    print("  若镜像/翻转明显更高 ⇒ 相机或图像朝向搞错了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
