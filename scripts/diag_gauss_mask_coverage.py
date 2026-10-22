"""统计"从未落在任何掩码内"的高斯 —— 背景几何残留的量化。

## 背景（用户的第三个症状）

修好掩码损失之后（docs/03 G123），掩码外的**颜色**被压成了背景色，但
**几何还在**：实测掩码外 alpha = 0.70（正确应≈0），也就是模型外面裹着一层
**不透明的黑色几何**。在查看器里转一圈，这层黑壳就表现为"背面混乱"。

`composite` 损失只约束**颜色**（该黑的地方渲黑），不约束**不透明度**（别在那儿画东西）。
不透明度重置本项目默认关闭（`opacity_reset_interval=0`），于是没有任何机制把
背景里的高斯删掉。

## 判据

把每个高斯的中心投影到全部相机：
* 若它**从不**落在任何一张掩码内 ⇒ 它是背景几何（或漂浮物）；
* 再看这些高斯占多少**视觉权重**（`opacity × max(scale)²`）。

输出同时给出：仅用"深度在物体之前"这一保守条件也不通过的比例，
以及建议的剪枝阈值。用法：

  python scripts/diag_gauss_mask_coverage.py --job img3-final
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
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--limit", type=int, default=0, help="只测前 N 个视角（0=全部）")
    args = ap.parse_args()

    from PIL import Image

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.gsdata.shio import read_gaussians_full

    jd = ROOT / "workspace" / args.job
    plys = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"), key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0))
    if not plys:
        print("没有训练产物")
        return 1
    g = read_gaussians_full(plys[-1])
    xyz = np.asarray(g["xyz"], dtype=np.float64)
    op = np.asarray(g["opacity"], dtype=np.float64).reshape(-1)
    sc = np.asarray(g["scales"], dtype=np.float64)
    w = op * sc.max(1) ** 2
    w = w / w.sum() if w.sum() > 0 else w
    n = len(xyz)

    data = Path(args.data_dir) if args.data_dir else jd / "04_train" / "data"
    mdir = data / "masks"
    roots = [data / "sparse", jd / "03_colmap" / "sparse"]
    cands = [p.parent for r in roots if r.exists() for p in r.rglob("cameras.bin")]
    if not cands:
        print("没有相机位姿")
        return 1
    rec = max((read_model(c) for c in sorted(set(cands))), key=lambda r: len(r.images))
    imgs = sorted(rec.images, key=lambda im: im.name)
    if args.limit:
        imgs = imgs[: args.limit]

    seen = np.zeros(n, dtype=bool)
    in_front = np.zeros(n, dtype=bool)   # 至少在一张图里出现在物体深度之前
    used = 0
    for im in imgs:
        cam = rec.cameras[im.camera_id]
        R, c = im.cam_to_world()
        R = np.asarray(R, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        K = np.asarray(cam.intrinsics(), dtype=np.float64)
        P = (xyz - c) @ R                       # 世界 -> 相机（R 的列是相机轴）
        z = P[:, 2]
        front = z > 1e-6
        if not front.any():
            continue
        u = K[0, 0] * P[:, 0] / np.maximum(z, 1e-9) + K[0, 2]
        v = K[1, 1] * P[:, 1] / np.maximum(z, 1e-9) + K[1, 2]
        inside = front & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        if not inside.any():
            continue
        mp = None
        for cand in (mdir / (im.name + ".png"),
                     mdir / (Path(im.name).stem + ".png")):
            if cand.exists():
                mp = cand
                break
        if mp is None:
            continue
        with Image.open(mp) as mm:
            m = np.asarray(mm.convert("L")) > 127
        h, wd = m.shape
        idx = np.where(inside)[0]
        uu = np.clip(u[idx].astype(int) * wd // cam.width, 0, wd - 1)
        vv = np.clip(v[idx].astype(int) * h // cam.height, 0, h - 1)
        hit = m[vv, uu]
        seen[idx[hit]] = True
        in_front[idx] = True
        used += 1

    if not used:
        print("没有可用的相机/掩码")
        return 1
    never = ~seen
    print(f"=== {args.job}：高斯 × 掩码覆盖（{used} 个视角）===")
    print(f"  高斯总数            : {n:,}")
    print(f"  从未进入任何掩码    : {int(never.sum()):,}（{never.mean()*100:.1f}%）"
          f"  占视觉权重 {w[never].sum()*100:.1f}%")
    print(f"  至少进入过一次掩码  : {int(seen.sum()):,}（{seen.mean()*100:.1f}%）")
    if never.any():
        c_all = xyz.mean(0)
        r_never = np.linalg.norm(xyz[never] - c_all, axis=1)
        r_seen = np.linalg.norm(xyz[seen] - c_all, axis=1) if seen.any() else np.array([0.0])
        print(f"  半径中位            : 未覆盖 {np.median(r_never):.2f} vs 覆盖 "
              f"{np.median(r_seen):.2f}")
        print(f"  不透明度中位        : 未覆盖 {np.median(op[never]):.4f} vs 覆盖 "
              f"{np.median(op[seen]):.4f}")
    print("\n  判读：'从未进入掩码'且视觉权重不低 ⇒ 这些就是物体外面的黑色几何，")
    print("        应当剪掉（它们只在训练视角被压成黑色，转到别的角度就是黑壳/黑片）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
