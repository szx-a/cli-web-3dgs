"""量化"我们的数据能支持多细"：GSD（每像素多少毫米）、物体占多少像素。

## 为什么（用户反复问：为什么对比图上那么好）

参照图（社区 demo / 论文图）通常是**近距离、高分辨率、物体占满画面**的拍摄；
我们的素材是 457×289、物体只占画面一部分。这个差距可以**直接算**，不需要猜：

* `f_px`：COLMAP 内参的焦距（像素）；
* `d`：相机到物体中心的距离（米）；
* **GSD = d / f_px**（物体处每个像素覆盖多少米）；
* 物体在画面里的跨度（像素）；
* 用三角化点云的尺度核对。

输出与常见参照的对比结论。

用法: python scripts/diag_gsd_budget.py --job img3-final
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
    args = ap.parse_args()

    from gsflow.gsdata.colmap_io import read_model

    jd = ROOT / "workspace" / args.job
    data = Path(args.data_dir) if args.data_dir else jd / "04_train" / "data"
    roots = [data / "sparse", jd / "03_colmap" / "sparse"]
    cands = [p.parent for r in roots if r.exists() for p in r.rglob("cameras.bin")]
    if not cands:
        print("找不到 COLMAP 模型")
        return 1
    rec = max((read_model(c) for c in sorted(set(cands))), key=lambda r: len(r.images))
    if len(rec.points3D) == 0:
        print("没有三角化点")
        return 1
    pts = np.asarray(rec.points3D, dtype=np.float64)
    center = np.median(pts, axis=0)
    lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
    extent = float(np.linalg.norm(hi - lo))

    fs, dists, spans, sizes = [], [], [], []
    for im in rec.images:
        cam = rec.cameras[im.camera_id]
        K = np.asarray(cam.intrinsics(), dtype=np.float64)
        R, c = im.cam_to_world()
        R = np.asarray(R, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        f = 0.5 * (K[0, 0] + K[1, 1])
        d = float(np.linalg.norm(c - center))
        fs.append(f)
        dists.append(d)
        # 物体跨度（像素）：extent * f / d
        spans.append(extent * f / max(d, 1e-9))
        sizes.append((cam.width, cam.height))

    f_med = float(np.median(fs))
    d_med = float(np.median(dists))
    gsd = d_med / max(f_med, 1e-9)
    w, h = sizes[0]
    span = float(np.median(spans))

    print(f"=== 数据能力量化：{args.job} ===")
    print(f"  图像分辨率            : **{w} × {h}**")
    print(f"  镜头焦距 f_px         : {f_med:.0f} px（水平视场约 "
          f"{2*np.degrees(np.arctan(w/(2*f_med))):.1f}°）")
    print(f"  相机到物体中心距离 d  : **{d_med:.2f} m**")
    print(f"  ⇒ **GSD = d/f = {gsd*1000:.2f} mm / 像素**")
    print(f"  物体三维尺寸约        : {extent:.2f} m")
    print(f"  物体在画面中的跨度    : **约 {span:.0f} px**（画面宽 {w} px）")
    print(f"  三角化点数            : {len(pts):,}")
    print()
    print("  参照对比（同样以 1.5 m 物体为例）：")
    for label, dist, fpx, res in (
            ("社区 demo 近距高分辨率", 2.0, 1400, "1920×1080"),
            ("论文常用 4K 近距", 2.5, 2600, "3840×2160"),
            ("我们（img3）", d_med, f_med, f"{w}×{h}")):
        g = dist / fpx * 1000
        print(f"    {label:<22} d={dist:>5.1f}m  f={fpx:>5.0f}px  "
              f"{res:>10}  ⇒ GSD ≈ **{g:>5.2f} mm/px**")
    print()
    print(f"  结论：要追平 1 mm/px 级别，需要把 GSD 降到 {gsd*1000:.2f} 的 1/4~1/5 —— ")
    print("        也就是**靠近 4~5 倍**，或**焦距加长 4~5 倍**，或**分辨率提高 4~5 倍**。")
    print("        这三者在光学上是同一件事：让物体在传感器上占更多像素。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
