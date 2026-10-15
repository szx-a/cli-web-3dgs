"""测量"反投影点到最近高斯"的距离分布，用来给法线迁移定**有依据的**搜索半径。

背景：把深度梯度法线换成高斯自带法线（翻转率 0.2% vs 17%）时，
需要决定"多远的点还算属于某个高斯"。我先用 `voxel*1.75` 只命中 3.9%、
`voxel*4` 只命中 5.3% —— 说明阈值不能拍脑袋，得看真实分布。

做法：用体素哈希近似最近邻（无需 KDTree），统计每个点到最近高斯中心的距离，
然后打印分位数，并给出"想要命中 X% 的点需要多大半径"。

用法: python scripts/probe_point_gaussian_distance.py --job flat-ab-base [--points 400000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--points", type=int, default=400000)
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="体素边长；0 = 用网格导出的 vds 估算")
    args = ap.parse_args()

    job = ROOT / "workspace" / args.job
    metrics = json.loads((job / "04_train" / "metrics.json").read_text(encoding="utf-8"))
    d = PlyData.read(str(metrics["best_ply"])).elements[0].data
    gx = np.stack([np.asarray(d["x"]), np.asarray(d["y"]), np.asarray(d["z"])], 1).astype(np.float64)
    print(f"高斯 {len(gx)}")

    # 反投影点不可得（需要渲染），这里用**高斯位置**近似"表面采样点"：
    # 它们本就贴在表面上，距离分布与反投影点同量级，足以定阈值量级。
    rng = np.random.default_rng(0)
    n = min(args.points, len(gx))
    pts = gx[rng.choice(len(gx), size=n, replace=False)]
    span = gx.max(0) - gx.min(0)
    if args.voxel > 0:
        voxel = args.voxel
    else:
        vol = float(np.prod(np.maximum(span, 1e-9)))
        voxel = (vol / max(len(gx), 1)) ** (1 / 3) * 3.0
    print(f"体素边长（估算/指定）= {voxel:.5f}   场景尺度 = {span.round(3).tolist()}")

    keys = np.floor(gx / voxel).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    acc = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(acc, inv, gx)
    np.add.at(cnt, inv, 1.0)
    cnt = np.maximum(cnt, 1.0)
    cell_p = acc / cnt[:, None]
    lut = {(int(uniq[i, 0]), int(uniq[i, 1]), int(uniq[i, 2])): i for i in range(len(uniq))}

    dists = np.full(n, np.inf)
    for i in range(n):
        cx, cy, cz = (int(v) for v in np.floor(pts[i] / voxel))
        best = np.inf
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                for dz in range(-4, 5):
                    j = lut.get((cx + dx, cy + dy, cz + dz))
                    if j is None:
                        continue
                    dd = float(((cell_p[j] - pts[i]) ** 2).sum())
                    if dd < best:
                        best = dd
        dists[i] = np.sqrt(best)

    fin = dists[np.isfinite(dists)]
    print(f"可查到最近桶的点: {len(fin)}/{n}")
    qs = [1, 5, 10, 25, 50, 75, 90, 99]
    print("距离分位数（单位：体素）:")
    for q in qs:
        v = float(np.percentile(fin, q)) if len(fin) else float("nan")
        print(f"  p{q:<3d} = {v:.4f}  (= {v / voxel:.2f} 体素)")
    print("\n若要命中 X% 的点，搜索半径需要:")
    for target in (50, 80, 90, 95, 99):
        if len(fin):
            v = float(np.percentile(fin, target))
            print(f"  {target}% -> {v:.4f}  (= {v / voxel:.2f} 体素)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
