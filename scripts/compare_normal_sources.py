"""对同一个模型比较两种"高斯法线"的质量，决定压平后该用哪一根轴。

背景（本轮的 2DGS 参数化工作）：
`stages/export.py::_gaussian_normals` 取的是四元数旋转后的 **+z 轴**（旋转矩阵第 3 列），
注释写明"3DGS 把高斯压扁对齐表面，所以 +z 朝向含表面信息"。
但如果模型是**压平过的**（2DGS 参数化），代表表面朝向的是**最短轴**（圆盘法线），
而不是 +z。

本脚本不跑导出的重活，直接在点云上比较两种法线的**邻域一致度**（越低翻转率越好），
用来回答"压平模型该用哪根轴"，避免盲改导出流程后再花十几分钟才发现更差。

用法：
  python scripts/compare_normal_sources.py --job img2-flat-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) -> [N,3,3]，列为三个轴（与训练侧 _quat_to_rotmat 同一约定）。"""
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def neighbour_coherence(pts: np.ndarray, nrm: np.ndarray, subsample: int = 60000,
                        k: int = 8) -> dict:
    """用 numpy 体素分桶找邻居，统计法线一致度。

    为什么不用 open3d KDTreeFlann：在百万级、含大量重合点的云上它会报
    `search_knn_vector_3d() error!` 且不说明原因（本项目踩过，见 docs/03）。
    """
    n = len(pts)
    if n > subsample:
        idx = np.linspace(0, n - 1, subsample).astype(np.int64)
        pts, nrm = pts[idx], nrm[idx]
        n = len(pts)
    # 体素边长取"平均最近邻距离"的粗估：用点数密度开三次方
    span = pts.max(0) - pts.min(0)
    vol = float(np.prod(np.maximum(span, 1e-9)))
    vox = (vol / max(n, 1)) ** (1.0 / 3.0) * 3.0
    keys = np.floor(pts / vox).astype(np.int64)
    buckets: dict[tuple, list[int]] = {}
    for i in range(n):
        buckets.setdefault(tuple(keys[i]), []).append(i)
    dots = []
    for i in range(n):
        cx, cy, cz = keys[i]
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cand.extend(buckets.get((cx + dx, cy + dy, cz + dz), ()))
        if len(cand) < 2:
            continue
        cand = np.asarray(cand, dtype=np.int64)
        d2 = ((pts[cand] - pts[i]) ** 2).sum(1)
        order = np.argsort(d2)[1:min(k + 1, len(cand))]
        nb = cand[order]
        if len(nb) == 0:
            continue
        dots.append(float((nrm[nb] * nrm[i]).sum(1).mean()))
    d = np.asarray(dots, dtype=np.float64)
    return {
        "pairs": int(len(d)),
        "mean_dot": float(d.mean()) if len(d) else 0.0,
        "median_dot": float(np.median(d)) if len(d) else 0.0,
        "flip_ratio": float((d < 0).mean()) if len(d) else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--subsample", type=int, default=60000)
    args = ap.parse_args()

    job = ROOT / "workspace" / args.job
    metrics = json.loads((job / "04_train" / "metrics.json").read_text(encoding="utf-8"))
    ply = Path(metrics["best_ply"])
    print(f"作业 {args.job}  检查点 {ply.parent.name}")
    d = PlyData.read(str(ply)).elements[0].data
    xyz = np.stack([np.asarray(d["x"]), np.asarray(d["y"]), np.asarray(d["z"])], 1)
    # open3d 写出的 PLY 可能带 nx/ny/nz（此处不用，我们比较的是"从旋转推算"的两种）
    sc = np.exp(np.stack([np.asarray(d["scale_0"]), np.asarray(d["scale_1"]),
                          np.asarray(d["scale_2"])], 1))
    qt = np.stack([np.asarray(d["rot_0"]), np.asarray(d["rot_1"]),
                   np.asarray(d["rot_2"]), np.asarray(d["rot_3"])], 1)
    R = quat_to_rotmat(qt)
    n = len(xyz)
    print(f"高斯 {n}  scale_min p50={np.median(sc.min(1)):.5f} "
          f"min/max p50={np.median(sc.min(1) / np.maximum(sc.max(1), 1e-12)):.4f}")

    # 候选 A：旋转后的 +z 轴（当前 _gaussian_normals 的做法）
    nz = R[:, :, 2]
    # 候选 B：最短轴（2DGS 语义：圆盘法线）
    ax = np.argmin(sc, axis=1)
    ns = R[np.arange(n), :, ax]
    # 候选 C：最长轴（对照，预期最差）
    axl = np.argmax(sc, axis=1)
    nl = R[np.arange(n), :, axl]

    results = {}
    for name, arr in (("+z 轴（当前实现）", nz),
                      ("最短轴（2DGS 语义）", ns),
                      ("最长轴（对照）", nl)):
        r = neighbour_coherence(xyz, arr, subsample=args.subsample)
        results[name] = r
        print(f"  {name:22s} mean_dot={r['mean_dot']:.4f}  "
              f"median={r['median_dot']:.4f}  翻转={r['flip_ratio'] * 100:.1f}%  "
              f"样本={r['pairs']}")

    best = max(results.items(), key=lambda kv: kv[1]["mean_dot"])
    print(f"\n邻域一致度最高 = {best[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
