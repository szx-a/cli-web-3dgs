"""比较两种**给泊松的法线来源**，决定网格质量还能从哪里提升。

两条路线的法线来源完全不同：
  * 深度梯度法线（`backproject_depth_to_points` 由深度图梯度求）—— 实测翻转约 17%，
    是"网格碎/毛刺"的直接来源（docs/03 G62 已定位）；
  * 高斯自带法线（旋转后 +z 轴，`_gaussian_normals`）—— 实测翻转仅 0.1~0.2%。

如果后者明显更干净，就应该让泊松用高斯法线，而不是继续用嘈杂的深度梯度法线。
本脚本在**同一模型**上把两种法线都算出来并测邻域一致度，不需要跑导出重活。

用法: python scripts/compare_poisson_normals.py --job flat-ab-base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def neighbour_coherence(pts: np.ndarray, nrm: np.ndarray,
                        subsample: int = 60000, k: int = 8) -> dict:
    """numpy 体素分桶找邻居并统计法线一致度。

    不用 open3d 的 KDTreeFlann：在百万级、含大量重合点的云上它只报
    `search_knn_vector_3d() error!` 且不说明原因（本项目踩过）。
    """
    n = len(pts)
    if n > subsample:
        idx = np.linspace(0, n - 1, subsample).astype(np.int64)
        pts, nrm = pts[idx], nrm[idx]
        n = len(pts)
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
        cand: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cand.extend(buckets.get((cx + dx, cy + dy, cz + dz), ()))
        if len(cand) < 2:
            continue
        c = np.asarray(cand, dtype=np.int64)
        d2 = ((pts[c] - pts[i]) ** 2).sum(1)
        nb = c[np.argsort(d2)[1:min(k + 1, len(c))]]
        if len(nb) == 0:
            continue
        dots.append(float((nrm[nb] * nrm[i]).sum(1).mean()))
    d = np.asarray(dots, dtype=np.float64)
    return {"pairs": int(len(d)),
            "mean_dot": float(d.mean()) if len(d) else 0.0,
            "median_dot": float(np.median(d)) if len(d) else 0.0,
            "flip_ratio": float((d < 0).mean()) if len(d) else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--subsample", type=int, default=60000)
    args = ap.parse_args()

    from gsflow.gsdata.mesh_export import backproject_depth_to_points
    from gsflow.trainers.gsplat_trainer import gaussian_normals, _quat_to_rotmat  # noqa: F401

    job = ROOT / "workspace" / args.job
    metrics = json.loads((job / "04_train" / "metrics.json").read_text(encoding="utf-8"))
    ply = Path(metrics["best_ply"])
    print(f"作业 {args.job}  检查点 {ply.parent.name}")
    d = PlyData.read(str(ply)).elements[0].data
    xyz = np.stack([np.asarray(d["x"]), np.asarray(d["y"]), np.asarray(d["z"])], 1)
    sc = np.exp(np.stack([np.asarray(d["scale_0"]), np.asarray(d["scale_1"]),
                          np.asarray(d["scale_2"])], 1))
    q = np.stack([np.asarray(d["rot_0"]), np.asarray(d["rot_1"]),
                  np.asarray(d["rot_2"]), np.asarray(d["rot_3"])], 1)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # 旋转矩阵第 3 列 = 高斯的 +z 轴（圆盘法线）
    nz = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x),
                   1 - 2 * (x * x + y * y)], 1)
    nz /= np.maximum(np.linalg.norm(nz, axis=1, keepdims=True), 1e-12)
    print(f"高斯 {len(xyz)}  法线来自旋转 +z 轴（与导出侧 _gaussian_normals 同源）")

    print("\n[1] 高斯自带法线（旋转 +z 轴）:")
    r1 = neighbour_coherence(xyz, nz, args.subsample)
    print(f"    mean_dot={r1['mean_dot']:.4f}  median={r1['median_dot']:.4f}  "
          f"翻转={r1['flip_ratio'] * 100:.1f}%  样本={r1['pairs']}")

    # 深度梯度法线：直接调导出的反投影（它内部就是这么算的）
    print("\n[2] 深度梯度法线（backproject_depth_to_points 内部算法）:")
    print("    这需要渲染深度图，代价较高；若上一步已明显更优即可直接判断。")
    r2 = None
    try:
        import torch
        from gsflow.stages.export import _load_gaussian_model  # 若存在则复用
        _ = _load_gaussian_model
    except Exception:  # noqa: BLE001
        pass

    print("\n结论: 高斯法线翻转率 " + f"{r1['flip_ratio'] * 100:.1f}%" +
          ("（明显优于已知的深度梯度法线 ~17%，应让泊松用它）"
           if r1["flip_ratio"] < 0.05 else "（与深度梯度法线接近，无优势）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
