"""量化网格的"细丝感"：细长三角形（sliver）与尖刺比例。

用户描述"模型像由细丝组成"。除了高斯渲染，另一个候选是**网格**：TSDF 在薄结构处
会产生大量**细长三角形**（最小角极小/边长比极大），或沿法线方向的**尖刺**。
这些面片着色后看起来就是"丝/毛刺"。

指标：
* sliver：最小角 < 5° 或 最长边/最短边 > 8 的三角形占比；
* spike：某个顶点到另两点连线的距离 < 0.02×最长边（近似退化尖刺）；
* 边长统计（中位/95 分位）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\3dgs-and-ai")
sys.path.insert(0, str(ROOT / "src"))


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, tris = [], []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:4]]
                if len(idx) == 3:
                    tris.append(idx)
    return np.asarray(verts, np.float64), np.asarray(tris, np.int64)


def stats(job: str) -> None:
    jd = ROOT / "workspace" / job
    obj = jd / "05_export" / "model_mesh.obj"
    if not obj.exists():
        print(f"  {job}: \u7f3a\u5c11 model_mesh.obj")
        return
    v, f = load_obj(obj)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    e = np.stack([np.linalg.norm(b - a, axis=1),
                  np.linalg.norm(c - b, axis=1),
                  np.linalg.norm(a - c, axis=1)], axis=1)
    longest = e.max(axis=1)
    shortest = np.maximum(e.min(axis=1), 1e-12)
    # \u6700\u5c0f\u89d2\uff08\u5bf9\u6700\u77ed\u8fb9\uff09
    order = np.sort(e, axis=1)
    # \u4f59\u5f26\u5b9a\u7406\uff1acos C = (a^2+b^2-c^2)/2ab\uff0c\u6700\u5c0f\u89d2\u5bf9\u5e94\u6700\u77ed\u8fb9
    x, y, z = order[:, 0], order[:, 1], order[:, 2]
    cos_min = np.clip((y * y + z * z - x * x) / np.maximum(2 * y * z, 1e-12), -1, 1)
    min_angle = np.degrees(np.arccos(cos_min))
    sliver = (min_angle < 5.0) | (longest / shortest > 8.0)
    print(f"  {job:<16} \u9762={len(f):>7}  \u8fb9\u957f p50={np.median(longest):.4f} p95={np.percentile(longest,95):.4f}  "
          f"\u6700\u5c0f\u89d2 p50={np.median(min_angle):5.1f}\u00b0  "
          f"\u7ec6\u957f\u4e09\u89d2\u5f62={sliver.mean()*100:5.2f}%\uff08\u5176\u4e2d\u6700\u5c0f\u89d2<1\u00b0={float((min_angle<1).mean())*100:.2f}%\uff09")


jobs = sys.argv[1:] or ["img5-solid-v3", "img5-long", "img5-lp10"]
for j in jobs:
    stats(j)
