"""量化高斯的各向异性（"细丝感"的来源）。

若大量高斯被压成**细长针状**（最大轴/最小轴比值极大、且最小轴接近 0），
从新视角看就会像"细丝/毛发"而不是实心表面。常见成因：
* 强细节梯度损失（`lambda_grad`）逼着高斯收缩去拟合高频；
* 不透明度下限把"体积"压成"薄片"；
* 深度方差正则把表面压平 ⇒ 只剩细长条。

本脚本输出：各向异性比值分位数、最小轴分位数、以及"细丝占比"（ratio>100 且 min<1e-3）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\3dgs-and-ai")
sys.path.insert(0, str(ROOT / "src"))

from gsflow.gsdata.shio import read_gaussians_full  # noqa: E402
from gsflow.paths import apply_runtime_env  # noqa: E402

apply_runtime_env()


def latest(job: str) -> Path:
    jd = ROOT / "workspace" / job
    manifest = jd / "05_export" / "manifest.json"
    if manifest.exists():
        import json
        try:
            src = json.loads(manifest.read_text(encoding="utf-8")).get("source_ply")
            if src and Path(src).exists():
                return Path(src)
        except Exception:
            pass
    return sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
                  key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))[-1]


jobs = sys.argv[1:] or ["img5-solid-v3", "img5-v1", "img5-quality4"]
for job in jobs:
    jd = ROOT / "workspace" / job
    if not jd.exists():
        continue
    p = latest(job)
    g = read_gaussians_full(p)
    sc = np.asarray(g["scales"], dtype=np.float64).reshape(-1, 3)
    mn, mx = sc.min(axis=1), sc.max(axis=1)
    ratio = mx / np.maximum(mn, 1e-12)
    thin = (ratio > 100) & (mn < 1e-3)
    print(f"  {job:<18} n={len(sc):>7}  \u6700\u5927\u8f74p50={np.median(mx):.4f}  "
          f"\u6700\u5c0f\u8f74p50={np.median(mn):.5f}  "
          f"\u5404\u5411\u5f02\u6027 p50={np.median(ratio):7.1f} p90={np.percentile(ratio,90):8.1f}  "
          f"\u7ec6\u4e1d\u5360\u6bd4={thin.mean()*100:5.2f}%")
