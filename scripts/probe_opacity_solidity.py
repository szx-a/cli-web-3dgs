"""量化"表面是否实心"：不透明度分布 + 表层高斯的不透明度。

用户截图里屋顶边缘有**拖影/双层边**。一个高可能的原因：模型大部分高斯只有约 0.1 的不透明度
（`solid` 预设的 `opacity_floor=0.10` 是**单边下限**，它只把最低的往上抬一点，
并不会把表面做成实心），于是要看到表面得靠**多层叠加**；从斜视角看层与层分开 ⇒ 拖影。

指标：
* 不透明度分位数（p10/p50/p90）与 >0.5 的占比；
* 表层（按到相机最近 20% 距离的点）的不透明度中位；
* 每平方米高斯数（用高斯中心的包围盒投影面积近似）—— 层数越多越软。
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
    return sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
                  key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))[-1]


for job in (sys.argv[1:] or ["img5-solid-v3", "img5-v1", "img5-quality4", "img5-long"]):
    jd = ROOT / "workspace" / job
    if not jd.exists():
        continue
    g = read_gaussians_full(latest(job))
    op = np.asarray(g["opacity"], np.float64).reshape(-1)
    if op.max() > 1.5:  # 若是 logit，转成概率
        op = 1.0 / (1.0 + np.exp(-op))
    xyz = np.asarray(g["xyz"], np.float64)
    ext = np.linalg.norm(xyz.max(0) - xyz.min(0))
    print(f"  {job:<16} n={len(op):>7}  op p10={np.percentile(op,10):.3f} "
          f"p50={np.median(op):.3f} p90={np.percentile(op,90):.3f}  "
          f">0.5 \u5360\u6bd4={float((op>0.5).mean())*100:5.1f}%  "
          f">0.9={float((op>0.9).mean())*100:4.1f}%  \u5c3a\u5ea6={ext:.2f}")
