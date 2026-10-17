"""回归测试：网格聚焦的包围盒**不能**把稀疏但真实的结构裁掉。

## 真实事故（用户："网络依然破碎"）

导出的网格只是中央一小块**扭曲碎片**，而高斯渲染是完整的。量化后：

    高斯 bbox            6.38 × 4.71 × 4.97
    网格 bbox            1.62 × 1.02 × 0.77   ← 只有 25% / 22% / 15%
    网格体积占物体体积   0.85%

根因：`auto_crop_bounds` 在按不透明度过滤之后**又按分位数裁了两端**
（`quantile=0.95/0.98`）。而 3DGS 把容量放在**细节密集**处：

    不透明度 > 0.3 的高斯 24,629 个，铺满 5.83 × 2.16 × 2.85
    但它们的 95% 分位框只有 1.67 × 1.05 × 0.49

屋檐/翼角面积大但纹理平滑 ⇒ 高斯数量少 ⇒ 按"点的分位数"裁一定先裁掉它们。
这与我在查看器取景上犯的错是同一个（docs/03 G113 之后又一次）。

修法：默认 `quantile=1.0`，**只用不透明度过滤，不按分位数裁**。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_sparse_far_structure_is_not_cropped_away() -> None:
    """少量远处真实点必须被框进去（模拟"稀疏的屋檐"）。"""
    from gsflow.gsdata.mesh_export import auto_crop_bounds

    rng = np.random.default_rng(0)
    # 90% 的点挤在一个 0.5 的小团里；10% 稀疏地铺在 6×6×4 的外壳上
    dense = rng.normal(0, 0.1, size=(900, 3))
    shell = np.column_stack([
        rng.uniform(-3, 3, 100), rng.uniform(-3, 3, 100), rng.uniform(-2, 2, 100)])
    pts = np.vstack([dense, shell])
    # 全部不透明度都很高 ⇒ 不透明度过滤不该救场，只有"不按分位数裁"能救
    op = np.full(len(pts), 0.9)

    lo, hi, info = auto_crop_bounds(pts, op, op_thresh=0.3, quantile=1.0)
    size = hi - lo
    assert size[0] > 5.0, f"x 方向被裁掉了（框 {size}）—— 稀疏结构又丢了"
    assert size[1] > 5.0, f"y 方向被裁掉了（框 {size}）"
    assert size[2] > 3.0, f"z 方向被裁掉了（框 {size}）"
    # 全部点都应在框内
    inside = np.all((pts >= lo) & (pts <= hi), axis=1)
    assert inside.all(), f"有 {int((~inside).sum())} 个点落在框外"

    # 对照：若显式要求 95% 分位，框会明显更小 —— 证明"分位数就是罪魁"
    lo95, hi95, _ = auto_crop_bounds(pts, op, op_thresh=0.3, quantile=0.95)
    assert np.all((hi95 - lo95) < size), \
        f"分位数应当更紧（{hi95-lo95} vs {size}），测试前提不成立"


def test_default_quantile_keeps_full_extent() -> None:
    """默认参数（不传 quantile）必须等于 quantile=1.0 的行为。"""
    from gsflow.gsdata.mesh_export import auto_crop_bounds

    rng = np.random.default_rng(1)
    pts = np.vstack([rng.normal(0, 0.1, size=(500, 3)),
                     np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])])
    op = np.full(len(pts), 0.9)
    lo, hi, _ = auto_crop_bounds(pts, op)          # 用默认值
    assert (hi - lo)[0] > 9.0, f"默认值仍在按分位数裁（框 {hi-lo}）"


def test_opacity_filter_still_works() -> None:
    """不透明度过滤必须保留：低不透明度的漂浮点仍要被排除。"""
    from gsflow.gsdata.mesh_export import auto_crop_bounds

    rng = np.random.default_rng(2)
    body = rng.normal(0, 0.3, size=(800, 3))
    floaters = rng.uniform(-50, 50, size=(200, 3))     # 远处漂浮
    pts = np.vstack([body, floaters])
    op = np.concatenate([np.full(800, 0.9), np.full(200, 0.01)])
    lo, hi, info = auto_crop_bounds(pts, op, op_thresh=0.3)
    size = hi - lo
    assert size.max() < 5.0, f"漂浮点没被过滤掉（框 {size}）"
    assert info["points_used"] == 800, info


@pytest.mark.skipif(
    not (ROOT / "workspace" / "img3-final" / "05_export" / "model_mesh.ply").exists(),
    reason="没有 img3-final 的网格产物")
def test_mesh_covers_the_models_opaque_gaussians() -> None:
    """真实产物检查：网格必须**包住**模型里"真正在画"的高斯，且面积不能塌。

    判据为什么这样定（而不是"网格 bbox ≈ 高斯 bbox"）：

    实测 `img3-final` 的高斯分布极度失衡 ——

        贡献 80% 视觉权重的高斯：4,301 / 94,598（4.5%）
        高斯尺度：中位最大轴 0.015，p95 0.064，**max 1.31**

    外观由**少数大尺度高斯**撑起来，它们数量少、位置分散；而 9 万个小高斯
    挤在中央雕花区。**按位置**建网格必然只覆盖那团核心 —— 这是分布本身的性质，
    不是网格算法漏了东西。所以判据取：

      1. 网格 bbox 必须包住 ≥95% 的"不透明度>0.3"高斯（不排除模型主体）；
      2. 表面积不能塌 —— 事故前 **1.72**、修好后 **18.10**（差 10 倍）。
    """
    o3d = pytest.importorskip("open3d")
    from gsflow.gsdata.shio import read_gaussians_full

    job = ROOT / "workspace" / "img3-final"
    plys = sorted((job / "04_train" / "point_cloud").glob("*/point_cloud.ply"))
    g = read_gaussians_full(plys[-1])
    xyz = np.asarray(g["xyz"], dtype=np.float64)
    op = np.asarray(g["opacity"], dtype=np.float64).reshape(-1)

    m = o3d.io.read_triangle_mesh(str(job / "05_export" / "model_mesh.ply"))
    v = np.asarray(m.vertices)
    lo, hi = v.min(0), v.max(0)

    sel = xyz[op > 0.3]
    assert len(sel) > 1000
    inside = np.all((sel >= lo) & (sel <= hi), axis=1)
    frac = float(inside.mean())
    assert frac > 0.95, (
        f"网格只包住了 {frac*100:.1f}% 的不透明高斯 ⇒ 聚焦框/重建把模型主体切掉了")

    area = float(m.get_surface_area())
    assert area > 8.0, (
        f"网格表面积只有 {area:.2f}（事故前 1.72、修好后 18.10）⇒ 又退化成碎片了")

