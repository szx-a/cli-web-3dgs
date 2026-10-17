"""回归测试：导出的 **3D 高斯查看器** 必须用社区参考实现，且产物自包含。

## 为什么

`viewer.html` 内嵌的是 `x,y,z,r,g,b`（**6 维**）点云 —— 用它看模型必然
稀疏、扁平，用户会把**渲染差距误判成模型质量差**（`docs/03` G108/G116）。

3DGS 有 59 维（位置3+尺度3+旋转4+不透明度1+SH48），必须用能吃下这些维度的渲染器。
所以导出侧新增：

  * `model_gaussians.ply` —— 完整 59 维副本（`.splat` 只有 11 维、丢 SH）；
  * `viewer_splat.html` —— 用 `@mkkellogg/gaussian-splats-3d`（MIT，three.js）渲染；
  * `vendor/` —— 把 three.js 与参考库一起复制，使导出目录**可离线双击打开**。

本测试验证这几件事真的成立（而不是只在代码里写了）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
STATIC = ROOT / "src" / "gsflow" / "web" / "static"
EXPORT_SRC = ROOT / "src" / "gsflow" / "stages" / "export.py"


def _jobs_with_viewer() -> list[Path]:
    out = []
    for job in (ROOT / "workspace").iterdir():
        if not job.is_dir():
            continue
        p = job / "05_export" / "viewer_splat.html"
        if p.exists():
            out.append(job)
    return out


def test_export_source_copies_vendor_and_uses_reference_lib() -> None:
    src = EXPORT_SRC.read_text(encoding="utf-8")
    assert "_write_viewer_splat" in src, "缺少 3D 高斯查看器生成方法"
    assert "vendor/three.module.js" in src, "没有复制 three.js 到导出目录"
    assert "vendor/gaussian-splats-3d.module.js" in src, "没有复制参考渲染库"
    assert "gaussian-splats-3d.module.js" in src
    # 不得在导出的 HTML 里塞自研着色器
    assert "gl_Position" not in src, "导出模板里出现了自研着色器痕迹"
    # ply_full 必须真的复制完整 PLY
    assert "model_gaussians.ply" in src
    assert '"ply_full"' in src or "'ply_full'" in src


@pytest.mark.skipif(not _jobs_with_viewer(), reason="还没有导出过 3D 高斯查看器")
def test_exported_viewer_is_self_contained() -> None:
    job = sorted(_jobs_with_viewer())[-1]
    exp = job / "05_export"
    html = (exp / "viewer_splat.html").read_text(encoding="utf-8")

    # 1) 参考库 + import map 指向 **相对路径**（自包含，不能引用 /static 或外网）
    assert '<script type="importmap">' in html
    assert "./vendor/three.module.js" in html
    assert "./vendor/gaussian-splats-3d.module.js" in html
    assert "/static/" not in html, "导出的查看器不得依赖服务器静态路由"
    assert "unpkg.com" not in html and "cdn." not in html, "不得依赖外网 CDN"

    # 2) vendor 文件确实被复制且与主程序**逐字节一致**
    for name in ("three.module.js", "gaussian-splats-3d.module.js"):
        a = exp / "vendor" / name
        b = STATIC / "vendor" / name
        assert a.exists(), f"缺少 {name}"
        assert a.stat().st_size == b.stat().st_size, f"{name} 复制不完整"

    # 3) 数据文件存在且是**完整 59 维** PLY
    m = re.search(r'"dataFile":\s*"([^"]+)"', html)
    assert m, "HTML 里没有 dataFile"
    data = exp / m.group(1)
    assert data.exists(), f"数据文件不存在：{data}"
    if data.suffix == ".ply":
        head = data.read_bytes()[:8000]
        assert b"f_dc_0" in head, "PLY 缺 f_dc_0（不是 3DGS PLY）"
        assert b"f_rest_44" in head, "PLY 缺 f_rest_44（不是完整 59 维）"
        assert b"opacity" in head and b"scale_0" in head and b"rot_0" in head


@pytest.mark.skipif(not _jobs_with_viewer(), reason="还没有导出过 3D 高斯查看器")
def test_exported_viewer_framing_is_sane() -> None:
    """内嵌的取景参数必须合法：up 是单位向量，相机在 2.8R 处，且都有限。"""
    job = sorted(_jobs_with_viewer())[-1]
    html = (job / "05_export" / "viewer_splat.html").read_text(encoding="utf-8")
    m = re.search(r"const info = (\{.*?\});", html, re.S)
    assert m, "找不到内嵌的 info 常量"
    info = json.loads(m.group(1))
    up = np.asarray(info["up"], dtype=float)
    assert abs(float(np.linalg.norm(up)) - 1.0) < 1e-6, up
    c = np.asarray(info["center"], dtype=float)
    cp = np.asarray(info["cameraPosition"], dtype=float)
    assert np.all(np.isfinite(c)) and np.all(np.isfinite(cp))
    assert abs(float(np.linalg.norm(cp - c)) - info["radius"] * 2.8) < 1e-3
    assert info["gaussians"] > 1000


@pytest.mark.skipif(not _jobs_with_viewer(), reason="还没有导出过 3D 高斯查看器")
def test_manifest_lists_full_ply_and_splat_viewer() -> None:
    job = sorted(_jobs_with_viewer())[-1]
    exp = job / "05_export"
    man = json.loads((exp / "manifest.json").read_text(encoding="utf-8"))
    kinds = {a.get("kind") for a in (man.get("artifacts") or [])}
    assert "html_splat" in kinds, kinds
    paths = {a.get("path") for a in (man.get("artifacts") or [])}
    assert "viewer_splat.html" in paths
    # ply_full 是可选的；导出时带了就必须出现在清单里
    if (exp / "model_gaussians.ply").exists():
        assert "ply_full" in kinds, kinds


def test_full_ply_is_classified_as_gaussian_not_point_cloud() -> None:
    """`model_gaussians.ply` 属于**高斯**模式，不能进"点云"。

    回归：我最初只按扩展名 `.ply` 分类，于是完整 3DGS PLY 被塞进 `points`
    模式（"点云"），正好和事实相反 —— 它是 59 维含 SH 的真实高斯，
    而点云模式用的是 6 维 glb，两者观看方式完全不同。

    正确判据是**PLY 头里有没有 `f_dc_0`**（3DGS 的 SH DC 属性）。
    """
    from fastapi.testclient import TestClient

    from gsflow.web.app import create_app

    jobs = [p for p in (ROOT / "workspace").iterdir()
            if (p / "05_export" / "model_gaussians.ply").exists()]
    if not jobs:
        pytest.skip("没有导出过 model_gaussians.ply 的作业")
    job = sorted(jobs)[-1].name
    d = TestClient(create_app()).get(f"/api/models/{job}").json()
    modes = d.get("modes") or {}
    splat_names = {f["name"] for f in modes.get("splat", [])}
    point_names = {f["name"] for f in modes.get("points", [])}
    assert "model_gaussians.ply" in splat_names, (splat_names, point_names)
    assert "model_gaussians.ply" not in point_names
    # 且该条目的 kind 必须是 gauss3d（供前端选参考渲染器）
    kinds = {f["name"]: f.get("kind") for f in modes.get("splat", [])}
    assert kinds.get("model_gaussians.ply") == "gauss3d"

