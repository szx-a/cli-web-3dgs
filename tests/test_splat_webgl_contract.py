"""回归测试：高斯渲染的**交接契约**（自研渲染器已下线）。

## 历史与结论

我此前用 WebGL1 从零写了一个泼溅渲染器（`splat.js`），并为它写了 5 个测试
（32 位索引扩展、黑屏自检等）。但在里面连续查出两个真 bug
（`transpose()`、`OES_element_index_uint`）之后，用户的判断是对的：

> 光栅化你从头写没必要啊，而且也不好，你至少参考一下官方的啊，
> 不然我这里看出来，模型差会误导我

⇒ 高斯渲染已交给社区参考实现 `@mkkellogg/gaussian-splats-3d`（MIT），
`splat.js` 归档为 `splat.legacy.js` 且**不再被加载**。

所以原来的 5 个测试一并下线 —— 它们验证的是一个不该存在的实现。
但**教训要留住**，所以这里改为断言"交接契约"：

  1. 页面不再加载 `splat.js`；
  2. `app.js` 把高斯（`gauss3d` / `.splat`）路由到 `GsRefViewer`；
  3. `viewer.js` 遇到高斯时**明确报错**，而不是悄悄用点云画；
  4. `refviewer.js` 里不含任何自研着色器代码。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "gsflow" / "web" / "static"
PROBE = ROOT / "scripts" / "probe_viewer_runtime.mjs"


def test_legacy_rasterizer_is_not_loaded() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'src="/static/splat.js"' not in html, "页面仍在加载已下线的自研渲染器"
    assert not (STATIC / "splat.js").exists(), "splat.js 应已归档为 splat.legacy.js"
    assert (STATIC / "splat.legacy.js").exists(), "归档文件不见了（保留是为了追溯）"


def test_app_routes_gaussians_to_reference_viewer() -> None:
    src = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "GsRefViewer" in src
    assert "ref.mount(" in src
    # 高斯分支必须在进入 GsViewer 之前 return
    i_gauss = src.find("isGauss")
    i_viewer_load = src.find("window.GsViewer.load(")
    assert 0 < i_gauss < i_viewer_load, "高斯分支应排在 WebGL1 查看器之前"


def test_viewer_refuses_gaussian_inputs_explicitly() -> None:
    src = (STATIC / "viewer.js").read_text(encoding="utf-8")
    assert "参考渲染器" in src, "viewer.js 遇到高斯应给出去向明确的报错"
    assert "GsSplatRenderer" not in src, "viewer.js 仍在调用已下线的自研渲染器"


def test_refviewer_has_no_handwritten_shader() -> None:
    src = (STATIC / "refviewer.js").read_text(encoding="utf-8")
    for bad in ("gl_Position", "drawElementsInstanced", "aCorner", "shaderSource"):
        assert bad not in src, f"refviewer.js 里出现了自研着色器痕迹：{bad}"
    assert "GaussianSplats3D.Viewer" in src


@pytest.mark.skipif(shutil.which("node") is None, reason="没有 node")
def test_probe_reports_handoff_error_for_3dgs_ply() -> None:
    """无头探针：3DGS PLY 必须被拒绝，并指向参考渲染器。"""
    ply = None
    for job in ("img3-final", "img3-crop"):
        for p in sorted((ROOT / "workspace" / job / "04_train" / "point_cloud")
                        .glob("*/point_cloud.ply")):
            if b"f_dc_0" in p.read_bytes()[:4000]:
                ply = p
                break
        if ply:
            break
    if ply is None:
        pytest.skip("工作区里没有 3DGS PLY")
    r = subprocess.run(["node", str(PROBE), str(ply)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
    out = (r.stdout or "") + (r.stderr or "")
    assert "参考渲染器" in out, out[-600:]
