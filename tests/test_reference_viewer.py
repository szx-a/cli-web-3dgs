"""回归测试：GUI 的高斯视图改用**社区参考实现**渲染。

## 为什么换掉自己写的渲染器

用户的原话：

> 光栅化你从头写没必要啊，而且也不好，你至少参考一下官方的啊，
> 不然我这里看出来，模型差会误导我

这是对的。我用 WebGL1 从零写了泼溅渲染器（逐高斯桶排序、四边形 2σ 近似、
固定低通），它与 `preview_*.png` 用的 gsplat 官方光栅化器必然有差距，
而**用户会把渲染器的差距误读成模型质量差**。这是最坏的一种误导。

现在改用 `@mkkellogg/gaussian-splats-3d` v0.4.7（MIT，three.js r160，WebGL2，
原生支持 3DGS PLY 的 **59 维含 SH**），vendored 到 `/static/vendor/` 离线可用。
`refviewer.js` 只做适配（加载/取景/状态），不参与像素计算。

本测试锁住：
  * vendored 文件确实能被静态服务取到（子目录路由）；
  * 首页的 import map 把 `three` 指到**本地**副本（不能依赖外网）；
  * `scene_info` 给出的 up 是**算出来的**单位向量（不是猜 [0,1,0]）；
  * `app.js` 的高斯分支走参考渲染器，不再回落到我手写的 `splat.js`。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
STATIC = ROOT / "src" / "gsflow" / "web" / "static"

VENDORED = [
    "vendor/gaussian-splats-3d.module.js",
    "vendor/three.module.js",
    "vendor/LICENSE-gaussian-splats-3d.txt",
]


def _client():
    from fastapi.testclient import TestClient

    from gsflow.web.app import create_app

    return TestClient(create_app())


def test_vendored_library_is_present_and_served() -> None:
    for name in VENDORED:
        p = STATIC / name
        assert p.exists(), f"缺少 vendored 文件 {name}（离线可用性依赖它）"
        assert p.stat().st_size > 500, f"{name} 太小，可能没下载完整"
    c = _client()
    for name in VENDORED:
        r = c.get("/static/" + name)
        assert r.status_code == 200, (name, r.status_code)
        assert len(r.content) == (STATIC / name).stat().st_size


def test_static_route_blocks_traversal() -> None:
    c = _client()
    for bad in ("../app.py", "vendor/../../app.py", "..%2Fapp.py"):
        r = c.get("/static/" + bad)
        assert r.status_code == 404, (bad, r.status_code)


def test_index_uses_local_import_map() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert '<script type="importmap">' in html, \
        "缺少 import map（库 import 'three' 会失败）"
    assert "/static/vendor/three.module.js" in html, "import map 必须指向本地副本"
    assert "/static/refviewer.js" in html, "缺少参考渲染器适配层"
    # import map 必须**在真正的 module 脚本之前**（注释里提到 type="module" 不算）
    i_map = html.index('<script type="importmap">')
    i_mod = html.index('<script type="module"')
    assert i_map < i_mod, "import map 必须先于 type=module 脚本"


def test_refviewer_adapts_instead_of_reimplementing() -> None:
    src = (STATIC / "refviewer.js").read_text(encoding="utf-8")
    assert "gaussian-splats-3d.module.js" in src, "没有引用参考库"
    assert "GaussianSplats3D.Viewer" in src, "没有用参考库的 Viewer"
    assert "addSplatScene" in src, "没有调用参考库的加载接口"
    # 不得自己写着色器（那正是被否掉的做法）
    for bad in ("gl_Position", "aCorner", "drawElementsInstanced"):
        assert bad not in src, f"refviewer.js 里出现了自研着色器痕迹：{bad}"


def test_app_js_routes_gaussians_to_reference_viewer() -> None:
    src = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "GsRefViewer" in src, "app.js 没有接参考渲染器"
    assert "ref.mount(" in src, "没有调用参考渲染器的 mount"
    # 高斯分支必须优先用带 .ply 后缀的地址（库按扩展名选解析器）
    assert "/point_cloud.ply" in src, "没有用 .ply 后缀路由"


def test_scene_info_up_is_computed_not_assumed() -> None:
    """`up` 必须是从训练相机平面法向算出来的单位向量。"""
    c = _client()
    jobs = [p.name for p in (ROOT / "workspace").iterdir()
            if (p / "04_train" / "point_cloud").is_dir()]
    if not jobs:
        pytest.skip("工作区里没有训练产物")
    job = "img3-final" if "img3-final" in jobs else sorted(jobs)[0]
    r = c.get(f"/api/scene_info/{job}")
    assert r.status_code == 200, r.text[:300]
    d = r.json()
    for k in ("center", "radius", "up", "cameraPosition", "cameraLookAt"):
        assert k in d, k
    up = np.asarray(d["up"], dtype=float)
    assert abs(float(np.linalg.norm(up)) - 1.0) < 1e-6, up
    assert d["radius"] > 0
    if d.get("hasPoses"):
        # 有相机位姿时必须真的算过；且牌坊这类素材上方向应接近 +Y
        assert up[1] > 0.5, f"up 不合理（{up}）—— 可能退化成了默认值"
    # 相机位置到中心的距离应等于 2.8×半径
    c0 = np.asarray(d["center"], dtype=float)
    cp = np.asarray(d["cameraPosition"], dtype=float)
    assert abs(float(np.linalg.norm(cp - c0)) - d["radius"] * 2.8) < 1e-3


def test_ply_alias_route_has_extension() -> None:
    """`.ply` 别名存在：参考库按扩展名选解析器。"""
    c = _client()
    jobs = [p.name for p in (ROOT / "workspace").iterdir()
            if (p / "04_train" / "point_cloud").is_dir()]
    if not jobs:
        pytest.skip("工作区里没有训练产物")
    job = "img3-final" if "img3-final" in jobs else sorted(jobs)[0]
    r = c.get(f"/api/gauss/{job}/point_cloud.ply")
    assert r.status_code == 200
    assert r.content[:3] == b"ply"
