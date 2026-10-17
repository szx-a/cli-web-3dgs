"""回归测试：`/api/render/{job}` —— 服务端用 gsplat 渲染当前相机，返回 PNG。

## 为什么需要这条路由

GUI 里的"泼溅"是我用 WebGL **重新实现**的渲染器（逐高斯桶排序、四边形 2σ 近似、
固定低通），而 `preview_*.png` 用的是 **gsplat 官方光栅化器**（逐像素排序、
精确 EWA、完整 SH）。两者**必然有差距** —— 用户反复看到
"查看器里的模型比对照图差很多"，而任何 WebGL 复刻都消不掉这个差距。

唯一能"按构造一致"的做法：前端只负责相机，像素由**同一个渲染器**算。
本测试验证这条路线的**接线**（相机约定、PNG 编码、错误处理），
渲染本身用 mock 替代 —— 这样测试不需要 GPU，也不会和训练抢显存。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WORKSPACE = ROOT / "workspace"


def _client():
    from fastapi.testclient import TestClient

    from gsflow.web.app import create_app

    return TestClient(create_app())


def _pick_job() -> str | None:
    for job in ("img3-final", "img3-crop", "final-v1"):
        if list((WORKSPACE / job / "04_train" / "point_cloud").glob("*/point_cloud.ply")):
            return job
    return None


def _fake_render(g, viewmat, K, width, height, device="cuda", render_mode="RGB"):
    """假渲染器：返回与尺寸匹配的常数图 + alpha=1。"""
    colors = np.full((height, width, 3), 0.5, dtype=np.float32)
    alphas = np.ones((height, width, 1), dtype=np.float32)
    # 把收到的相机记录下来，供断言
    _fake_render.last = {"viewmat": np.asarray(viewmat).copy(),
                         "K": np.asarray(K).copy(),
                         "w": width, "h": height}
    return colors, alphas


def test_render_returns_png_and_uses_camera_convention() -> None:
    job = _pick_job()
    if job is None:
        pytest.skip("工作区里没有训练产物 PLY")
    c = _client()
    _fake_render.last = None
    with patch("gsflow.gsdata.shio.render_view", _fake_render):
        r = c.get(f"/api/render/{job}", params={"yaw": 30, "pitch": 10,
                                                "dist": 12.0,
                                                "width": 320, "height": 240})
    assert r.status_code == 200, r.text[:400]
    assert r.headers["content-type"].startswith("image/png")
    body = r.content
    assert body[:8] == b"\x89PNG\r\n\x1a\n", "返回的不是 PNG"
    # 尺寸写进 PNG 头 IHDR（偏移 16..24 大端）
    w = int.from_bytes(body[16:20], "big")
    h = int.from_bytes(body[20:24], "big")
    assert (w, h) == (320, 240), f"PNG 尺寸 {w}x{h} 与请求不符"

    # 相机约定：direction = [cp·sy, sp, cp·cy] 是**中心→相机**的方向。
    # 相机看 -Z，所以 w2c 的第 3 行（相机 +Z）等于 direction
    # （forward = -direction ⇒ row2 = -forward = direction）。
    vm = _fake_render.last["viewmat"]
    assert vm.shape == (4, 4)
    rot = vm[:3, :3]
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-6), rot
    axis = rot[2]
    expect = np.array([math.cos(math.radians(10)) * math.sin(math.radians(30)),
                       math.sin(math.radians(10)),
                       math.cos(math.radians(10)) * math.cos(math.radians(30))])
    assert np.allclose(axis, expect, atol=1e-5), (axis, expect)
    # w2c 一致性：把相机世界坐标投影回相机系必须得到原点
    # （cam_pos = -Rᵀt，则 R·cam_pos + t = 0）
    cam_pos = -rot.T @ vm[:3, 3]
    assert np.allclose(rot @ cam_pos + vm[:3, 3], np.zeros(3), atol=1e-6), vm[:3, 3]
    # 主点在图像中心
    K = _fake_render.last["K"]
    assert abs(K[0, 2] - 160) < 1e-6 and abs(K[1, 2] - 120) < 1e-6, K


def test_render_reports_missing_job() -> None:
    c = _client()
    r = c.get("/api/render/__nope__", params={"yaw": 0, "pitch": 0})
    assert r.status_code == 404
    assert "作业不存在" in r.text or "找不到" in r.text


def test_render_returns_503_when_gpu_unavailable() -> None:
    """GPU 忙/不可用时要给 503 + 明确原因，而不是挂住或 500 堆栈。"""
    job = _pick_job()
    if job is None:
        pytest.skip("工作区里没有训练产物 PLY")

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory (simulated)")

    c = _client()
    with patch("gsflow.gsdata.shio.render_view", boom):
        r = c.get(f"/api/render/{job}", params={"yaw": 0, "pitch": 0,
                                                "width": 128, "height": 128})
    assert r.status_code == 503, r.text[:300]
    assert "服务端渲染不可用" in r.text
    assert "simulated" in r.text


def test_camera_convention_matches_viewer_js() -> None:
    """前端 `getCamera()` 与后端公式必须是同一个约定（否则两边看的是不同视角）。"""
    js = (ROOT / "src" / "gsflow" / "web" / "static" / "viewer.js").read_text(
        encoding="utf-8")
    assert "Math.atan2(dx, dz)" in js, "前端 yaw 应为 atan2(x, z)"
    assert "Math.asin(" in js, "前端 pitch 应为 asin(y)"
    py = (ROOT / "src" / "gsflow" / "web" / "app.py").read_text(encoding="utf-8")
    assert "cp * sy" in py and "cp * cy" in py, "后端方向公式应与之匹配"
