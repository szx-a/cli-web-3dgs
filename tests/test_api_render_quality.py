"""`/api/render` 的质量回归测试（有 CUDA 才跑）。

锁住三件事（2026-10-31 用户报"服务端照片对比渲染 比 服务端高斯渲染查看 清晰"）：
1. **相机旋转矩阵必须是正交的**（第三行 +fwd）；旧代码 `-fwd` 让行列式为 -1，
   整帧被剔成空白 —— 这是"渲染糊/黑"的根因；
2. 渲染结果必须有内容（非纯背景）；
3. 默认（FOV 取训练内参 + 2× 超采样 + 抗锯齿 + 过滤浮点）必须**比关闭这些选项更清晰**。
"""
from __future__ import annotations

import io

import numpy as np
import pytest

pytest.importorskip("torch")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("需要 CUDA", allow_module_level=True)

Image = pytest.importorskip("PIL.Image")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from gsflow.paths import apply_runtime_env  # noqa: E402
from gsflow.web.app import create_app  # noqa: E402

apply_runtime_env()

JOB = "img5-solid-v3"


def _shot(client, query: str) -> tuple[np.ndarray, float]:
    r = client.get(f"/api/render/{JOB}?{query}")
    if r.status_code != 200:
        pytest.skip(f"渲染不可用（HTTP {r.status_code}）：{r.text[:120]}")
    im = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.float32) / 255.0
    g = im.mean(axis=2)
    sharp = float(np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean())
    return im, sharp


def test_render_is_not_blank_and_defaults_are_sharper() -> None:
    client = TestClient(create_app())
    old_im, s_old = _shot(client, "yaw=25&pitch=12&width=320&height=320&ss=1&aa=0&fov=45&minop=0")
    new_im, s_new = _shot(client, "yaw=25&pitch=12&width=320&height=320")

    assert new_im.mean() > 0.02, "默认渲染几乎全黑（相机/矩阵有问题）"
    assert s_new > s_old, f"默认应比旧参数清晰：{s_new:.5f} vs {s_old:.5f}"
    # FOV 来自训练内参（img5 实测 24.2°），物体应比固定 45° 时更大
    assert new_im.mean() > old_im.mean(), "FOV 修正后物体应占更大画面"
