"""用 FastAPI TestClient 验证 `/api/render`（不依赖用户正在运行的服务进程）。

比较：
* 旧参数：`ss=1&aa=0&fov=45&minop=0`（等价于修复前的行为，但旋转矩阵已修正）
* 新默认：`ss=2&aa=1&fov=auto&minop=0.02`

断言：新图必须有内容（非纯背景）、并且清晰度更高。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\3dgs-and-ai")
sys.path.insert(0, str(ROOT / "src"))

from gsflow.paths import apply_runtime_env  # noqa: E402

apply_runtime_env()

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from gsflow.web.app import create_app  # noqa: E402

app = create_app()
client = TestClient(app)


def shot(query: str) -> tuple[np.ndarray, float]:
    r = client.get(f"/api/render/img5-solid-v3?{query}")
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    im = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.float32) / 255.0
    g = im.mean(axis=2)
    sharp = float(np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean())
    return im, sharp


old, s_old = shot("yaw=25&pitch=12&width=320&height=320&ss=1&aa=0&fov=45&minop=0")
new, s_new = shot("yaw=25&pitch=12&width=320&height=320")
print(f"  旧参数：亮度={old.mean():.4f} 清晰度={s_old:.5f}")
print(f"  新默认：亮度={new.mean():.4f} 清晰度={s_new:.5f}  提升={s_new/max(s_old,1e-9):.2f}×")
assert new.mean() > 0.02, "新渲染几乎全黑，说明相机或矩阵仍有问题"
assert s_new > s_old, "新默认并不比旧参数更清晰"

out = ROOT / "env" / "tmp" / "api_render_testclient.png"
Image.fromarray((np.concatenate([old, new], axis=1) * 255).astype(np.uint8)).save(out)
print(f"  已写 {out}")
print("  通过：/api/render 有内容且更清晰")
