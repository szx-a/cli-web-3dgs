"""证明"交互视图糊"= 放大，而不是渲染器：把 preview 的一块区域放大 3 倍，与同视角近距离渲染对比。

做法：
1. 取 `preview_front.png` 左半（512²，训练相机渲染）；
2. 裁中心 170×170 区域、用 LANCZOS 放大到 512²（模拟"放大 3 倍"）；
3. 用同一台相机、把距离缩到 1/3、渲染 512²（真正的近距离渲染）；
4. 比较两者的清晰度与互相 PSNR —— 若数值接近，说明"糊"来自放大，与渲染路径无关。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\3dgs-and-ai")
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from gsflow.gsdata.colmap_io import read_model  # noqa: E402
from gsflow.gsdata.mesh_export import scale_intrinsics  # noqa: E402
from gsflow.gsdata.shio import read_gaussians_full, render_view  # noqa: E402
from gsflow.paths import apply_runtime_env  # noqa: E402

apply_runtime_env()

JOB = sys.argv[1] if len(sys.argv) > 1 else "img5-solid-v3"
jd = ROOT / "workspace" / JOB
prev_path = jd / "05_export" / "preview_front.png"
prev = np.asarray(Image.open(prev_path).convert("RGB"), dtype=np.float32) / 255.0
left = prev[:, : prev.shape[1] // 2] if prev.shape[1] == prev.shape[0] * 2 else prev
H, W = left.shape[:2]
print(f"  preview \u5de6\u534a {W}x{H}")

# 1) \u88c1\u4e2d\u5fc3\u533a\u57df\u5e76\u653e\u5927 3 \u500d
crop = 170
cy, cx = H // 2, W // 2
region = left[cy - crop // 2: cy + crop // 2, cx - crop // 2: cx + crop // 2]
upscaled = np.asarray(Image.fromarray((region * 255).astype(np.uint8))
                      .resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0

# 2) \u540c\u4e00\u76f8\u673a\u3001\u8ddd\u79bb /3 \u6e32\u67d3
sp = jd / "04_train" / "data" / "sparse"
rec = read_model(sp / "0" if (sp / "0" / "cameras.bin").exists() else sp)
imgs = sorted(rec.images, key=lambda im: im.name)
im = imgs[0]
cam = rec.cameras[im.camera_id]
R, c = im.cam_to_world()
R = np.asarray(R, float)
c = np.asarray(c, float)
xyz = None
g = read_gaussians_full(sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
                              key=lambda p: int("".join(ch for ch in p.parent.name if ch.isdigit()) or 0))[-1])
xyz = np.asarray(g["xyz"], np.float64)
center = (xyz.min(0) + xyz.max(0)) / 2.0
c_new = center + (c - center) / 3.0          # \u8ddd\u79bb\u7f29\u5230 1/3
fwd = center - c_new
fwd /= np.linalg.norm(fwd)
up = R[:, 1] / np.linalg.norm(R[:, 1])
right = np.cross(fwd, up)
right /= np.linalg.norm(right)
up2 = np.cross(right, fwd)
w2c = np.eye(4)
w2c[0, :3], w2c[1, :3], w2c[2, :3] = right, up2, fwd
w2c[:3, 3] = -w2c[:3, :3] @ c_new
K = scale_intrinsics(np.asarray(cam.intrinsics(), float), (cam.width, cam.height), (W, H))
col, _ = render_view(g, w2c, K, W, H)
close = np.clip(np.asarray(col.detach().cpu() if hasattr(col, "detach") else col)[..., :3], 0, 1)


def sharp(x: np.ndarray) -> float:
    a = x.mean(axis=2)
    return float(np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean())


db = lambda m: 99.0 if m <= 1e-12 else 10 * np.log10(1 / m)
print(f"  \u653e\u5927 3\u00d7\u7684 preview \u5757   \u6e05\u6670\u5ea6={sharp(upscaled):.5f}")
print(f"  \u771f\u5b9e\u8fd1\u8ddd\u79bb\u6e32\u67d3\uff08/3\uff09  \u6e05\u6670\u5ea6={sharp(close):.5f}")
print(f"  \u4e24\u8005\u4e92\u76f8 PSNR = {db(float(((upscaled - close) ** 2).mean())):.2f} dB")
print(f"  \u539f\u59cb preview \u6e05\u6670\u5ea6={sharp(left):.5f}\uff08\u5bf9\u7167\uff09")

out = ROOT / "env" / "tmp" / "zoom_vs_close.png"
Image.fromarray((np.concatenate([upscaled, close], axis=1) * 255).astype(np.uint8)).save(out)
print(f"  \u5df2\u5199 {out}\uff08\u5de6=\u653e\u5927 3\u00d7 \u7684 preview\uff0c\u53f3=\u771f\u5b9e\u8fd1\u8ddd\u79bb\u6e32\u67d3\uff09")
