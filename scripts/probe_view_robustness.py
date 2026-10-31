"""量化"离开训练视角多远就变糊" —— 回答"preview 清楚、交互糊"的机制。

假设：模型是**多层半透明叠出来的表面**（op 中位 0.101，约 10 层才够 65% 覆盖）。
在训练视角上 alpha 合成把它抹平，看起来正常；一旦相机偏转几度，层与层错位 ⇒ 画面变糊、
出现拖影/丝状。若如此，**清晰度应随角度偏移快速下降**。

做法：取若干训练相机，绕世界"上方向"与俯仰各旋转 0/1/2/5/10/20 度，
渲染后统计梯度能量（清晰度）与图像变化（相对 0° 的 PSNR）。清晰度随角度掉的越快，
说明视角越脆弱。
"""
from __future__ import annotations

import math
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
data = jd / "04_train" / "data"
ply = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
             key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))[-1]
g = read_gaussians_full(ply)

sp = data / "sparse"
rec = read_model(sp / "0" if (sp / "0" / "cameras.bin").exists() else sp)
imgs = sorted(rec.images, key=lambda im: im.name)
picks = [imgs[i] for i in np.linspace(0, len(imgs) - 1, 3).astype(int).tolist()]

W = H = 512
OFFSETS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]


def sharp(x: np.ndarray) -> float:
    g_ = x.mean(axis=2)
    return float(np.abs(np.diff(g_, axis=1)).mean() + np.abs(np.diff(g_, axis=0)).mean())


_XYZ = np.asarray(g["xyz"], np.float64)
CENTER = (_XYZ.min(0) + _XYZ.max(0)) / 2.0


def _rodrigues(axis: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], float)
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def render_at(im, d_yaw_deg: float, d_pitch_deg: float = 0.0) -> np.ndarray:
    """相机绕物体中心**公转** d_yaw 度（位置与朝向一起变），距离不变。"""
    cam = rec.cameras[im.camera_id]
    R, c = im.cam_to_world()
    R = np.asarray(R, float).copy()
    c = np.asarray(c, float).copy()
    up = R[:, 1] / max(np.linalg.norm(R[:, 1]), 1e-9)
    # 先绕 up 公转（推动相机位置）
    Rrot = _rodrigues(up, d_yaw_deg)
    c_new = CENTER + Rrot @ (c - CENTER)
    # 再按俯仰调整：绕世界坐标里与 up 垂直的轴旋转位置
    if abs(d_pitch_deg) > 1e-9:
        side = np.cross(up, c_new - CENTER)
        n = np.linalg.norm(side)
        if n > 1e-9:
            c_new = CENTER + _rodrigues(side / n, d_pitch_deg) @ (c_new - CENTER)
    # 重新对准中心
    fwd = CENTER - c_new
    fwd /= max(np.linalg.norm(fwd), 1e-12)
    right = np.cross(fwd, up)
    right /= max(np.linalg.norm(right), 1e-12)
    up2 = np.cross(right, fwd)
    w2c = np.eye(4)
    w2c[0, :3], w2c[1, :3], w2c[2, :3] = right, up2, fwd
    w2c[:3, 3] = -w2c[:3, :3] @ c_new
    K = scale_intrinsics(np.asarray(cam.intrinsics(), float), (cam.width, cam.height), (W, H))
    col, _ = render_view(g, w2c, K, W, H)
    col = np.asarray(col.detach().cpu() if hasattr(col, "detach") else col)[..., :3]
    return np.clip(col, 0, 1)


print(f"  {JOB}: 清晰度 vs 偏离训练视角的角度（3 个视角平均）")
base = {}
for off in OFFSETS:
    vals, psnrs = [], []
    for im in picks:
        img = render_at(im, off, 0.0)
        vals.append(sharp(img))
        if off == 0.0:
            base[id(im)] = img
        else:
            b = base[id(im)]
            mse = float(((img - b) ** 2).mean())
            psnrs.append(99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse))
    tag = f"  {off:5.1f}\u00b0  清晰度={np.mean(vals):.5f}"
    if psnrs:
        tag += f"   相对 0\u00b0 的 PSNR={np.mean(psnrs):5.2f} dB"
    print(tag)
