"""对比 `/api/render` 的旧/新设置：证明"交互式服务端渲染为什么更糊"以及修好后的差别。

旧：FOV 固定 45°、无超采样、classic 光栅化、不过滤浮点
新：FOV 用训练内参（img5 是 24.2°）、2× 超采样 + LANCZOS、antialiased、过滤 opacity<=0.02

输出：清晰度（梯度能量）、与"训练视角预览"的相似度、并排图。
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

from gsflow.gsdata.shio import read_gaussians_full, render_view  # noqa: E402
from gsflow.paths import apply_runtime_env  # noqa: E402

apply_runtime_env()


def load(job: str) -> dict:
    jd = ROOT / "workspace" / job
    ply = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
                 key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))[-1]
    return read_gaussians_full(ply)


def build_camera(g: dict, yaw_deg: float, pitch_deg: float, dist: float,
                 fov_deg: float, width: int, height: int):
    xyz = np.asarray(g["xyz"], dtype=np.float64)
    mn, mx = xyz.min(0), xyz.max(0)
    center = (mn + mx) / 2.0
    radius = max(1e-6, float(np.linalg.norm(mx - mn) / 2.0))
    d = dist if dist > 0 else radius * 2.8
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    eye = center + np.array([cp * sy, sp, cp * cy]) * d
    fwd = center - eye
    fwd /= max(np.linalg.norm(fwd), 1e-12)
    up_world = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(fwd, up_world))) > 0.999:
        up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_world)
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(right, fwd)
    w2c = np.eye(4)
    w2c[0, :3], w2c[1, :3], w2c[2, :3] = right, up, fwd  # 正交旋转（旧代码写锐成 -fwd 导致全黑）
    w2c[:3, 3] = -w2c[:3, :3] @ eye
    f = 0.5 * height / math.tan(math.radians(fov_deg) / 2.0)
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)
    return w2c, K


def render(g: dict, yaw: float, pitch: float, dist: float, fov: float,
           width: int, height: int, ss: int, aa: bool, minop: float) -> np.ndarray:
    w2c, K = build_camera(g, yaw, pitch, dist, fov, width, height)
    Ks = K.copy()
    Ks[0, :] *= ss
    Ks[1, :] *= ss
    colors, alphas = render_view(g, w2c, Ks, width * ss, height * ss,
                                 rasterize_mode=("antialiased" if aa else "classic"),
                                 keep_min_opacity=minop)
    img = np.asarray(colors.detach().cpu() if hasattr(colors, "detach") else colors)[..., :3]
    a = np.asarray(alphas.detach().cpu() if hasattr(alphas, "detach") else alphas)
    a = a[..., 0] if a.ndim == 3 else a
    bg = np.array([0.02, 0.03, 0.045], dtype=np.float32)
    comp = img * a[..., None] + bg * (1.0 - a[..., None])
    comp = np.clip(comp, 0, 1)
    if ss > 1:
        comp = np.asarray(Image.fromarray((comp * 255).astype(np.uint8))
                          .resize((width, height), Image.LANCZOS), dtype=np.float32) / 255.0
    return comp


def sharpness(x: np.ndarray) -> float:
    g = x.mean(axis=2)
    return float(np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean())


def main() -> int:
    job = sys.argv[1] if len(sys.argv) > 1 else "img5-solid-v3"
    g = load(job)
    W, H = 720, 720
    yaw, pitch = 25.0, 12.0
    fov_new = 24.2  # 由 _job_fov_deg 实测

    old = render(g, yaw, pitch, 0, 45.0, W, H, ss=1, aa=False, minop=0.0)
    new = render(g, yaw, pitch, 0, fov_new, W, H, ss=2, aa=True, minop=0.02)
    print(f"  {job}: 旧 FOV45°/无 SSAA、新 FOV{fov_new}/2×SSAA+AA")
    print(f"  清晰度（梯度能量）：旧 {sharpness(old):.5f}  新 {sharpness(new):.5f}  "
          f"提升 {sharpness(new)/max(sharpness(old),1e-9):.2f}×")
    out = ROOT / "env" / "tmp" / "render_old_vs_new.png"
    side = np.concatenate([old, new], axis=1)
    Image.fromarray((side * 255).astype(np.uint8)).save(out)
    print(f"  已写 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
