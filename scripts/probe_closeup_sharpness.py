"""近距离对比：不同高斯数量/尺度在"放大 3 倍"时谁更清楚。

用户的核心抱怨是"贴近看糊"。高斯越多、尺度越小，近距离时每个高斯覆盖的像素越少 ⇒ 越锐。
本脚本对多个作业：同一相机、距离 /1、/2、/3 渲染，统计清晰度与"每像素高斯数"代理量。
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

W = H = 512


def latest_ply(jd: Path) -> Path:
    return sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
                  key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))[-1]


def sharp(x: np.ndarray) -> float:
    a = x.mean(axis=2)
    return float(np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean())


for job in (sys.argv[1:] or ["img5-solid-v3", "img5-long"]):
    jd = ROOT / "workspace" / job
    if not jd.exists():
        continue
    g = read_gaussians_full(latest_ply(jd))
    xyz = np.asarray(g["xyz"], np.float64)
    center = (xyz.min(0) + xyz.max(0)) / 2.0
    sc = np.asarray(g["scales"], np.float64).reshape(-1, 3)

    sp = jd / "04_train" / "data" / "sparse"
    rec = read_model(sp / "0" if (sp / "0" / "cameras.bin").exists() else sp)
    im = sorted(rec.images, key=lambda im: im.name)[0]
    cam = rec.cameras[im.camera_id]
    R, c = im.cam_to_world()
    R = np.asarray(R, float)
    c = np.asarray(c, float)
    up = R[:, 1] / np.linalg.norm(R[:, 1])
    K = scale_intrinsics(np.asarray(cam.intrinsics(), float), (cam.width, cam.height), (W, H))

    row = []
    for div in (1.0, 2.0, 3.0):
        c_new = center + (c - center) / div
        fwd = center - c_new
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        up2 = np.cross(right, fwd)
        w2c = np.eye(4)
        w2c[0, :3], w2c[1, :3], w2c[2, :3] = right, up2, fwd
        w2c[:3, 3] = -w2c[:3, :3] @ c_new
        col, _ = render_view(g, w2c, K, W, H)
        img = np.clip(np.asarray(col.detach().cpu() if hasattr(col, "detach") else col)[..., :3], 0, 1)
        row.append(sharp(img))
    print(f"  {job:<16} n={len(xyz):>7}  \u5c3a\u5ea6p50={np.median(sc.max(1)):.4f}  "
          f"\u6e05\u6670\u5ea6 1x={row[0]:.5f}  2x={row[1]:.5f}  3x={row[2]:.5f}  "
          f"(3x/1x={row[2]/max(row[0],1e-9)*100:.0f}%)")
