"""说明清楚 PSNR 的口径：为什么"同一模型"会同时出现 35 dB 和 17 dB。

* **35 dB**：`scripts/probe_lpips_new.py` —— 8 个留出视角、**原生训练分辨率**、
  前景掩码（黑底模式）、与论文口径一致。
* **17 dB**：`scripts/compare_preview_vs_interactive.py` 里为了快，把渲染和真值都压到 **512²**，
  掩码用"非黑像素"（含所有边缘像素）。分辨率骤降 + 边缘像素全算进误差 ⇒ 数值天然低。
  那里只用于**同一设置下横向比较**（1× vs 2× vs AA），不能当作绝对质量。

本脚本在同一台训练相机上分别用 512² 与原生分辨率算 PSNR，验证这个解释。
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
ply = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
             key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))[-1]
g = read_gaussians_full(ply)

sp = jd / "04_train" / "data" / "sparse"
rec = read_model(sp / "0" if (sp / "0" / "cameras.bin").exists() else sp)
imgs = sorted(rec.images, key=lambda im: im.name)


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    mse = float(((a - b) ** 2)[mask].mean())
    return 99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse)


def run(im, W: int, H: int) -> tuple[float, float]:
    cam = rec.cameras[im.camera_id]
    R, ctr = im.cam_to_world()
    w2c = np.eye(4)
    w2c[:3, :3] = np.asarray(R).T
    w2c[:3, 3] = -np.asarray(R).T @ np.asarray(ctr)
    K = scale_intrinsics(np.asarray(cam.intrinsics(), dtype=np.float64), (cam.width, cam.height), (W, H))
    colors, alphas = render_view(g, w2c, K, W, H)
    img = np.asarray(colors.detach().cpu() if hasattr(colors, "detach") else colors)[..., :3]
    a = np.asarray(alphas.detach().cpu() if hasattr(alphas, "detach") else alphas)
    a = a[..., 0] if a.ndim == 3 else a
    render = np.clip(img * a[..., None], 0, 1)

    gt_path = jd / "04_train" / "data" / "images" / im.name
    with Image.open(gt_path) as gi:
        gt = np.asarray(gi.convert("RGB").resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0
    # 口径 A：所有非黑像素（边缘全算）
    mask_all = gt.max(axis=2) > 6 / 255.0
    # 口径 B：腐蚀 3 像素（去掉轮廓边缘的抖动/抗锯齿差异）
    import scipy.ndimage as ndi
    mask_er = ndi.binary_erosion(mask_all, iterations=3)
    return psnr(render, gt, mask_all), psnr(render, gt, mask_er)


print(f"  {JOB}")
for (W, H) in ((512, 512), (1600, 900)):
    pa, pe = [], []
    for im in imgs[:4]:
        a, b = run(im, W, H)
        pa.append(a)
        pe.append(b)
    print(f"  {W}x{H}: 非黑掩码 PSNR={np.mean(pa):.2f} dB   腐蚀3px掩码 PSNR={np.mean(pe):.2f} dB")
