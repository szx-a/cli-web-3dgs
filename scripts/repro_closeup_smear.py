"""复现用户截图里的"特写 + 拖影/糊"，并测试几种抑制手段。

用户截图特征：贴近模型（只看到半个屋顶）、深色背景、表面像被**沿一个方向拖过**、
边缘发虚。候选原因：
1. 近距离放大：高斯只有约 1 像素大小，放大后必然糊（数据分辨率上限）；
2. 低不透明度、超大尺度的"雾状"高斯叠在表面 ⇒ 拖影；
3. 各向异性极强的长条高斯被斜视 ⇒ 条状拖影。

本脚本在近似视角（dist=0.5×半径、yaw 30°、pitch 15°）渲四组：
  (a) 原样   (b) 过滤 op<=0.05   (c) 过滤 op<=0.15   (d) 过滤 + 剔除超大高斯
并把结果拼图，方便肉眼判断哪种最接近"干净"。
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

JOB = sys.argv[1] if len(sys.argv) > 1 else "img5-solid-v3"
jd = ROOT / "workspace" / JOB
ply = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
             key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))[-1]
g0 = read_gaussians_full(ply)
xyz = np.asarray(g0["xyz"], np.float64)
mn, mx = xyz.min(0), xyz.max(0)
center = (mn + mx) / 2
radius = float(np.linalg.norm(mx - mn) / 2)

W = H = 640
yaw, pitch = 30.0, 15.0
d = radius * 0.55
eye = center + np.array([math.cos(math.radians(pitch)) * math.sin(math.radians(yaw)),
                         math.sin(math.radians(pitch)),
                         math.cos(math.radians(pitch)) * math.cos(math.radians(yaw))]) * d
fwd = center - eye
fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, np.array([0.0, 1.0, 0.0])); right /= np.linalg.norm(right)
up = np.cross(right, fwd)
w2c = np.eye(4)
w2c[0, :3], w2c[1, :3], w2c[2, :3] = right, up, fwd
w2c[:3, 3] = -w2c[:3, :3] @ eye
f = 0.5 * H / math.tan(math.radians(24.2) / 2)
K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float64)


def subset(minop: float, max_scale_mult: float = 0.0):
    g = dict(g0)
    op = np.asarray(g["opacity"], np.float64).reshape(-1)
    keep = op > minop
    if max_scale_mult > 0:
        sc = np.asarray(g["scales"], np.float64).reshape(-1, 3)
        keep &= sc.max(axis=1) < max_scale_mult * radius
    if keep.all():
        return g
    out = {}
    for k, v in g.items():
        a = np.asarray(v) if hasattr(v, "__len__") else None
        if a is not None and a.ndim >= 1 and len(a) == len(op) and k != "sh_degree":
            out[k] = a[keep]
        else:
            out[k] = v
    out["n"] = int(keep.sum())
    return out


variants = [("a 原样", 0.0, 0.0),
            ("b op>0.05", 0.05, 0.0),
            ("c op>0.15", 0.15, 0.0),
            ("d op>0.05+限尺度", 0.05, 0.5)]
panels = []
for name, minop, msm in variants:
    g = subset(minop, msm)
    colors, alphas = render_view(g, w2c, K, W, H)
    img = np.asarray(colors.detach().cpu() if hasattr(colors, "detach") else colors)[..., :3]
    a = np.asarray(alphas.detach().cpu() if hasattr(alphas, "detach") else alphas)
    a = a[..., 0] if a.ndim == 3 else a
    comp = np.clip(img + np.array([0.02, 0.03, 0.045], np.float32) * (1 - a[..., None]), 0, 1)
    panels.append(comp)
    grad = float(np.abs(np.diff(comp.mean(2), axis=1)).mean() + np.abs(np.diff(comp.mean(2), axis=0)).mean())
    print(f"  {name:<16} n={int(g.get('n') or len(np.asarray(g['xyz']))):>7}  "
          f"\u5e73\u5747\u4eae\u5ea6={comp.mean():.3f}  \u6e05\u6670\u5ea6={grad:.5f}")

grid = np.concatenate([np.concatenate(panels[:2], axis=1),
                       np.concatenate(panels[2:], axis=1)], axis=0)
out = ROOT / "env" / "tmp" / "closeup_variants.png"
Image.fromarray((grid * 255).astype(np.uint8)).save(out)
print(f"  \u5df2\u5199 {out}")
