"""同视角对比：背景色 + 不透明度过滤 对"细丝感"的影响。

假设（用户说"模型像由细丝组成、比 preview 糊"）：
* `preview_*.png` 是**白底**合成，低不透明度浮点被白色背景"洗掉"，看起来干净；
* 交互式渲染是**深色背景**，同样的浮点在暗底上显示为灰白细丝 ⇒ 观感像"毛发/细丝"。

验证：同一轨道视角、同 FOV/分辨率，分别渲
  (a) 白底、不过滤   (b) 白底、过滤 op>0.1   (c) 深底、不过滤   (d) 深底、过滤 op>0.1
并统计"背景区域的非零像素比例"（浮点多少）。
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
im = sorted(rec.images, key=lambda im: im.name)[0]
cam = rec.cameras[im.camera_id]
R, ctr = im.cam_to_world()
w2c = np.eye(4)
w2c[:3, :3] = np.asarray(R).T
w2c[:3, 3] = -np.asarray(R).T @ np.asarray(ctr)

W = H = 640
K = scale_intrinsics(np.asarray(cam.intrinsics(), dtype=np.float64),
                     (cam.width, cam.height), (W, H))

panels = []
for bg_name, bg in (("white", np.array([1, 1, 1], np.float32)),
                    ("dark", np.array([0.02, 0.03, 0.045], np.float32))):
    for minop in (0.0, 0.1, 0.3):
        colors, alphas = render_view(g, w2c, K, W, H, keep_min_opacity=minop)
        img = np.asarray(colors.detach().cpu() if hasattr(colors, "detach") else colors)[..., :3]
        a = np.asarray(alphas.detach().cpu() if hasattr(alphas, "detach") else alphas)
        a = a[..., 0] if a.ndim == 3 else a
        comp = np.clip(img + bg * (1 - a[..., None]), 0, 1)
        panels.append(comp)
        # \u80cc\u666f\u6d6e\u70b9\uff1aalpha \u5f88\u4f4e\u4f46\u975e\u96f6\u7684\u50cf\u7d20\u6bd4\u4f8b
        faint = float(((a > 1e-4) & (a < 0.05)).mean())
        print(f"  bg={bg_name:<5} minop={minop:<4}  alpha\u542b\u91cf>1e-4={float((a>1e-4).mean()):.3f}  "
              f"\u6de1\u8584\u6d6e\u70b9\u6bd4\u4f8b={faint*100:5.1f}%  \u5e73\u5747\u4eae\u5ea6={comp.mean():.3f}")

grid = np.concatenate([np.concatenate(panels[:3], axis=1),
                       np.concatenate(panels[3:], axis=1)], axis=0)
out = ROOT / "env" / "tmp" / "bg_vs_minop.png"
Image.fromarray((grid * 255).astype(np.uint8)).save(out)
print(f"  \u5df2\u5199 {out}\uff08\u4e0a\u6392=\u767d\u5e95\uff0c\u4e0b\u6392=\u6df1\u5e95\uff1b\u5217=minop 0/0.1/0.3\uff09")
