"""测试"剪掉最弱的高斯"对画面的影响 —— 能否既减少雾状层又几乎不损画质。

`solid-v3` 的不透明度 p50=0.101、p90=0.126 ⇒ 绝大多数高斯几乎透明，靠多层叠加成像，
近距离/斜视角就会出现拖影与"发闷"。若把最弱的一批直接删掉，画面损失多少？
（在多个训练视角上算前景掩码 PSNR，并统计剪枝后不透明度分布。）
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
data = jd / "04_train" / "data"
ply = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
             key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))[-1]
g0 = read_gaussians_full(ply)

sp = data / "sparse"
rec = read_model(sp / "0" if (sp / "0" / "cameras.bin").exists() else sp)
imgs = sorted(rec.images, key=lambda im: im.name)
picks = np.linspace(0, len(imgs) - 1, 4).astype(int).tolist()


def subset(g, keep):
    out = {}
    for k, v in g.items():
        a = np.asarray(v) if hasattr(v, "__len__") and k != "sh_degree" else None
        if a is not None and a.ndim >= 1 and len(a) == len(keep):
            out[k] = a[keep]
        else:
            out[k] = v
    out["n"] = int(keep.sum())
    return out


def psnr_masked(a, b, m):
    mse = float(((a - b) ** 2)[m].mean())
    return 99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse)


op = np.asarray(g0["opacity"], np.float64).reshape(-1)
print(f"  {JOB}: n={len(op)}  op p50={np.median(op):.3f}")

for thresh in (0.0, 0.05, 0.1, 0.15):
    keep = op > thresh
    g = subset(g0, keep) if thresh > 0 else g0
    vals = []
    for k in picks:
        im = imgs[k]
        cam = rec.cameras[im.camera_id]
        R, c = im.cam_to_world()
        R = np.asarray(R, float); c = np.asarray(c, float)
        K = np.asarray(cam.intrinsics(), float)
        w2c = np.eye(4); w2c[:3, :3] = R.T; w2c[:3, 3] = -R.T @ c
        col, _ = render_view(g, w2c, K, cam.width, cam.height)
        col = np.asarray(col.detach().cpu() if hasattr(col, "detach") else col)[..., :3]
        with Image.open(data / "images" / im.name) as gi:
            gt = np.asarray(gi.convert("RGB"), dtype=np.float32) / 255.0
        if gt.shape[:2] != col.shape[:2]:
            continue
        m = gt.max(axis=2) > 6 / 255.0
        vals.append(psnr_masked(col, gt, m))
    kept = int(keep.sum())
    print(f"  op>{thresh:<5} \u5269\u4f59 {kept:>7} ({kept/len(op)*100:5.1f}%)  "
          f"\u975e\u9ed1\u63a9\u7801 PSNR={np.mean(vals):.2f} dB")
