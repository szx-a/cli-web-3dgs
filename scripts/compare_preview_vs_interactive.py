"""对比"导出预览"与"交互式服务端渲染"两条路径的质量（同一台训练相机）。

路径 A（preview_*.png）：512²、classic 光栅化、1× 采样、不过滤浮点
路径 B（/api/render 新默认）：同相机、ss=2 超采样后缩到 512²、antialiased、过滤 op<=0.02

指标：
* PSNR(A,B)：两条路径彼此差多少；
* 各自与**真值图**的掩码 PSNR（谁更接近真值才是关键）；
* 梯度能量（清晰度）。
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

W = H = 512


def render_cam(im, ss: int, aa: bool, minop: float) -> np.ndarray:
    cam = rec.cameras[im.camera_id]
    R, ctr = im.cam_to_world()
    w2c = np.eye(4)
    w2c[:3, :3] = np.asarray(R).T
    w2c[:3, 3] = -np.asarray(R).T @ np.asarray(ctr)
    K = scale_intrinsics(np.asarray(cam.intrinsics(), dtype=np.float64),
                         (cam.width, cam.height), (W, H))
    Ks = K.copy()
    Ks[0, :] *= ss
    Ks[1, :] *= ss
    colors, alphas = render_view(g, w2c, Ks, W * ss, H * ss,
                                 rasterize_mode=("antialiased" if aa else "classic"),
                                 keep_min_opacity=minop)
    img = np.asarray(colors.detach().cpu() if hasattr(colors, "detach") else colors)[..., :3]
    a = np.asarray(alphas.detach().cpu() if hasattr(alphas, "detach") else alphas)
    a = a[..., 0] if a.ndim == 3 else a
    # ✗ 之前用白底合成，而真值图是**黑底** ⇒ PSNR 只有 16 dB。改为黑底。
    comp = np.clip(img * a[..., None], 0, 1)
    if ss > 1:
        comp = np.asarray(Image.fromarray((comp * 255).astype(np.uint8))
                          .resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0
    return comp, a


def sharp(x: np.ndarray) -> float:
    g_ = x.mean(axis=2)
    return float(np.abs(np.diff(g_, axis=1)).mean() + np.abs(np.diff(g_, axis=0)).mean())


def psnr_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    mse = float(((a - b) ** 2)[mask].mean())
    return 99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse)


VARIANTS = {
    "A preview":     (1, False, 0.0),
    "E interactive": (1, False, 0.02),
}
res = {k: {"psnr": [], "sharp": []} for k in VARIANTS}
cache = {}
for im in imgs[:4]:
    gt_path = jd / "04_train" / "data" / "images" / im.name
    if not gt_path.exists():
        continue
    gt = np.asarray(Image.open(gt_path).convert("RGB").resize((W, H)), dtype=np.float32) / 255.0
    mask = (gt.max(axis=2) > 6 / 255.0)
    for name, (ss, aa, mo) in VARIANTS.items():
        img, _ = render_cam(im, ss=ss, aa=aa, minop=mo)
        res[name]["psnr"].append(psnr_masked(img, gt, mask))
        res[name]["sharp"].append(sharp(img))
        if name == "A preview":
            cache["a"] = img
        if name == "E interactive":
            cache["c"] = img
    cache["gt"] = gt

print(f"  {JOB}  同一训练相机，512²，黑底合成（与真值一致）")
for name, r in res.items():
    print(f"  {name:<16} 与真值 PSNR {np.mean(r['psnr']):6.2f} dB   清晰度 {np.mean(r['sharp']):.5f}")
best = max(res.items(), key=lambda kv: np.mean(kv[1]["psnr"]))
print(f"  → 最优：{best[0]}")
print(f"  与真值掩码 SSIM 略（用 PSNR 判断即可）")

out = ROOT / "env" / "tmp" / "preview_vs_interactive.png"
side = np.concatenate([cache["a"], cache["c"], cache["gt"]], axis=1)
Image.fromarray((np.clip(side, 0, 1) * 255).astype(np.uint8)).save(out)
print(f"  已写 {out}（左=A 1×/classic，中=C 2×/AA，右=真值）")
