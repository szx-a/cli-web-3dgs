"""验证"模型全透明但 PSNR 不低"这一退化：把 PSNR 拆成前景/背景分别看。

动机（IMG2 的一个真实退化）：训练后 PLY 的不透明度中位数只有 0.0138，
导出的预览图**近乎全白**（不透明度 > 0.02 的高斯只剩 0.2%），
但报告的全留出集 PSNR 仍有 21.17 dB。两者不可能同时为真 —— 除非指标本身被"糊弄"了。

怀疑：掩码模式把背景涂成纯黑，而真值里背景也占大多数像素（前景占比 ~37%）。
所以一个"几乎什么都不渲染"的模型，只在背景像素上就能得到不低的整体 PSNR。
本脚本把 PSNR 按**前景/背景**拆开算，并与"纯黑基线"（直接输出全黑）对比，
一眼看出模型到底有没有真的重建出东西。

用法：
  env\\venv\\Scripts\\python.exe scripts\\diag_psnr_decompose.py --job img2-04
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-10))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--images", default="", help="真值图像目录，默认 04_train/data/images")
    ap.add_argument("--masks", default="", help="掩码目录，默认 04_train/data/masks")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    from gsflow import paths

    paths.apply_runtime_env()
    import torch
    from gsplat import rasterization
    from PIL import Image

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.stages.export import ExportStage
    from gsflow.trainers.gsplat_trainer import find_model_dir

    job = ROOT / "workspace" / args.job
    data_dir = job / "04_train" / "data"
    img_dir = Path(args.images) if args.images else data_dir / "images"
    msk_dir = Path(args.masks) if args.masks else data_dir / "masks"

    rec = read_model(find_model_dir(data_dir))
    names = [im.name for im in rec.images][: args.limit]
    print(f"{args.job}: 用 {len(names)} 张训练视图做拆解（分辨率 320）")

    dev = "cuda"
    g = ExportStage()._read_ply(ExportStage._find_ply(job), max_points=0)
    n = len(g["xyz"])
    print(f"PLY 高斯 {n}；不透明度中位数 "
          f"{float(np.median(g['opacity'])):.4f}，>0.02 占比 {float((g['opacity']>0.02).mean()):.3f}")
    xyz = torch.as_tensor(g["xyz"], dtype=torch.float32, device=dev)
    sc = torch.as_tensor(g["scales"], dtype=torch.float32, device=dev)
    qt = torch.as_tensor(g["quats"], dtype=torch.float32, device=dev)
    op = torch.as_tensor(g["opacity"], dtype=torch.float32, device=dev).reshape(-1)
    rgb = torch.as_tensor(g["rgb"], dtype=torch.float32, device=dev) / 255.0
    shs = ((rgb - 0.5) / 0.28209479177387814).reshape(n, 1, 3)

    S = 320
    rows = []
    for im in rec.images[: args.limit]:
        p = img_dir / im.name
        if not p.exists():
            cands = list(img_dir.glob(Path(im.name).stem + ".*"))
            if not cands:
                continue
            p = cands[0]
        with Image.open(p) as x:
            gt_img = x.convert("RGB")
        w, h = gt_img.size
        s = S / max(w, h)
        gt_img = gt_img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        gt = torch.from_numpy(np.asarray(gt_img, dtype=np.uint8).copy()).float().to(dev) / 255.0
        H, W = gt.shape[:2]

        cam = rec.cameras[im.camera_id]
        K = cam.intrinsics().astype(np.float32)
        K[0, 0] *= W / cam.width
        K[1, 1] *= H / cam.height
        K[0, 2] *= W / cam.width
        K[1, 2] *= H / cam.height
        r, t = im.cam_to_world()
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = r
        c2w[:3, 3] = t
        vm = torch.from_numpy(np.linalg.inv(c2w)).to(dev)

        with torch.no_grad():
            colors, alphas, _ = rasterization(
                means=xyz, quats=qt, scales=torch.exp(sc), opacities=op.clamp(0, 1),
                colors=shs, viewmats=vm[None], Ks=torch.from_numpy(K).to(dev)[None],
                width=W, height=H, sh_degree=0, packed=False, render_mode="RGB")
        render = colors[0].clamp(0, 1).cpu().numpy()
        gt_np = gt.cpu().numpy()
        a_np = alphas[0].cpu().numpy().reshape(H, W)
        # 决定性指标：渲染覆盖率。若 alpha 均值接近 0，说明模型根本没渲出东西；
        # 若 alpha 正常但 PSNR 低，才是几何/颜色不对。
        cover = float((a_np > 0.5).mean())
        alpha_mean = float(a_np.mean())
        rows.append((psnr(render, gt_np), alpha_mean, cover,
                     float((gt_np.max(axis=2) > 0.05).mean())))

    r = np.asarray(rows)
    ok = ~np.isnan(r).any(axis=1)
    print(f"\n{'整体 PSNR':>12}{'alpha均值':>12}{'覆盖(>0.5)':>12}{'真值前景占比':>14}")
    print(f"{np.mean(r[ok,0]):>12.2f}{np.mean(r[ok,1]):>12.4f}"
          f"{np.mean(r[ok,2])*100:>11.1f}%{np.mean(r[ok,3])*100:>13.1f}%")
    print("\n判读：")
    print("  alpha 均值 ~0 且覆盖 ~0  -> 模型**没渲出任何东西**（不透明度塌陷/高斯不在视锥里）")
    print("  alpha 正常但 PSNR 低      -> 几何或颜色不对")
    print("  alpha 与覆盖都正常、PSNR 也高 -> 模型是好的")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
