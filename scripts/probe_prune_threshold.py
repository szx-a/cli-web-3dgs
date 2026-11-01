"""扫描"导出时剪掉低不透明度高斯"的阈值，看能否在几乎不损 LPIPS 的前提下补回 PSNR。

背景：MCMC 模型的 op 中位只有 0.022（official 0.036、旧 solid 0.101），意味着大量
极淡的"雾层"参与 alpha 合成 —— 这既拖低逐像素 PSNR，也让画面发虚。若剪掉它们
只用少量损失，就能作为**导出后处理**（不重训）改善质量。

用法：
  python scripts/probe_prune_threshold.py --job img5-mcmc --thresholds 0,0.02,0.03,0.05,0.08
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\3dgs-and-ai")
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from gsflow.gsdata.colmap_io import read_model  # noqa: E402
from gsflow.gsdata.shio import read_gaussians_full, render_view  # noqa: E402
from gsflow.paths import apply_runtime_env  # noqa: E402
from gsflow.trainers.perceptual import get_lpips  # noqa: E402

apply_runtime_env()

EVAL_DATA = ROOT / "workspace" / "img5-quality4" / "04_train" / "data"


def latest_ply(jd: Path) -> Path:
    return max((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
               key=lambda p: p.stat().st_mtime)


def subset(g: dict, keep: np.ndarray) -> dict:
    out = {}
    for k, v in g.items():
        a = np.asarray(v) if hasattr(v, "__len__") and k != "sh_degree" else None
        if a is not None and a.ndim >= 1 and len(a) == len(keep):
            out[k] = a[keep]
        else:
            out[k] = v
    out["n"] = int(keep.sum())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--thresholds", default="0,0.02,0.03,0.05,0.08")
    ap.add_argument("--views", type=int, default=8)
    args = ap.parse_args()

    jd = ROOT / "workspace" / args.job
    g0 = read_gaussians_full(latest_ply(jd))
    op = np.asarray(g0["opacity"], np.float64).reshape(-1)

    sp = EVAL_DATA / "sparse"
    rec = read_model(sp / "0" if (sp / "0" / "cameras.bin").exists() else sp)
    imgs = sorted(rec.images, key=lambda im: im.name)
    picks = np.linspace(0, len(imgs) - 1, min(args.views, len(imgs))).astype(int).tolist()
    fn = get_lpips("cuda", input_size=224)

    print(f"  {args.job}: n={len(op)}  op p10/p50/p90="
          f"{np.percentile(op,10):.3f}/{np.median(op):.3f}/{np.percentile(op,90):.3f}")
    print("  | 阈值 | 剩余高斯 | LPIPS | PSNR(m) |")
    print("  |---|---|---|---|")
    for th in [float(x) for x in args.thresholds.split(",")]:
        keep = op > th
        if keep.sum() < 16:
            continue
        g = subset(g0, keep) if th > 0 else g0
        vals, psnrs = [], []
        for k in picks:
            im = imgs[k]
            cam = rec.cameras[im.camera_id]
            R, c = im.cam_to_world()
            R = np.asarray(R, float)
            c = np.asarray(c, float)
            w2c = np.eye(4)
            w2c[:3, :3] = R.T
            w2c[:3, 3] = -R.T @ c
            K = np.asarray(cam.intrinsics(), float)
            col, _ = render_view(g, w2c, K, cam.width, cam.height)
            col_t = torch.as_tensor(np.asarray(
                col.detach().cpu() if hasattr(col, "detach") else col)[..., :3],
                dtype=torch.float32, device="cuda").clamp(0, 1)
            with Image.open(EVAL_DATA / "images" / im.name) as gi:
                gt = torch.as_tensor(np.asarray(gi.convert("RGB"), dtype=np.float32) / 255.0,
                                     device="cuda")
            if gt.shape[:2] != col_t.shape[:2]:
                continue
            with torch.no_grad():
                vals.append(float(fn(col_t, gt)))
            mp = EVAL_DATA / "masks" / (im.name.split(".")[0] + ".png")
            if mp.exists():
                m = np.asarray(Image.open(mp).convert("L")) > 127
                if m.shape == gt.shape[:2] and m.any():
                    ca = col_t.detach().cpu().numpy().astype(np.float64)
                    ga = gt.detach().cpu().numpy().astype(np.float64)
                    psnrs.append(10 * np.log10(1.0 / max(float(((ca - ga) ** 2)[m].mean()), 1e-10)))
        print(f"  | {th:.2f} | {int(keep.sum()):,} | {np.mean(vals):.4f} | {np.mean(psnrs):.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
