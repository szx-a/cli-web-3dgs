"""度量"质感"：渲染图相对真值的高频/纹理保真度。

三个互补指标（都只在掩码内统计）：

1. `grad_ratio`   = mean|∇C| / mean|∇GT|（梯度能量比；1.0 = 纹理强度一致，偏小 = 发糊）
2. `lap_ratio`    = Var(∇²C) / Var(∇²GT)（二阶高频能量比；对"表面发蜡"敏感）
3. `hf_corr`      = 高频分量的相关系数（corr(∇²C, ∇²GT)；越高说明细节位置对得上，而不只是强度够）

用法：
  python scripts/probe_texture_quality.py --jobs jobA,jobB --views 8 --data-from img5-quality4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from gsplat import rasterization  # noqa: E402
from PIL import Image  # noqa: E402

from gsflow.gsdata.colmap_io import read_model  # noqa: E402
from gsflow.gsdata.mesh_export import scale_intrinsics  # noqa: E402
from gsflow.gsdata.shio import read_gaussians_full  # noqa: E402
from gsflow.paths import apply_runtime_env  # noqa: E402


def _ply(job: str) -> Path:
    cands = sorted((ROOT / "workspace" / job / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
                   key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))
    if not cands:
        raise SystemExit(f"{job} 没有 point_cloud.ply")
    return cands[-1]


def _luma(x: np.ndarray) -> np.ndarray:
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114


def _grad_mag(img: np.ndarray) -> np.ndarray:
    gx = np.abs(np.diff(img, axis=1))[:-1, :]
    gy = np.abs(np.diff(img, axis=0))[:, :-1]
    return gx + gy


def _lap(img: np.ndarray) -> np.ndarray:
    return (4 * img[1:-1, 1:-1] - img[:-2, 1:-1] - img[2:, 1:-1]
            - img[1:-1, :-2] - img[1:-1, 2:])


def probe(job: str, data_from: str, views: int) -> dict:
    data = ROOT / "workspace" / (data_from or job) / "04_train" / "data"
    g = read_gaussians_full(_ply(job))
    sp = data / "sparse"
    rec = read_model(sp / "0" if (sp / "0" / "cameras.bin").exists() else sp)
    imgs = sorted(rec.images, key=lambda im: im.name)
    picks = np.linspace(0, len(imgs) - 1, max(1, views)).astype(int).tolist()

    means = torch.as_tensor(np.asarray(g["xyz"]), dtype=torch.float32, device="cuda")
    quats = torch.as_tensor(np.asarray(g["quats"]), dtype=torch.float32, device="cuda")
    scales = torch.as_tensor(np.asarray(g["scales"]), dtype=torch.float32, device="cuda")
    opac = torch.as_tensor(np.asarray(g["opacity"]).reshape(-1), dtype=torch.float32, device="cuda")
    shs = torch.as_tensor(np.asarray(g["shs"]), dtype=torch.float32, device="cuda")
    sh_deg = int(g["sh_degree"])

    rs, ls, cs = [], [], []
    for k in picks:
        im = imgs[k]
        cam = rec.cameras[im.camera_id]
        W, H = cam.width, cam.height
        R, c = im.cam_to_world()
        R = np.asarray(R, float)
        c = np.asarray(c, float)
        K = np.asarray(cam.intrinsics(), float)
        w2c = np.eye(4)
        w2c[:3, :3] = R.T
        w2c[:3, 3] = -R.T @ c
        with torch.no_grad():
            colors, alphas, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opac.clamp(0, 1),
                colors=shs, viewmats=torch.as_tensor(w2c[None], dtype=torch.float32, device="cuda"),
                Ks=torch.as_tensor(scale_intrinsics(K, (W, H), (W, H))[None], dtype=torch.float32, device="cuda"),
                width=W, height=H, sh_degree=sh_deg, near_plane=0.01, far_plane=1e10,
                packed=False, render_mode="RGB",
            )
            rgb = (colors[0] + (1 - alphas[0])).clamp(0, 1).cpu().numpy()

        gt_path = data / "images" / im.name
        if not gt_path.exists():
            continue
        gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
        if gt.shape != rgb.shape:
            continue
        mp = data / "masks" / (im.name.split(".")[0] + ".png")
        m = np.asarray(Image.open(mp).convert("L")) > 127 if mp.exists() else np.ones(gt.shape[:2], bool)
        if m.shape != gt.shape[:2]:
            m = np.ones(gt.shape[:2], bool)

        r_l, g_l = _luma(rgb), _luma(gt)
        gr, gg = _grad_mag(r_l), _grad_mag(g_l)
        mm = m[:-1, :-1]
        if mm.sum() < 32:
            continue
        rs.append(gr[mm].mean() / max(gg[mm].mean(), 1e-8))
        lr, lg = _lap(r_l), _lap(g_l)
        mm2 = m[1:-1, 1:-1]
        if mm2.sum() < 32:
            continue
        ls.append(lr[mm2].var() / max(lg[mm2].var(), 1e-12))
        a, b = lr[mm2] - lr[mm2].mean(), lg[mm2] - lg[mm2].mean()
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        cs.append(float((a * b).sum() / denom) if denom > 0 else 0.0)

    return {
        "job": job,
        "grad_ratio": float(np.mean(rs)) if rs else float("nan"),
        "lap_ratio": float(np.mean(ls)) if ls else float("nan"),
        "hf_corr": float(np.mean(cs)) if cs else float("nan"),
        "views": len(rs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--data-from", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    apply_runtime_env()
    if not torch.cuda.is_available():
        print("需要 CUDA")
        return 2

    out = []
    print(f"{'作业':<24}{'梯度能量比':>10}{'高频能量比':>11}{'高频相关':>9}{'视角':>6}")
    for job in [j.strip() for j in args.jobs.split(",") if j.strip()]:
        r = probe(job, args.data_from, args.views)
        out.append(r)
        print(f"{r['job']:<24}{r['grad_ratio']:>10.3f}{r['lap_ratio']:>11.3f}"
              f"{r['hf_corr']:>9.3f}{r['views']:>6}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已写 {args.json}")
    print("\n判读：梯度能量比与高频能量比越接近 1 越好（<0.9 偏糊，>1.15 过锐/噪）；高频相关越高越好。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
