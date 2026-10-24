"""直接用训练器自己的 `evaluate()` 复算 PLY，与 `metrics.json` 对账。

## 为什么要这一条

同一份 30k PLY：

* 训练器写进 `metrics.json` 的 `psnr_masked` = **20.97**；
* 我从 PLY 渲染逐视角算 = **12~14 dB**（`probe_render_vs_gt` 里过曝+镶边）；
* 但 200 步的小模型两者吻合（14.5 vs 15.5）。

所以要么 `metrics.json` 与"实际渲染"不同源，要么我的渲染调用漏了某个参数。
本脚本调用**训练器自己的** `evaluate()`（不重写渲染逻辑），对同一 PLY 复算，
从而把"渲染实现"与"指标来源"分开。

用法: python scripts/probe_eval_vs_ply.py --job img3-final
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="img3-final")
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--data-dir", default="",
                    help="含 images/masks/sparse 的目录（默认 jd/04_train/data）")
    args = ap.parse_args()

    import torch
    from gsplat import rasterization

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.gsdata.shio import read_gaussians_full
    from gsflow.paths import apply_runtime_env
    from gsflow.trainers.gsplat_trainer import evaluate

    apply_runtime_env()
    dev = "cuda"
    jd = ROOT / "workspace" / args.job
    data = Path(args.data_dir) if args.data_dir else jd / "04_train" / "data"
    ply = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"), key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0))[-1]
    g = read_gaussians_full(ply)
    rec = max((read_model(c) for c in sorted(
        {p.parent for p in (data / "sparse").rglob("cameras.bin")})),
        key=lambda r: len(r.images))
    imgs = sorted(rec.images, key=lambda im: im.name)

    from PIL import Image

    W = rec.cameras[imgs[0].camera_id].width
    H = rec.cameras[imgs[0].camera_id].height

    means = torch.as_tensor(g["xyz"], dtype=torch.float32, device=dev)
    quats = torch.as_tensor(g["quats"], dtype=torch.float32, device=dev)
    scales = torch.as_tensor(g["scales"], dtype=torch.float32, device=dev)
    opac = torch.as_tensor(g["opacity"], dtype=torch.float32, device=dev).reshape(-1)
    shs = torch.as_tensor(g["shs"], dtype=torch.float32, device=dev)
    deg = int(g["sh_degree"])

    ims, masks, ks, vms = [], [], [], []
    picks = np.linspace(0, len(imgs) - 1, args.views).astype(int).tolist()
    for k in picks:
        im = imgs[k]
        cam = rec.cameras[im.camera_id]
        with Image.open(data / "images" / im.name) as gi:
            arr = np.asarray(gi.convert("RGB"), dtype=np.float32) / 255.0
        ims.append(torch.from_numpy(arr))
        mp = None
        for cand in (data / "masks" / (im.name + ".png"),
                     data / "masks" / (Path(im.name).stem + ".png")):
            if cand.exists():
                mp = cand
                break
        with Image.open(mp) as mi:
            masks.append(torch.from_numpy(np.asarray(mi.convert("L")) > 127))
        ks.append(torch.as_tensor(cam.intrinsics(), dtype=torch.float32))
        R, c = im.cam_to_world()
        R = np.asarray(R, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        w2c = np.eye(4)
        w2c[:3, :3] = R.T
        w2c[:3, 3] = -R.T @ c
        vms.append(torch.as_tensor(w2c, dtype=torch.float32))

    res = evaluate(
        rasterization=rasterization,
        means=means,
        quats=quats,
        scales=scales,
        opacities=opac,
        shs=shs,
        imgs=ims,
        masks=masks,
        ks=ks,
        viewmats=vms,
        eval_idx=list(range(len(picks))),
        sh_degree=deg,
        background=torch.zeros(3, device=dev),
        device=dev,
        max_side=None,
    )
    print(f"=== 训练器 evaluate() 复算 {args.job} / {ply.parent.name} ===")
    print(f"  evaluate(): psnr={res.get('psnr'):.3f}  psnr_masked={res.get('psnr_masked')}"
          f"  ssim={res.get('ssim'):.4f}  ssim_masked={res.get('ssim_masked')}"
          f"  views={res.get('views')}")
    m = json.loads((jd / "04_train" / "metrics.json").read_text(encoding="utf-8"))
    print(f"  metrics.json: psnr={m.get('psnr')}  psnr_masked={m.get('psnr_masked')}"
          f"  ssim_masked={m.get('ssim_masked')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
