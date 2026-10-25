"""用**同一套**（论文对齐的）LPIPS 重新评估若干个作业，保证可比。

## 为什么需要

`perceptual.py` 在 2026-10-25 按论文重写（每层官方线性权重、逐通道空间均值）。
重写前 `metrics.json` 里的 `lpips` 是**旧实现**算的，量纲不同 ——
拿旧数字和新数字比较会得出错误结论（旧 0.547 vs 新 0.94 并不是"变差"，而是尺子换了）。

本脚本对**同一批留出视角**、用**当前实现**重新计算 LPIPS，给出可比的结果。

用法:
    python scripts/probe_lpips_new.py --jobs img5-v1,ab-lpfix-lpipsC-r1 --views 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def latest_ply(job: Path) -> Path | None:
    plys = list((job / "04_train" / "point_cloud").glob("*/point_cloud.ply"))
    if not plys:
        return None
    return max(plys, key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--data-dir", default="",
                    help="共享数据目录（对比实验的多个作业共用一份 data）")
    args = ap.parse_args()

    import torch
    from PIL import Image

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.gsdata.shio import read_gaussians_full, render_view
    from gsflow.paths import apply_runtime_env
    from gsflow.trainers.perceptual import get_lpips

    apply_runtime_env()
    fn = get_lpips("cuda", input_size=224)

    print("=== LPIPS 重评估（论文对齐实现，224 输入）===")
    for job_name in [j.strip() for j in args.jobs.split(",") if j.strip()]:
        jd = ROOT / "workspace" / job_name
        data = Path(args.data_dir) if args.data_dir else (jd / "04_train" / "data")
        ply = latest_ply(jd)
        if ply is None or not data.exists():
            print(f"  {job_name:<26} 缺产物")
            continue
        # 不同作业的稀疏模型可能在 sparse/ 或 sparse/0/ 下（COLMAP 两种布局都常见）
        sp = data / "sparse"
        cand = sp / "0" if (sp / "0" / "cameras.bin").exists() else sp
        rec = read_model(cand)
        imgs = sorted(rec.images, key=lambda im: im.name)
        picks = np.linspace(0, len(imgs) - 1, min(args.views, len(imgs))).astype(int).tolist()
        g = read_gaussians_full(ply)
        vals = []
        for k in picks:
            im = imgs[k]
            cam = rec.cameras[im.camera_id]
            W, H = cam.width, cam.height
            R, c = im.cam_to_world()
            R = np.asarray(R, float); c = np.asarray(c, float)
            K = np.asarray(cam.intrinsics(), float)
            w2c = np.eye(4); w2c[:3, :3] = R.T; w2c[:3, 3] = -R.T @ c
            col, _ = render_view(g, w2c, K, W, H)
            col = torch.as_tensor(np.asarray(col.detach().cpu().numpy() if hasattr(col, "detach") else col)[..., :3],
                                  dtype=torch.float32, device="cuda").clamp(0, 1)
            with Image.open(data / "images" / im.name) as gi:
                gt = torch.as_tensor(np.asarray(gi.convert("RGB"), dtype=np.float32) / 255.0,
                                     device="cuda")
            if gt.shape[:2] != col.shape[:2]:
                continue
            with torch.no_grad():
                vals.append(float(fn(col, gt)))
        if vals:
            print(f"  {job_name:<26} LPIPS={np.mean(vals):.4f}  (n={len(vals)}, 越低越好)")
        else:
            print(f"  {job_name:<26} 无法评估（尺寸不一致）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
