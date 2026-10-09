"""诊断：把 TSDF 融合用的"渲染深度"可视化出来，判断网格缺陷的来源。

背景：TSDF 网格出现"只有内壁、外壁缺失"时，问题可能在
  (a) 渲染的深度本身不对（尺度/范围/无效区），或
  (b) 融合参数不当（体素太大/截断太小/视角不够）。
本脚本把 (a) 直接画出来，先把输入端排除掉。

用法：scripts\\gpu_python.cmd scripts\\diagnose_mesh_depth.py --job <job_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow import paths  # noqa: E402

paths.apply_runtime_env()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch
    from PIL import Image

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.gsdata.mesh_export import render_depths, scale_intrinsics
    from gsflow.stages.export import ExportStage
    from gsflow.trainers.gsplat_trainer import find_model_dir

    job = ROOT / "workspace" / args.job
    ply = ExportStage._find_ply(job)
    print(f"模型: {ply}")

    gauss = ExportStage()._read_ply(ply, max_points=0)
    n = len(gauss["xyz"])
    data_dir = job / "04_train" / "data"
    model_dir = find_model_dir(data_dir)
    rec = read_model(model_dir)
    print(f"高斯 {n}；COLMAP 图像 {len(rec.images)}")

    poses = []
    for im in rec.images:
        cam = rec.cameras.get(im.camera_id)
        if cam is None:
            continue
        r, t = im.cam_to_world()
        c2w = np.eye(4)
        c2w[:3, :3] = r
        c2w[:3, 3] = t
        poses.append((c2w.astype(np.float32), cam.intrinsics().astype(np.float32),
                      int(cam.width), int(cam.height)))
    c2ws = np.stack([p[0] for p in poses])
    ks = np.stack([p[1] for p in poses])
    src_wh = (poses[0][2], poses[0][3])

    xyz = torch.as_tensor(gauss["xyz"], dtype=torch.float32, device="cuda")
    scales_real = torch.as_tensor(gauss["scales"], dtype=torch.float32, device="cuda")
    quats = torch.as_tensor(gauss["quats"], dtype=torch.float32, device="cuda")
    opac = torch.as_tensor(gauss["opacity"], dtype=torch.float32, device="cuda").reshape(-1)
    rgb = torch.as_tensor(gauss["rgb"], dtype=torch.float32, device="cuda") / 255.0
    shs = ((rgb - 0.5) / 0.28209479177387814).reshape(n, 1, 3)

    size = args.size
    rendered, used = render_depths(
        means=xyz, quats=quats, scales_real=scales_real,
        opacities=opac.clamp(0, 1), shs=shs,
        c2ws=c2ws, ks=ks, width=size, height=size, sh_degree=0,
        src_wh=src_wh, max_views=args.views,
    )

    pts = np.asarray(gauss["xyz"], dtype=np.float64)
    lo, hi = pts.min(0), pts.max(0)
    print(f"点云范围 {np.round(hi-lo,3)}  对角 {np.linalg.norm(hi-lo):.3f}")
    print(f"相机中心范围 {np.round(c2ws[:,:3,3].max(0)-c2ws[:,:3,3].min(0),3)}")
    print()
    print(f"{'view':>5}{'valid%':>9}{'depth min':>11}{'depth p50':>11}{'depth max':>11}")
    print("-" * 48)
    rows = []
    for k, i in enumerate(used):
        r, d, a = rendered[k]
        a2 = a.reshape(d.shape)
        valid = (a2 > 0.35) & np.isfinite(d) & (d > 1e-4)
        if valid.any():
            dv = d[valid]
            print(f"{int(i):>5}{float(valid.mean())*100:>8.1f}%{dv.min():>11.3f}"
                  f"{np.median(dv):>11.3f}{dv.max():>11.3f}")
        else:
            print(f"{int(i):>5}{float(valid.mean())*100:>8.1f}%{'--':>11}{'--':>11}{'--':>11}")
        # 可视化：RGB | 深度（归一化到有效范围）| alpha
        out = Path(args.out) if args.out else (ROOT / "env" / "diag_mesh")
        out.mkdir(parents=True, exist_ok=True)
        dnorm = np.zeros_like(d)
        if valid.any():
            lo_d, hi_d = dv.min(), dv.max()
            dnorm = np.clip((d - lo_d) / max(hi_d - lo_d, 1e-6), 0, 1)
            dnorm = np.where(valid, dnorm, 0)
        strip = np.concatenate([
            (r * 255).astype(np.uint8),
            np.stack([(dnorm * 255).astype(np.uint8)] * 3, -1),
            np.stack([(np.clip(a2, 0, 1) * 255).astype(np.uint8)] * 3, -1),
        ], axis=1)
        Image.fromarray(strip).save(out / f"depth_view_{int(i):03d}.png")
        rows.append((int(i), float(valid.mean())))

    print(f"\n对比图（RGB | 深度 | alpha）已写入 {args.out or (ROOT / 'env' / 'diag_mesh')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
