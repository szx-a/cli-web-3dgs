"""对**已训练好的** 3DGS 模型快速试网格提取参数，并用轮廓 IoU 打分。

为什么需要：走完整流程试一个网格参数要重跑 COLMAP（几十秒）+ 训练（几分钟），
而网格提取本身只要几秒到一分钟。这个脚本直接复用已有 job 的
`04_train/point_cloud/**.ply`，把"调网格参数"变成秒级迭代。

用法示例：
  scripts\\gpu_python.cmd scripts\\tune_mesh.py --job cmp-maskfull

会依次尝试若干组参数，输出每组的三角形数与轮廓 IoU，最后给出最佳组合。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gsflow import paths  # noqa: E402

paths.apply_runtime_env()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max-combos", type=int, default=12)
    ap.add_argument("--extended", action="store_true",
                    help="额外扫描法线定向 / 离群点阈值 / KNN")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch
    from gsplat import rasterization

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.gsdata.mesh_export import (auto_crop_bounds, poisson_mesh,
                                           to_largest_component)
    from gsflow.stages.export import ExportStage, _gaussian_normals
    from gsflow.trainers.gsplat_trainer import find_model_dir
    from verify_mesh_quality import rasterize_mesh_silhouette

    job = ROOT / "workspace" / args.job
    ply = ExportStage._find_ply(job)
    print(f"模型: {ply}")
    base = ExportStage()._read_ply(ply, max_points=0)
    print(f"高斯 {len(base['xyz'])}")

    data_dir = job / "04_train" / "data"
    rec = read_model(find_model_dir(data_dir))
    poses = []
    for im in rec.images:
        cam = rec.cameras.get(im.camera_id)
        if cam is None:
            continue
        r, t = im.cam_to_world()
        c2w = np.eye(4)
        c2w[:3, :3] = r
        c2w[:3, 3] = t
        poses.append((c2w.astype(np.float64), cam.intrinsics().astype(np.float64),
                      int(cam.width), int(cam.height)))

    n = len(base["xyz"])
    xyz = torch.as_tensor(base["xyz"], dtype=torch.float32, device="cuda")
    scales_real = torch.as_tensor(base["scales"], dtype=torch.float32, device="cuda")
    quats = torch.as_tensor(base["quats"], dtype=torch.float32, device="cuda")
    opac = torch.as_tensor(base["opacity"], dtype=torch.float32, device="cuda").reshape(-1)
    rgb = torch.as_tensor(base["rgb"], dtype=torch.float32, device="cuda") / 255.0
    shs = ((rgb - 0.5) / 0.28209479177387814).reshape(n, 1, 3)

    # 预渲染高斯轮廓（与网格参数无关，只算一次）
    size = args.size
    step = max(1, len(poses) // args.views)
    sel = list(range(0, len(poses), step))
    silA = []
    with torch.no_grad():
        for i in sel:
            c2w, K, cw, ch = poses[i]
            Ks = K.copy()
            Ks[0, 0] *= size / cw
            Ks[1, 1] *= size / ch
            Ks[0, 2] *= size / cw
            Ks[1, 2] *= size / ch
            _, a, _ = rasterization(
                means=xyz, quats=quats, scales=scales_real,
                opacities=opac.clamp(0, 1), colors=shs,
                viewmats=torch.linalg.inv(torch.as_tensor(c2w, dtype=torch.float32,
                                                          device="cuda"))[None],
                Ks=torch.as_tensor(Ks, dtype=torch.float32, device="cuda")[None],
                width=size, height=size, sh_degree=0, packed=False, render_mode="RGB")
            silA.append((a[0].cpu().numpy().reshape(size, size) > 0.5, Ks))

    xyz_np = np.asarray(base["xyz"], dtype=np.float64)
    op_np = np.asarray(base["opacity"], dtype=np.float64).reshape(-1)
    normals_all = _gaussian_normals(base, type("C", (), {"info": lambda *a, **k: None})())

    def build(crop_q, dens_q, depth, largest=True, stat_std=2.0, knn=30, orient=True):
        g = base
        nrm = normals_all
        if crop_q is not None:
            lo, hi, _ = auto_crop_bounds(xyz_np, op_np, quantile=crop_q)
            m = np.all((xyz_np >= lo) & (xyz_np <= hi), axis=1)
            if m.sum() >= 256:
                g = {k: (v[m] if hasattr(v, "__len__") and len(v) == n else v)
                     for k, v in base.items()}
                if nrm is not None:
                    nrm = nrm[m]
        mesh, _ = poisson_mesh(np.asarray(g["xyz"], dtype=np.float64),
                               np.asarray(g["rgb"]), normals=nrm,
                               depth=depth, density_quantile=dens_q,
                               statistical_std=stat_std, knn=knn, orient=orient)
        if largest:
            mesh, _ = to_largest_component(mesh)
        return mesh

    def score(mesh):
        ious = []
        for (A, Ks), i in zip(silA, sel):
            B = rasterize_mesh_silhouette(mesh, poses[i][0], Ks, size, size)
            u = int((A | B).sum())
            ious.append(int((A & B).sum()) / u if u else 0.0)
        return float(np.mean(ious))

    # 第一轮：裁剪/密度/泊松层数（上一轮已定位 dens_q=0.0 最佳，这里继续找最优 depth 与 crop）
    combos = []
    for crop_q in (0.90, 0.95):
        for depth in (9, 10):
            for dens_q in (0.0, 0.001, 0.005):
                combos.append(dict(crop_q=crop_q, dens_q=dens_q, depth=depth))
    # 第二轮（--extended）：法线方向与离群点阈值
    if args.extended:
        for stat_std in (1.0, 2.0, 3.5):
            combos.append(dict(crop_q=0.90, dens_q=0.0, depth=9, stat_std=stat_std))
        for orient in (True, False):
            combos.append(dict(crop_q=0.90, dens_q=0.0, depth=9, orient=orient))
        for knn in (10, 50):
            combos.append(dict(crop_q=0.90, dens_q=0.0, depth=9, knn=knn))
    combos = combos[: args.max_combos]

    keys = ("crop_q", "dens_q", "depth", "stat_std", "knn", "orient")
    print("\n" + "".join(f"{k:>9}" for k in keys) + f"{'triangles':>11}{'IOU':>8}")
    print("-" * (9 * len(keys) + 19))
    results = []
    for c in combos:
        try:
            mesh = build(**c)
            iou = score(mesh)
            row = "".join(f"{str(c.get(k, '-')):>9}" for k in keys)
            print(f"{row}{len(mesh.triangles):>11}{iou:>8.3f}")
            results.append((iou, dict(c), len(mesh.triangles)))
        except Exception as exc:  # noqa: BLE001
            row = "".join(f"{str(c.get(k, '-')):>9}" for k in keys)
            print(f"{row}{'FAIL':>11}  {type(exc).__name__}: {str(exc)[:40]}")
    if results:
        results.sort(key=lambda r: -r[0])
        iou, best, tri = results[0]
        print(f"\n最佳: {best} -> IoU {iou:.3f}, {tri} 三角形")
        flags = " ".join(f"--set export.mesh_{k}={v}" for k, v in best.items()
                         if v is not None and k in ("crop_quantile", "density_quantile",
                                                    "poisson_depth", "stat_std", "knn", "orient"))
        print(f"建议参数：{flags}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
