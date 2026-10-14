"""复刻导出阶段的**精确**调用参数，定位"反投影得到 0 个点"。

与 diag_backproject.py 的区别：这里完全按 `export.py` 的取值来
（80 视角、640px、depth_stride/min_alpha/erode/grad 都从配置读），
并把 `filter_depth` 的每一步存活数、以及 `backproject_depth_to_points` 的内部统计都打印出来。

用法：scripts\\gpu_python.cmd scripts\\diag_backproject.py --job <job> --exact
"""

from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--exact", action="store_true", help="完全按导出阶段的参数复刻")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--views", type=int, default=80)
    args = ap.parse_args()

    import torch

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.gsdata.mesh_export import (backproject_depth_to_points,
                                           filter_depth, render_depths,
                                           select_views_by_distance, _scene_extent)
    from gsflow.stages.export import ExportStage
    from gsflow.trainers.gsplat_trainer import find_model_dir

    job = ROOT / "workspace" / args.job
    stage = ExportStage()
    mp = job / "04_train" / "metrics.json"
    best = json.loads(mp.read_text(encoding="utf-8")).get("best_ply") if mp.exists() else ""
    ply = Path(best) if best else None
    if ply is None or not ply.exists():
        c = sorted((job / "04_train" / "point_cloud").glob("iteration_*/point_cloud.ply"),
                   key=lambda p: int(p.parent.name.split("_")[1]))
        ply = c[-1]
    gauss = stage._read_ply(ply, max_points=0)
    n = len(gauss["xyz"])
    print(f"{args.job}: 检查点 {ply.parent.name}，高斯 {n}")

    data_dir = job / "04_train" / "data"
    rec = read_model(find_model_dir(data_dir))
    cams = []
    for im in rec.images:
        cam = rec.cameras[im.camera_id]
        K = cam.intrinsics().astype(np.float64)
        r, t = im.cam_to_world()
        c2w = np.eye(4); c2w[:3, :3] = r; c2w[:3, 3] = t
        cams.append((c2w, K, cam.width, cam.height))
    c2ws = np.stack([c[0] for c in cams]); ks = np.stack([c[1] for c in cams])
    src_wh = (cams[0][2], cams[0][3])

    pts = np.asarray(gauss["xyz"], dtype=np.float64)
    keep, vstat = select_views_by_distance(c2ws, pts)
    print(f"视角筛选：{vstat}")
    c2ws, ks = c2ws[keep], ks[keep]
    center, extent = _scene_extent(c2ws, pts)
    print(f"场景尺度 extent={extent:.4f}")

    rendered, used = render_depths(
        means=torch.as_tensor(gauss["xyz"], dtype=torch.float32, device="cuda"),
        quats=torch.as_tensor(gauss["quats"], dtype=torch.float32, device="cuda"),
        scales_real=torch.as_tensor(gauss["scales"], dtype=torch.float32, device="cuda"),
        opacities=torch.as_tensor(gauss["opacity"], dtype=torch.float32, device="cuda").reshape(-1).clamp(0, 1),
        shs=((torch.as_tensor(gauss["rgb"], dtype=torch.float32, device="cuda") / 255.0 - 0.5)
             / 0.28209479177387814).reshape(n, 1, 3),
        c2ws=c2ws, ks=ks, width=args.width, height=args.width, sh_degree=0,
        src_wh=src_wh, max_views=args.views)
    print(f"渲染 {len(rendered)} 个视角 @ {args.width}px；used_idx={np.asarray(used).tolist()[:8]}...")

    # 逐视角跑 filter_depth（与 backproject 内部完全同一参数）
    tot = 0
    for k, (rgb, depth, alpha) in enumerate(rendered):
        m, st = filter_depth(depth, alpha, min_alpha=0.5, erode_px=1,
                             grad_thresh_ratio=0.06,
                             extent=max(float(np.nanmax(depth)), 1e-6))
        tot += int(m.sum())
        if k < 3 or (k < 8 and m.sum() == 0):
            print(f"  视角 {k}: raw={st['px_raw']} eroded={st['px_eroded']} "
                  f"final={st['px_final']}  alpha_max={float(np.asarray(alpha).max()):.4f}")
    print(f"filter_depth 合计有效像素 = {tot}")

    bp, bc, bn, bst = backproject_depth_to_points(
        rendered=rendered, c2ws=c2ws, view_idx=used, ks=ks, src_wh=src_wh,
        width=args.width, height=args.width, depth_stride=2, min_alpha=0.5,
        erode_px=1, grad_thresh_ratio=0.06)
    print(f"backproject 统计 = {bst}")
    print(f"点数 = {len(bp)}")

    # === 二分定位：用逐步增大的视角数调用，看什么时候变成 0 ===
    print("\n=== 二分：不同视角数下的返回 ===")
    for nv in (1, 2, 3, 5, 10, 80):
        sub_r = rendered[:nv]
        sub_idx = np.asarray(used)[:nv]
        b2, _, _, s2 = backproject_depth_to_points(
            rendered=sub_r, c2ws=c2ws, view_idx=sub_idx, ks=ks, src_wh=src_wh,
            width=args.width, height=args.width, depth_stride=2, min_alpha=0.5,
            erode_px=1, grad_thresh_ratio=0.06)
        print(f"  视角数={nv:>3}: points={len(b2):>9}  stats={s2}")

    # 单视角深挖：把循环体逐句复刻，找出在哪一步点数变 0
    print("\n=== 单视角逐句复刻（kh=0）===")
    rgb, depth, alpha = rendered[0]
    valid, _ = filter_depth(depth, alpha, min_alpha=0.5, erode_px=1,
                            grad_thresh_ratio=0.06,
                            extent=max(float(np.nanmax(depth)), 1e-6))
    vs, us = np.nonzero(valid)
    print(f"  valid={len(vs)}")
    sel = ((vs % 2) == 0) & ((us % 2) == 0)
    vs, us = vs[sel], us[sel]
    print(f"  stride 后={len(vs)}")
    from gsflow.gsdata.mesh_export import scale_intrinsics
    Ks = scale_intrinsics(ks[used[0]], src_wh, (args.width, args.width))
    fx, fy, cx, cy = Ks[0, 0], Ks[1, 1], Ks[0, 2], Ks[1, 2]
    print(f"  Ks= fx{fx:.2f} fy{fy:.2f} cx{cx:.2f} cy{cy:.2f}  ks[used[0]]={np.round(ks[used[0]],2).tolist()}")
    z = depth[vs, us].astype(np.float64)
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    cam_pts = np.stack([x, y, z], axis=1)
    print(f"  z 范围 {z.min():.3f}~{z.max():.3f}  cam_pts 有限={np.isfinite(cam_pts).all()}")
    z_img = depth.astype(np.float64)
    zu_img = np.zeros_like(z_img); zv_img = np.zeros_like(z_img)
    zu_img[:, 1:-1] = 0.5 * (z_img[:, 2:] - z_img[:, :-2])
    zv_img[1:-1, :] = 0.5 * (z_img[2:, :] - z_img[:-2, :])
    zu = zu_img[vs, us]; zv = zv_img[vs, us]
    zc = np.maximum(z, 1e-12)
    m = np.stack([x / zc, y / zc, np.ones_like(z)], axis=1)
    pu = zu[:, None] * m + z[:, None] * np.stack(
        [np.full_like(z, 1.0 / fx), np.zeros_like(z), np.zeros_like(z)], axis=1)
    pv = zv[:, None] * m + z[:, None] * np.stack(
        [np.zeros_like(z), np.full_like(z, 1.0 / fy), np.zeros_like(z)], axis=1)
    cam_n = np.cross(pu, pv)
    nl = np.linalg.norm(cam_n, axis=1, keepdims=True)
    cam_n = np.divide(cam_n, np.where(nl > 1e-12, nl, 1.0))
    facing = np.einsum("ij,ij->i", cam_n, -cam_pts) > 0
    print(f"  cam_n 有限={np.isfinite(cam_n).all()}  len(cam_n)={len(cam_n)}")
    print(f"  dot(cam_n, -cam_pts) 分位数: "
          f"p1={np.percentile(np.einsum('ij,ij->i', cam_n, -cam_pts),1):.4f} "
          f"p50={np.percentile(np.einsum('ij,ij->i', cam_n, -cam_pts),50):.4f} "
          f"p99={np.percentile(np.einsum('ij,ij->i', cam_n, -cam_pts),99):.4f}")
    print(f"  **facing = dot(cam_n, -cam_pts) > 0 为真的数量 = {int(facing.sum())} / {len(facing)}**")
    print(f"  dot(cam_n, +cam_pts) > 0 为真的数量 = {int((np.einsum('ij,ij->i', cam_n, cam_pts) > 0).sum())}")
    # 检查"法线是否朝相机"的最直接判据：相机系里相机在原点、看向 +Z，
    # 因此可见表面法线应与 +Z 同向（即 n_z > 0）
    nz = cam_n[:, 2]
    print(f"  n_z > 0 的数量 = {int((nz > 0).sum())} / {len(nz)}（可见表面应大多数为正）")
    print(f"  n_z 分位数: p1={np.percentile(nz,1):.4f} p50={np.percentile(nz,50):.4f} p99={np.percentile(nz,99):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
