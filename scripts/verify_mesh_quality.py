"""量化验证网格质量：把网格投影到训练视角，与"高斯渲染图"比轮廓一致性。

为什么不用离屏渲染：本机 open3d 的 Filament 离屏渲染在无窗口环境下返回全黑图，
所以没法直接"看"。但几何质量可以用**轮廓 IoU** 客观衡量：

  1. 高斯模型渲染 alpha -> 真实物体的轮廓（silhouette A）
  2. 网格用针孔投影 + z-buffer 光栅化 -> 网格轮廓（silhouette B）
  3. 两者 IoU 越高，说明网格在几何上越贴合模型

这样即使看不到图，也能判断"网格是不是把物体做出来了"。

用法：scripts\\gpu_python.cmd scripts\\verify_mesh_quality.py --job <job_id> [--mesh path]
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


def rasterize_mesh_silhouette(mesh, c2w, K, w, h):
    """把三角网格光栅化成轮廓掩码（纯 numpy 的 z-buffer 扫线，够用且无依赖）。"""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    if len(tris) == 0:
        return np.zeros((h, w), dtype=bool)

    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    vc = (w2c[:3, :3] @ verts.T).T + w2c[:3, 3]
    front = vc[:, 2] > 1e-6
    proj = (np.asarray(K, dtype=np.float64) @ vc.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = proj[:, :2] / proj[:, 2:3]
    uv[~front] = np.nan

    mask = np.zeros((h, w), dtype=bool)
    for t in tris:
        if not (front[t[0]] and front[t[1]] and front[t[2]]):
            continue
        p = uv[t]
        if not np.all(np.isfinite(p)):
            continue
        x0, y0 = int(np.floor(p[:, 0].min())), int(np.floor(p[:, 1].min()))
        x1, y1 = int(np.ceil(p[:, 0].max())), int(np.ceil(p[:, 1].max()))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w - 1), min(y1, h - 1)
        if x1 <= x0 or y1 <= y0:
            continue
        # 只填包围盒，靠三角形内点测试（简单可靠，速度对 512px 够用）
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64) + 0.5
        a, b, c = p[0], p[1], p[2]
        v0, v1 = b - a, c - a
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            continue
        v2 = pts - a
        u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / den
        v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / den
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        if inside.any():
            mask[pts[inside][:, 1].astype(int), pts[inside][:, 0].astype(int)] = True
    return mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--mesh", default="")
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    import torch
    import open3d as o3d
    from gsplat import rasterization

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.stages.export import ExportStage
    from gsflow.trainers.gsplat_trainer import find_model_dir

    job = ROOT / "workspace" / args.job
    mesh_path = Path(args.mesh) if args.mesh else None
    if mesh_path is None:
        cand = job / "05_export" / "model_mesh.obj"
        if not cand.exists():
            print(f"找不到网格：{cand}")
            return 1
        mesh_path = cand
    print(f"网格: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    print(f"三角形 {len(mesh.triangles)}，顶点 {len(mesh.vertices)}，"
          f"边流形={mesh.is_edge_manifold()}")

    gauss = ExportStage()._read_ply(ExportStage._find_ply(job), max_points=0)
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
    print(f"COLMAP 相机 {len(poses)}")

    n = len(gauss["xyz"])
    xyz = torch.as_tensor(gauss["xyz"], dtype=torch.float32, device="cuda")
    scales_real = torch.as_tensor(gauss["scales"], dtype=torch.float32, device="cuda")
    quats = torch.as_tensor(gauss["quats"], dtype=torch.float32, device="cuda")
    opac = torch.as_tensor(gauss["opacity"], dtype=torch.float32, device="cuda").reshape(-1)
    rgb = torch.as_tensor(gauss["rgb"], dtype=torch.float32, device="cuda") / 255.0
    shs = ((rgb - 0.5) / 0.28209479177387814).reshape(n, 1, 3)

    size = args.size
    step = max(1, len(poses) // args.views)
    print(f"\n{'view':>6}{'silhouette A':>14}{'silhouette B':>14}{'IoU':>8}")
    print("-" * 44)
    ious = []
    for i in range(0, len(poses), step):
        c2w, K, cw, ch = poses[i]
        sx, sy = size / cw, size / ch
        Ks = K.copy()
        Ks[0, 0] *= sx
        Ks[1, 1] *= sy
        Ks[0, 2] *= sx
        Ks[1, 2] *= sy
        with torch.no_grad():
            _, alphas, _ = rasterization(
                means=xyz, quats=quats, scales=scales_real,
                opacities=opac.clamp(0, 1), colors=shs,
                viewmats=torch.linalg.inv(torch.as_tensor(c2w, dtype=torch.float32,
                                                          device="cuda"))[None],
                Ks=torch.as_tensor(Ks, dtype=torch.float32, device="cuda")[None],
                width=size, height=size, sh_degree=0, packed=False, render_mode="RGB",
            )
        A = (alphas[0].cpu().numpy().reshape(size, size) > 0.5)
        B = rasterize_mesh_silhouette(mesh, c2w, Ks, size, size)
        inter = int((A & B).sum())
        union = int((A | B).sum())
        iou = inter / union if union else 0.0
        ious.append(iou)
        print(f"{i:>6}{int(A.sum()):>14}{int(B.sum()):>14}{iou:>8.3f}")

    print(f"\n平均轮廓 IoU = {np.mean(ious):.3f}")
    print("（IoU > 0.7 说明网格在几何上基本覆盖了物体；接近 0 说明网格和模型对不上）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
