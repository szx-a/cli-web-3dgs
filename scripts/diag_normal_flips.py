"""定位法线翻转发生在哪一步：反投影 / 体素下采样 / 泊松。

背景：导出的网格有 **40.6% 的法线翻转**（邻域法线一致仅 0.156），
虽然轮廓 IoU 已到 0.855。翻转让光照出现大片黑斑（用户反馈的"杂乱"）。
可能的原因有三层，必须逐层测量才能定位：
  1. **反投影阶段**：单视角内相邻像素的法线是否已经互相矛盾（深度噪声）
  2. **体素下采样**：多视角重叠处方向相反的法线被**平均成 ~0**（这是设计上的风险点）
  3. **泊松**：输入法线若不一致，指示函数无解 -> 输出面片朝向随机

判据：对每个阶段的点云/网格，计算"随机取 2000 个点，其 8 近邻法线的平均点积"
（越大越一致），以及"相邻点法线点积 < 0 的比例"。

用法：scripts\\gpu_python.cmd scripts\\diag_normal_flips.py --job img2-opfix
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


def coherence(pts: np.ndarray, nrm: np.ndarray, k: int = 8, sample: int = 2000) -> dict:
    """邻域法线一致性（**纯 numpy 实现**，不用 open3d）。

    为什么不用 open3d 的 KDTreeFlann：实测它在反投影点云上会报
    `search_knn_vector_3d() error!` **且不说明原因**（点云有大量重合点/规模大时都会触发）。
    用"体素分桶 + 桶内以及相邻 27 桶找邻居"的 numpy 实现更稳，
    对本次判据（局部法线是否互相矛盾）完全够用。
    """
    pts = np.asarray(pts, dtype=np.float64)
    nrm = np.asarray(nrm, dtype=np.float64)
    fin = np.isfinite(pts).all(1) & np.isfinite(nrm).all(1)
    pts, nrm = pts[fin], nrm[fin]
    n = len(pts)
    if n < 50:
        return {"n": n, "note": "有效点太少"}

    # 归一化法线（一致性判据只关心方向）
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(ln, 1e-12)

    # 体素尺度取"平均最近邻距离"的估计：用包围盒/点数^(1/3) 近似
    span = pts.max(0) - pts.min(0)
    vol = float(np.prod(np.maximum(span, 1e-9)))
    cell = max((vol / max(n, 1)) ** (1.0 / 3.0) * 2.0, 1e-9)
    keys = np.floor(pts / cell).astype(np.int64)

    # 用字典聚合桶 -> 点索引（只保留少量点，控制内存）
    rng = np.random.default_rng(0)
    # 先随机抽样，减少建桶开销
    if n > 400000:
        sel = rng.choice(n, size=400000, replace=False)
        pts, nrm, keys = pts[sel], nrm[sel], keys[sel]
        n = len(pts)

    from collections import defaultdict

    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i in range(n):
        buckets[(int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))].append(i)

    probe = rng.choice(n, size=min(sample, n), replace=False)
    dots = []
    for i in probe:
        cx, cy, cz = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cand.extend(buckets.get((cx + dx, cy + dy, cz + dz), ()))
        if len(cand) < 2:
            continue
        cand = np.asarray(cand)
        d2 = ((pts[cand] - pts[i]) ** 2).sum(1)
        order = np.argsort(d2)[: k + 1]
        for j in cand[order[1:]]:
            dots.append(float(np.dot(nrm[i], nrm[j])))
    d = np.asarray(dots)
    if d.size == 0:
        return {"n": n}
    return {
        "n": n,
        "mean_dot": float(d.mean()),
        "median_dot": float(np.median(d)),
        "flip_ratio": float((d < 0).mean()),
        "zero_len_ratio": float((np.linalg.norm(nrm, axis=1) < 0.5).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--views", type=int, default=80)
    args = ap.parse_args()

    import torch

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.gsdata.mesh_export import (_scene_extent, backproject_depth_to_points,
                                           render_depths, select_views_by_distance,
                                           voxel_downsample)
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
    pts0 = np.asarray(gauss["xyz"], dtype=np.float64)
    keep, _ = select_views_by_distance(c2ws, pts0)
    c2ws, ks = c2ws[keep], ks[keep]
    center, extent = _scene_extent(c2ws, pts0)

    rendered, used = render_depths(
        means=torch.as_tensor(gauss["xyz"], dtype=torch.float32, device="cuda"),
        quats=torch.as_tensor(gauss["quats"], dtype=torch.float32, device="cuda"),
        scales_real=torch.as_tensor(gauss["scales"], dtype=torch.float32, device="cuda"),
        opacities=torch.as_tensor(gauss["opacity"], dtype=torch.float32, device="cuda").reshape(-1).clamp(0, 1),
        shs=((torch.as_tensor(gauss["rgb"], dtype=torch.float32, device="cuda") / 255.0 - 0.5)
             / 0.28209479177387814).reshape(n, 1, 3),
        c2ws=c2ws, ks=ks, width=args.width, height=args.width, sh_degree=0,
        src_wh=src_wh, max_views=args.views)

    # 1) 反投影（不做下采样）
    bp, bc, bn, bst = backproject_depth_to_points(
        rendered=rendered, c2ws=c2ws, view_idx=used, ks=ks, src_wh=src_wh,
        width=args.width, height=args.width, depth_stride=2, min_alpha=0.5,
        erode_px=1, grad_thresh_ratio=0.06)
    print(f"\n[1] 反投影: {len(bp)} 点  stats={bst}")
    if len(bp):
        print("    ", coherence(bp, bn))

    # 2) 体素下采样
    vds = extent / 400.0 * 0.25
    dp, dc, dn = voxel_downsample(bp, bc, bn, voxel=vds)
    print(f"\n[2] 体素下采样({vds:.5f}): {len(dp)} 点")
    if len(dp):
        print("    ", coherence(dp, dn))

    # 3) 现有网格
    mm = job / "05_export" / "model_mesh.ply"
    if mm.exists():
        import open3d as o3d

        m = o3d.io.read_triangle_mesh(str(mm))
        m.compute_triangle_normals()
        tri = np.asarray(m.triangles)
        fn = np.asarray(m.triangle_normals)
        print(f"\n[3] 导出网格: {len(tri)} 面")
        # ⚠️ 这里不用体素启发式取邻居（网格的顶点间距不均匀，启发式会取到很远的点，
        # 得到虚低的"一致性"）。改用**真实的网格邻接**：共享一条边的两个面，
        # 其法线夹角才是"表面是否平滑"的严格度量。
        from collections import defaultdict

        edge: dict[tuple[int, int], list[int]] = defaultdict(list)
        for f, (a, b, c) in enumerate(tri):
            for u, v in ((a, b), (b, c), (c, a)):
                edge[(min(u, v), max(u, v))].append(f)
        dots = []
        for fs in edge.values():
            if len(fs) == 2:
                dots.append(float(np.dot(fn[fs[0]], fn[fs[1]])))
        d = np.asarray(dots)
        print(f"    相邻面法线点积: mean={d.mean():.4f} median={np.median(d):.4f} "
              f"翻转(<0)={float((d < 0).mean()):.3f} 样本={d.size}")
        print(f"    二面角 >90° 的边占比 = {float((d < 0).mean())*100:.1f}%"
              "（越高越说明表面自交/翻折）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
