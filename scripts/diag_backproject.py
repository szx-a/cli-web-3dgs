"""诊断"深度反投影得到 0 个点"：打印渲染出的 alpha/depth 分布，并逐级跑 filter_depth。

动机（实测）：`img2-2dgs-reg`（开启 2DGS 正则训练的模型）渲染预览**看起来正常**，
但网格导出报"反投影只得到 0 个点"。要定位到底是三个判据中的哪一个把像素全杀了：
  1. alpha 阈值（min_alpha）
  2. alpha 腐蚀（erode_px）
  3. 深度不连续性（grad_thresh_ratio）
也有可能是"渲染出的 alpha 本身就很小"（例如模型整体不透明度过低）。

用法：
  scripts\\gpu_python.cmd scripts\\diag_backproject.py --job img2-2dgs-reg
  scripts\\gpu_python.cmd scripts\\diag_backproject.py --job img2-opfix
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


def q(a: np.ndarray) -> str:
    if a.size == 0:
        return "（空）"
    ps = np.percentile(a, [1, 25, 50, 75, 99])
    return " ".join(f"p{p}={v:.4g}" for p, v in zip((1, 25, 50, 75, 99), ps))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--size", type=int, default=320)
    args = ap.parse_args()

    import torch

    from gsflow.gsdata.mesh_export import (_scene_extent, filter_depth,
                                           render_depths, select_views_by_distance)
    from gsflow.stages.export import ExportStage
    from gsflow.trainers.gsplat_trainer import find_model_dir
    from gsflow.gsdata.colmap_io import read_model

    job = ROOT / "workspace" / args.job
    stage = ExportStage()

    # 找 PLY（导出优先用 best_ply）
    import json

    mp = job / "04_train" / "metrics.json"
    best = ""
    if mp.exists():
        best = json.loads(mp.read_text(encoding="utf-8")).get("best_ply") or ""
    ply = Path(best) if best else None
    if ply is None or not ply.exists():
        cands = sorted((job / "04_train" / "point_cloud").glob("iteration_*/point_cloud.ply"),
                       key=lambda p: int(p.parent.name.split("_")[1]))
        ply = cands[-1] if cands else None
    if ply is None:
        print("找不到 PLY")
        return 1
    print(f"作业 {args.job}  使用检查点 {ply.parent.name}")

    gauss = stage._read_ply(ply, max_points=0)
    n = len(gauss["xyz"])
    print(f"高斯 {n} 个；不透明度 p50={np.median(gauss['opacity']):.4f} "
          f">0.5 占比={float((gauss['opacity']>0.5).mean()):.3f}")

    # 相机
    data_dir = job / "04_train" / "data"
    rec = read_model(find_model_dir(data_dir))
    names = [im.name for im in rec.images]
    cams_all = []
    for im in rec.images:
        cam = rec.cameras[im.camera_id]
        K = cam.intrinsics().astype(np.float64)
        r, t = im.cam_to_world()
        c2w = np.eye(4)
        c2w[:3, :3] = r
        c2w[:3, 3] = t
        cams_all.append((c2w, K, cam.width, cam.height, im.name))
    c2ws = np.stack([c[0] for c in cams_all])
    ks = np.stack([c[1] for c in cams_all])
    src_wh = (cams_all[0][2], cams_all[0][3])

    pts = np.asarray(gauss["xyz"], dtype=np.float64)
    keep, vstat = select_views_by_distance(c2ws, pts)
    print(f"视角筛选：保留 {vstat['kept']}/{vstat['total']}（距离中位数 {vstat['median_dist']}，MAD {vstat['mad']}）")
    c2ws_f, ks_f = c2ws[keep], ks[keep]
    center, extent = _scene_extent(c2ws_f, pts)
    print(f"场景中心 {np.round(center,3).tolist()} 尺度 {extent:.4f}")

    # 渲染深度
    rendered, used = render_depths(
        means=torch.as_tensor(gauss["xyz"], dtype=torch.float32, device="cuda"),
        quats=torch.as_tensor(gauss["quats"], dtype=torch.float32, device="cuda"),
        scales_real=torch.as_tensor(gauss["scales"], dtype=torch.float32, device="cuda"),
        opacities=torch.as_tensor(gauss["opacity"], dtype=torch.float32, device="cuda").reshape(-1).clamp(0, 1),
        shs=((torch.as_tensor(gauss["rgb"], dtype=torch.float32, device="cuda") / 255.0 - 0.5)
             / 0.28209479177387814).reshape(n, 1, 3),
        c2ws=c2ws_f, ks=ks_f, width=args.size, height=args.size, sh_degree=0,
        src_wh=src_wh, max_views=args.views,
    )
    print(f"\n渲染了 {len(rendered)} 个视角 @ {args.size}px")

    for k, (rgb, depth, alpha) in enumerate(rendered[:3]):
        d = np.asarray(depth, dtype=np.float32)
        a = np.asarray(alpha, dtype=np.float32).reshape(d.shape)
        print(f"\n--- 视角 {k} ---")
        print(f"  alpha:  {q(a)}   >0.5 占比={float((a>0.5).mean()):.4f}  >0.9 占比={float((a>0.9).mean()):.4f}")
        print(f"  depth:  {q(d)}   >1e-4 占比={float((d>1e-4).mean()):.4f}")
        for ma, er, gr in ((0.5, 1, 0.06), (0.5, 0, 0.06), (0.5, 1, 0.0), (0.1, 0, 0.0)):
            m, st = filter_depth(d, a, min_alpha=ma, erode_px=er,
                                 grad_thresh_ratio=gr, extent=extent)
            print(f"    min_alpha={ma} erode={er} grad={gr}: 存活 {int(m.sum())}"
                  f"（raw={st['px_raw']} eroded={st['px_eroded']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
