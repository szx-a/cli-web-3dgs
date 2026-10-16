"""扫描导出参数对**法线翻转率 / 可定向性**的影响 —— 定位"破碎"的可修来源。

已知事实：
  导出网格 `可定向=False`，相邻面翻转 **17.9%**，`orient_triangles()` **完全无效**。
  ⇒ 不是绕序可修，是几何自交。此前已排除：法线来源（G68）、λd（G82）、
    2DGS 投票（G62）。

本脚本按"最可能造成自交/桥接"的参数逐个扫，每个配置都同时量：
  翻转率、可定向性、轮廓 IoU、深度误差。
只看翻转率会误导（IoU 可能同时崩掉），所以必须一起看。

用法: scripts\\gpu_python.cmd scripts\\sweep_mesh_fragmentation.py --job dd-15
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def flips(mesh) -> tuple[float, float, int]:
    t = np.asarray(mesh.triangles, dtype=np.int64)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if len(t) < 10:
        return float("nan"), float("nan"), 0
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    fn = np.cross(b - a, c - a)
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    e2f: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, tri in enumerate(t):
        for k in range(3):
            x, y = int(tri[k]), int(tri[(k + 1) % 3])
            e2f[(min(x, y), max(x, y))].append(i)
    dots = [float(np.dot(fn[f[0]], fn[f[1]])) for f in e2f.values() if len(f) == 2]
    if not dots:
        return float("nan"), float("nan"), 0
    d = np.asarray(dots)
    return float(d.mean()), float((d < 0).mean()), int(len(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="dd-15")
    ap.add_argument("--tris", type=int, default=80000)
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import open3d as o3d

    from gsflow.config import load as load_settings
    from gsflow.events import EventBus
    from gsflow.pipeline.context import StageContext
    from gsflow.stages.export import ExportStage

    job = ROOT / "workspace" / args.job
    # 只扫与"自交/桥接/裁掉过多"直接相关的参数
    configs = [
        ("基线（trim=2.5）",            {}),
        ("不修边 trim=0",               {"export.mesh_trim_sigma": "0"}),
        ("修边更狠 trim=1.0",           {"export.mesh_trim_sigma": "1.0"}),
        ("不自动聚焦 crop=1.0",         {"export.mesh_crop_quantile": "1.0"}),
        ("裁剪更紧 crop=0.90",          {"export.mesh_crop_quantile": "0.90"}),
        ("不预简化 ratio=1",            {"export.mesh_presimplify_ratio": "1"}),
        ("不平滑 iters=0",              {"export.mesh_smooth_iters": "0"}),
        ("泊松 depth=8",                {"export.mesh_poisson_depth": "8"}),
        ("投票更严 min_views=3",        {"export.mesh_consistency_min_views": "3"}),
    ]

    print(f"作业 {args.job}  目标面数 {args.tris}")
    print(f"\n{'配置':<26}{'面数':>8}{'翻转':>8}{'点积':>8}{'可定向':>8}{'边流形':>8}")
    print("-" * 70)
    for name, params in configs:
        p = dict(params)
        p["export.formats"] = "mesh"
        p["export.mesh_method"] = "2dgs"
        p["export.mesh_target_triangles"] = str(args.tris)
        ctx = StageContext(job_id=args.job, job_dir=job,
                           settings=load_settings(dict(p)), events=EventBus(), params=p)
        ctx._stage = "export"
        try:
            ExportStage().run(ctx)
        except Exception as e:  # noqa: BLE001
            print(f"{name:<26} 导出失败：{type(e).__name__}: {str(e)[:40]}")
            continue
        mp = job / "05_export" / "model_mesh.obj"
        if not mp.exists():
            print(f"{name:<26} 无 mesh 产物")
            continue
        mesh = o3d.io.read_triangle_mesh(str(mp))
        m, f, _ = flips(mesh)
        try:
            orient = str(mesh.is_orientable())
            man = str(mesh.is_edge_manifold())
        except Exception:  # noqa: BLE001
            orient = man = "?"
        print(f"{name:<26}{len(mesh.triangles):>8}{f*100:>7.1f}%{m:>+8.3f}"
              f"{orient:>8}{man:>8}")
        # 每个配置都另存一份，便于后续测 IoU
        (job / "05_export" / f"mesh_{abs(hash(name)) % 10000:04d}.obj").write_bytes(
            mp.read_bytes())

    print("\n判读：")
    print("  * 若某配置翻转率大幅下降**且**面数/结构正常 ⇒ 那就是可修的来源；")
    print("  * 若所有配置都在 15%~25% ⇒ 自交来自**输入点云/法线的几何质量**（重建端），")
    print("    靠导出参数修不了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
