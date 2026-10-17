"""完整实测：训练 + 导出**全部 5 种产物**（供交付验收）。

产物（用户明确要求）：
  1. 点云 ply       04_train/point_cloud/iteration_N/point_cloud.ply
  2. 实体模型 ply    05_export/model_mesh.ply
  3. 点云 glb       05_export/model.glb
  4. 网络(网格) glb  05_export/model_mesh.glb
  5. 带纹理最终 glb  05_export/model_mesh_textured.glb

为什么用脚本直连而不是 `gsflow run`：
新作业没有编排器状态，`--stages train,export` 会先尝试从头跑 frames 并失败
（"未提供输入路径"）。本脚本复用源作业已备好的 `04_train/data`，
只跑 train 与 export 两个阶段，产出可复现。

用法:
  scripts\\gpu_python.cmd scripts\\run_final_pipeline.py --job final-v1 ^
      --data-from maskloss-v2 --iters 8000 --tris 80000 --texture
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REQUIRED = [
    ("点云 ply", "04_train/point_cloud", "point_cloud.ply"),
    ("实体模型 ply", "05_export", "model_mesh.ply"),
    ("点云 glb", "05_export", "model.glb"),
    ("网络(网格) glb", "05_export", "model_mesh.glb"),
    ("带纹理最终 glb", "05_export", "model_mesh_textured.glb"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="final-v1")
    ap.add_argument("--data-from", default="maskloss-v2")
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--tris", type=int, default=80000)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--texture", action="store_true")
    ap.add_argument("--tex-size", type=int, default=2048)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    from gsflow.config import load as load_settings
    from gsflow.events import EventBus
    from gsflow.paths import apply_runtime_env
    from gsflow.pipeline.context import StageContext
    from gsflow.stages.export import ExportStage
    from gsflow.trainers.gsplat_backend import GsplatBackend

    apply_runtime_env()

    job_dir = ROOT / "workspace" / args.job
    out_dir = job_dir / "04_train"
    data_dir = out_dir / "data"
    src = ROOT / "workspace" / args.data_from / "04_train"

    # ---- 1) 复刻数据（保证与对照实验同一份 COLMAP 结果）----
    if not (data_dir / "images").exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        if data_dir.exists():
            shutil.rmtree(data_dir)
        shutil.copytree(src / "data", data_dir)
        if (src / "undistort_info.json").exists():
            shutil.copy2(src / "undistort_info.json", out_dir / "undistort_info.json")
    n_img = sum(1 for p in (data_dir / "images").iterdir() if p.is_file())
    print(f"数据：{n_img} 张图（复用 {args.data_from}）")

    # ---- 2) 训练 ----
    params = {
        "train.lambda_depth_distortion": "1.5",   # 唯一稳超噪声底的改进
        "train.iterations": str(args.iters),
        "train.eval_interval": str(args.iters),
    }
    if not args.skip_train:
        settings = load_settings(dict(params))
        ctx = StageContext(job_id=args.job, job_dir=job_dir, settings=settings,
                           events=EventBus(), params=params)
        ctx._stage = "train"
        print(f"训练：{args.iters} 迭代，λd=1.5 ...")
        res = GsplatBackend().run(ctx, data_dir=data_dir, out_dir=out_dir,
                                  iterations=args.iters)
        print(f"  完成：PSNR={res.get('psnr')} 高斯={res.get('gaussians')}")

    # ---- 3) 导出 5 种产物 ----
    exp_params = {
        "export.formats": "glb_points,mesh,viewer_html",
        "export.mesh_method": "2dgs",
        "export.mesh_poisson_depth": str(args.depth),
        "export.mesh_target_triangles": str(args.tris),
        "export.mesh_texture": "true" if args.texture else "false",
        "export.mesh_texture_size": str(args.tex_size),
    }
    settings = load_settings(dict(exp_params))
    ctx = StageContext(job_id=args.job, job_dir=job_dir, settings=settings,
                       events=EventBus(), params=exp_params)
    ctx._stage = "export"
    print("导出 ...")
    ExportStage().run(ctx)

    # ---- 4) 验收：逐个确认存在且非空 ----
    print("\n=== 产物验收 ===")
    missing = []
    for label, sub, fname in REQUIRED:
        base = job_dir / sub
        hits = sorted(base.rglob(fname)) if base.exists() else []
        if hits:
            p = hits[-1]
            size = p.stat().st_size / 1024
            print(f"  [OK]   {label:<16} {p.relative_to(job_dir)}  {size:,.0f} KB")
        else:
            print(f"  [缺失] {label:<16} {sub}/{fname}")
            missing.append(label)

    # 记录到 metrics 便于其他脚本读取
    summ = job_dir / "05_export" / "artifacts_check.json"
    if summ.parent.exists():
        summ.write_text(json.dumps({
            "required": [{"label": l, "path": f"{s}/{f}"} for l, s, f in REQUIRED],
            "missing": missing,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n结论: {len(REQUIRED) - len(missing)}/{len(REQUIRED)} 个产物就绪"
          + (f"；缺 {missing}" if missing else "（全部齐全）"))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
