"""受控对照：**固定同一份 COLMAP/去畸变数据**，只改 `mask_loss_mode`。

为什么必须这样做（本轮最重要的方法学修正）：
我先前"损失修复带来 +3.14 dB"的结论是**跨作业**比较，而每个作业都会**各跑一次 COLMAP**。
实测两个作业的去畸变尺寸分别是 **720x720 / 722x722**、内参 fx **853.62 / 849.34** ——
所以那类对照里，"我改的东西"与"COLMAP 运行间的随机差异"是混在一起的。
本脚本让两个作业共用**同一份 data**，只切换损失口径，从而真正隔离变量。

用法:
  scripts\\gpu_python.cmd scripts\\ab_loss_mode.py --data-from maskloss-v2 ^
      --jobs ab-composite,ab-inside --modes composite,inside --iters 20000
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-from", required=True, help="提供 04_train/data 的源作业")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--modes", required=True)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    from gsflow.config import load as load_settings
    from gsflow.events import EventBus
    from gsflow.pipeline.context import StageContext
    from gsflow.trainers.gsplat_backend import GsplatBackend

    src = ROOT / "workspace" / args.data_from / "04_train"
    jobs = [j.strip() for j in args.jobs.split(",") if j.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if len(jobs) != len(modes):
        print("--jobs 与 --modes 数量必须一致")
        return 2

    out_rows = []
    for job, mode in zip(jobs, modes):
        job_dir = ROOT / "workspace" / job
        out_dir = job_dir / "04_train"
        data_dir = out_dir / "data"
        if not (args.reuse and (data_dir / "images").exists()):
            out_dir.mkdir(parents=True, exist_ok=True)
            if data_dir.exists():
                shutil.rmtree(data_dir)
            shutil.copytree(src / "data", data_dir)
            if (src / "undistort_info.json").exists():
                shutil.copy2(src / "undistort_info.json", out_dir / "undistort_info.json")
        n_img = sum(1 for p in (data_dir / "images").iterdir() if p.is_file())
        print(f"\n===== {job}  mask_loss_mode={mode}  图像 {n_img} 张 =====", flush=True)

        params = {"train.mask_loss_mode": mode,
                  "train.iterations": str(args.iters),
                  "train.eval_interval": str(args.iters)}
        settings = load_settings(dict(params))
        ctx = StageContext(job_id=job, job_dir=job_dir, settings=settings,
                           events=EventBus(), params=params)
        res = GsplatBackend().run(ctx, data_dir=data_dir, out_dir=out_dir,
                                  iterations=args.iters)
        out_rows.append((job, mode, res if isinstance(res, dict) else {}))

    print("\n=== 汇总（同一份数据，仅损失口径不同）===")
    for job, mode, d in out_rows:
        print(f"  {job:14s} mode={mode:10s} PSNR={d.get('psnr')}  "
              f"高斯={d.get('gaussians')}  最佳={d.get('best_psnr')}@{d.get('best_step')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
