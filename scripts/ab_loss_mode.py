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
    # 逐作业的参数覆盖：--override "jobA=train.lr_schedule=exp;jobB=train.lr_schedule=cosine"
    # （多个作业用换行/分号分隔不安全，这里改成每个作业一项，可重复传入）
    ap.add_argument("--override", action="append", default=[],
                    help="形如 job=键=值;键=值（可重复，每个作业一条）")
    args = ap.parse_args()

    overrides: dict[str, str] = {}
    for o in args.override:
        if "=" not in o:
            print(f"**--override 格式应为 job=键=值**: {o!r}")
            return 2
        job_name, spec = o.split("=", 1)
        overrides[job_name.strip()] = spec.strip()

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
        # 逐作业附加覆盖：--override "k=v;k=v" 用 **分号** 分隔多个键
        # （不能用逗号：值里可能含逗号，例如 export.formats=mesh,ply）。
        # 这个参数让本脚本成为**通用**的同数据对照工具，而不只是比 mask_loss_mode。
        spec = overrides.get(job, "") if overrides else ""
        if spec:
            for kv in spec.split(";"):
                kv = kv.strip()
                if not kv:
                    continue
                if "=" not in kv:
                    print(f"  **忽略无法解析的覆盖项**: {kv!r}")
                    continue
                k, v = kv.split("=", 1)
                params[k.strip()] = v.strip()
            print(f"  覆盖参数: {spec}")
        settings = load_settings(dict(params))
        ctx = StageContext(job_id=job, job_dir=job_dir, settings=settings,
                           events=EventBus(), params=params)
        # ⚠️ 必须设置 `_stage`：`ctx.cfg("k")` 会先拼成 `train.k` 再查。
        # 不设置时前缀为空，短键解析路径与生产环境（`stages/train.py`）**不一致**，
        # 表现为"覆盖看起来传进去了、实际没生效"。
        # （同类坑已在 run_export_direct.py 踩过一次：导出"成功"但没有网格。）
        ctx._stage = "train"
        # 自证：把将要生效的关键参数打出来，避免"静默无效"
        _probe = {k: ctx.cfg(k, None) for k in
                  ("lambda_lpips", "lambda_depth_distortion", "opacity_decay",
                   "utilization_prune_interval", "iterations", "mask_loss_mode",
                   "ssim_legacy_eps", "lr_schedule", "warmup_iters")}
        print("  生效参数: " + "  ".join(f"{k}={v}" for k, v in _probe.items()
                                          if v not in (None, 0, 0.0, "1.0", False)),
              flush=True)
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
