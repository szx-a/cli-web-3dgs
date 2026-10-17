"""绕过编排器，**直接调用导出阶段**。

用途：本轮的受控 A/B（`ab_loss_mode.py`）是用脚本直接调训练函数的，
因此作业目录里没有编排器的 job 状态，`gsflow run --stages export` 会试图
从头跑 frames/colmap 而失败。这个脚本直接驱动 ExportStage，
参数走 `--set` 同样的点路径语义。

用法:
  scripts\\gpu_python.cmd scripts\\run_export_direct.py --job ab-composite ^
      --set export.formats=mesh --set export.mesh_method=2dgs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    # 交付目录里已有 >=3 个交付物时默认拒绝写入（导出会清空整个目录）。
    # 做对照实验请换作业目录，或显式允许覆盖。
    ap.add_argument("--allow-overwrite", action="store_true",
                    help="允许覆盖已有交付物的导出目录（会清空它！）")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    from gsflow.config import load as load_settings
    from gsflow.events import EventBus
    from gsflow.pipeline.context import StageContext
    from gsflow.stages.export import ExportStage

    params = {}
    for kv in args.sets:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        params[k.strip()] = v.strip()

    job_dir = ROOT / "workspace" / args.job
    # ⚠️ 交付物保护（真实事故）：`export.py` 每次重跑会**清空整个 05_export**。
    # 我用本脚本做"只导 mesh"的 depth 对照时，把 final-v1 里已有的
    # `model.glb`（点云 glb）与 `model_mesh_textured.glb`（带纹理）删掉了 ——
    # 用户打开目录只看到 3 个模型文件。
    # 所以这里默认**拒绝**写入已有交付物的目录，除非显式 --allow-overwrite。
    from gsflow.gsdata.export_guard import assert_safe_export_target
    assert_safe_export_target(job_dir / "05_export",
                              allow_overwrite=bool(args.allow_overwrite))
    ev = EventBus()
    # 把事件实时打出来：导出阶段内部会 try/except 吞掉网格异常并记成 WARN，
    # 不打印就只看到"没有产物"而不知道原因。
    def _emit(e):
        try:
            msg = getattr(e, "message", "") or ""
            lvl = getattr(e, "level", "")
            if msg:
                print(f"  [{lvl or 'info'}] {msg}" if lvl else f"  {msg}", flush=True)
        except Exception:  # noqa: BLE001
            pass
    try:
        ev.subscribe(_emit)          # 有则订阅
    except Exception:  # noqa: BLE001
        pass
    ctx = StageContext(job_id=args.job, job_dir=job_dir,
                       settings=load_settings(dict(params)),
                       events=ev, params=params)
    # ⚠️ 必须设置 `_stage`：`ctx.cfg("formats")` 会据此前缀成 `export.formats`。
    # 不设置时 `_stage` 为空串，短键会直接落到 `settings.get(key)`，
    # 于是 `--set export.formats=mesh` **读不到** —— 表现为"导出成功但没有网格产物"。
    ctx._stage = "export"
    res = ExportStage().run(ctx)
    print(f"导出阶段返回：{type(res).__name__}")
    print(f"  状态={getattr(res, 'status', '?')}")
    print(f"  产物={getattr(res, 'artifacts', None) or getattr(res, 'outputs', None) or '（未提供）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
