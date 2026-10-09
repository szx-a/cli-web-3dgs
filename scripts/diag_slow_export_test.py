"""计时定位：export 测试里到底哪一步慢。

用法：env\\venv\\Scripts\\python.exe scripts\\diag_slow_export_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TMP = ROOT / "env" / "tmp" / "diag_export"


def mark(label: str, t0: float) -> float:
    now = time.time()
    print(f"  {label:<38} {now - t0:7.2f}s", flush=True)
    return now


def main() -> int:
    t = time.time()
    import torch  # noqa: F401
    t = mark("import torch", t)

    from gsflow import paths
    t = mark("import paths", t)
    paths.apply_runtime_env()
    t = mark("apply_runtime_env()", t)

    from gsflow.config import load
    from gsflow.events import EventBus
    from gsflow.pipeline.context import StageContext
    from gsflow.stages import ExportStage
    from gsflow.trainers.gsplat_trainer import save_ply
    t = mark("import gsflow 模块", t)

    TMP.mkdir(parents=True, exist_ok=True)
    n = 50
    means = torch.randn(n, 3)
    scales = torch.full((n, 3), -3.0)
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    opac = torch.zeros(n, 1)
    shs = torch.zeros(n, 16, 3)
    job_dir = TMP / "ws" / "job2"
    ply = job_dir / "04_train" / "point_cloud" / "iteration_100" / "point_cloud.ply"
    save_ply(ply, means, scales, quats, opac, shs, 3)
    t = mark("save_ply", t)

    settings = load()
    settings.set("project.workspace_dir", str(TMP / "ws"))
    ctx = StageContext(job_id="job2", job_dir=job_dir, settings=settings,
                       events=EventBus(), spec={})
    ctx.bind_stage("export")
    t = mark("StageContext", t)
    stage = ExportStage()
    t = mark("ExportStage()", t)
    print("  _splat_kernel_ready:", stage._splat_kernel_ready(), flush=True)
    t = mark("_splat_kernel_ready()", t)
    res = stage.execute(ctx)
    mark(f"execute -> {res.status}", t)
    print("  产物:", sorted(p.name for p in (job_dir / "05_export").iterdir()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
