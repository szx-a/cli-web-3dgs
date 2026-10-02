"""在已加载 MSVC + 便携 CUDA 的环境里直接调用 train()，把完整异常栈打出来。

用途：阶段执行会把异常包成 StageError（隐藏原始栈），排障时需要直接调用。

用法（必须经 GPU 包装器，才有 cl.exe）：
    scripts\\gsflow_gpu.cmd
    env\\venv\\Scripts\\python.exe scripts\\debug_train.py --iterations 100 --resolution 300
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow import paths  # noqa: E402

paths.apply_runtime_env()


class DummyCtx:
    """最小 ctx：train() 只用到 info/warn/success/emit/cancel_requested/logs_dir。"""

    job_dir = Path(os.environ.get("GSFLOW_DBG_JOB", str(ROOT / "workspace" / "debug-train")))
    params: dict = {}
    spec: dict = {}

    def __init__(self) -> None:
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self._stage = "train"

    @property
    def cancel_requested(self) -> bool:
        return False

    def _emit(self, msg, level="info", **kw):
        print(f"[{level}] {msg}", flush=True)

    def info(self, msg, **kw):
        self._emit(msg)

    def warn(self, msg, **kw):
        self._emit(msg, "warn")

    def error(self, msg, **kw):
        self._emit(msg, "error")

    def success(self, msg, **kw):
        self._emit(msg, "success")

    def emit(self, msg, level="info", **kw):
        self._emit(msg, level)

    def logs_dir(self) -> Path:
        d = self.job_dir / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=os.environ.get("GSFLOW_DBG_JOB", str(ROOT / "workspace" / "debug-train")))
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--resolution", type=int, default=300)
    ap.add_argument("--sh-degree", type=int, default=3)
    args = ap.parse_args()

    from gsflow.trainers.gsplat_trainer import TrainConfig, train

    job_dir = Path(args.job)
    data_dir = job_dir / "04_train" / "data"
    out_dir = job_dir / "04_train"
    print(f"CUDA_HOME      = {os.environ.get('CUDA_HOME')}")
    print(f"torch ext dir  = {os.environ.get('TORCH_EXTENSIONS_DIR')}")
    print(f"数据目录       = {data_dir}  (存在: {(data_dir / 'images').exists()})")

    DummyCtx.job_dir = job_dir
    cfg = TrainConfig(
        data_dir=data_dir,
        out_dir=out_dir,
        iterations=args.iterations,
        resolution=args.resolution,
        sh_degree=args.sh_degree,
        cap_max=200_000,
        eval=False,
    )
    try:
        metrics = train(cfg, DummyCtx())
        print("\n训练成功：", metrics)
        return 0
    except Exception:
        print("\n=== 完整异常栈 ===")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
