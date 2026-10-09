"""验证：预加载 gsplat 内核后，`import gsplat` 与光栅化是否秒级可用（不重编）。

用法：scripts\\gpu_python.cmd scripts\\diag_gsplat_preload.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import torch  # 必须先进：.pyd 依赖 torch 的 DLL

    from gsflow import paths

    t = time.time()
    paths.apply_runtime_env()
    print(f"apply_runtime_env : {time.time()-t:6.2f}s  "
          f"预加载成功={sys.modules.get('gsplat_cuda') is not None}")

    t = time.time()
    import gsplat  # noqa: F401
    print(f"import gsplat     : {time.time()-t:6.2f}s  （重编会 >60s）")

    t = time.time()
    from gsplat import rasterization

    dev = "cuda"
    n = 200
    means = torch.randn(n, 3, device=dev) * 0.2
    quats = torch.zeros(n, 4, device=dev)
    quats[:, 0] = 1.0
    opac = torch.full((n,), 0.5, device=dev)
    cols = torch.rand(n, 1, 3, device=dev)
    vm = torch.eye(4, device=dev)[None]
    vm[0, 2, 3] = 3.0
    K = torch.tensor([[200.0, 0, 32], [0, 200.0, 32], [0, 0, 1.0]], device=dev)[None]
    with torch.no_grad():
        _, alpha, _ = rasterization(
            means=means, quats=quats,
            scales=torch.exp(torch.full((n, 3), -3.0, device=dev)),
            opacities=opac, colors=cols, viewmats=vm, Ks=K,
            width=64, height=64, sh_degree=0, packed=False, render_mode="RGB")
    print(f"最小光栅化        : {time.time()-t:6.2f}s  alpha和={float(alpha.sum()):.1f}")

    # 预览依赖的渲染路径：真渲染必须能用
    from gsflow.stages.export import ExportStage

    print(f"_splat_kernel_ready: {ExportStage()._splat_kernel_ready()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
