"""PLY 写入 → 读回 的**往返一致性**测试（纯 CPU，不需要 GPU）。

## 为什么要这一条

从 PLY 渲染（gsplat）时出现"过曝 + 彩虹镶边"，且**逐阶 SH 消融**显示：

| sh_degree | 掩码内 PSNR | 亮度（真值 0.533）|
|---|---|---|
| 0 | 18.38 | 0.538 ✓ |
| 3 | **12.21** | 0.693 ✗ |

DC 正常、高阶越加越差 ⇒ 高阶系数在「内存 → PLY → 读回」链条上被改过。
读/写的代码看起来都符合"通道主序"，所以必须用**往返测试**钉死，而不是继续肉眼审查。

用法: python scripts/probe_ply_roundtrip.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import torch

    from gsflow.gsdata.shio import read_gaussians_full
    from gsflow.trainers.gsplat_trainer import save_ply

    N = 7
    rng = np.random.default_rng(0)
    shs = rng.normal(0, 0.1, (N, 16, 3)).astype(np.float32)
    means = rng.normal(0, 1, (N, 3)).astype(np.float32)
    scales = np.abs(rng.normal(0.05, 0.01, (N, 3))).astype(np.float32)
    quats = np.tile(np.array([1, 0, 0, 0], np.float32), (N, 1))
    opac = np.full(N, 0.5, np.float32)

    p = ROOT / "env" / "tmp" / "roundtrip.ply"
    p.parent.mkdir(parents=True, exist_ok=True)
    save_ply(p, torch.from_numpy(means), torch.from_numpy(scales),
             torch.from_numpy(quats), torch.from_numpy(opac.reshape(-1, 1)),
             torch.from_numpy(shs), 3)
    g = read_gaussians_full(p)
    back = np.asarray(g["shs"])
    print(f"  原始 shs {shs.shape}  读回 {back.shape}")
    # 全部不透明度相同 ⇒ 排序应稳定；保险起见按坐标匹配一次
    A = shs
    B = back
    d_dc = float(np.abs(B[:, 0, :] - A[:, 0, :]).max())
    d_rest = float(np.abs(B[:, 1:, :] - A[:, 1:, :]).max())
    print(f"  DC   max|diff| = {d_dc:.6g}")
    print(f"  REST max|diff| = {d_rest:.6g}   "
          f"{'OK' if d_rest < 1e-6 else '**不一致**'}")
    if d_rest >= 1e-6:
        Arest = A[:, 1:, :]                 # [N,15,3]
        for name, cand in (
            ("原样", Arest),
            ("transpose(0,2,1)", Arest.transpose(0, 2, 1)),
            ("通道逆序", Arest[:, :, ::-1]),
            ("基内逆序", Arest[:, ::-1, :]),
        ):
            print(f"    与 {name:<20} 的最大差 = {np.abs(B[:, 1:, :] - cand).max():.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
