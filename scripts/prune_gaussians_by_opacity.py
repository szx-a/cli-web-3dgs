"""离线剪枝：按不透明度阈值去掉"很淡"的高斯，另存一份 PLY。

## 为什么（2026-10-25 实测）

img4-v1 的高斯分布：**不透明度中位 0.066**，`op<0.1` 占 **67.2%**，`op>0.5` 只占 3.7%。
表面是由**大量半透明层叠加**出来的 ⇒ 从训练视角看颜色对得上（掩码内 PSNR 27.2），
但换到新视角就"透"，屋顶显得薄。

这个脚本用来**快速验证**"把淡高斯剪掉会不会更实"：不需要重训，剪完直接渲。

用法:
    python scripts/prune_gaussians_by_opacity.py --job img4-v1 --min-opacity 0.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="img4-v1")
    ap.add_argument("--in-ply", default="")
    ap.add_argument("--min-opacity", type=float, default=0.1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch

    from gsflow.gsdata.shio import read_gaussians_full
    from gsflow.trainers.gsplat_trainer import save_ply

    jd = ROOT / "workspace" / args.job
    if args.in_ply:
        src = Path(args.in_ply)
    else:
        src = max((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
                  key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))
    g = read_gaussians_full(src)
    op = np.asarray(g["opacity"]).reshape(-1)
    keep = op >= args.min_opacity
    print(f"  源：{src.parent.name}  {len(op):,} 个高斯；阈值 op≥{args.min_opacity}")
    print(f"  保留 {keep.sum():,}（{keep.sum()/len(op)*100:.1f}%），剪掉 {(~keep).sum():,}")
    if keep.sum() == 0:
        print("  阈值过高，全部被剪掉")
        return 1

    out = Path(args.out) if args.out else jd / "05_export" / f"model_gaussians_op{args.min_opacity:g}.ply"
    out.parent.mkdir(parents=True, exist_ok=True)
    shs = np.asarray(g["shs"])
    save_ply(
        out,
        torch.as_tensor(np.asarray(g["xyz"])[keep], dtype=torch.float32),
        torch.as_tensor(np.asarray(g["scales"])[keep], dtype=torch.float32),
        torch.as_tensor(np.asarray(g["quats"])[keep], dtype=torch.float32),
        torch.as_tensor(op[keep].reshape(-1, 1), dtype=torch.float32),
        torch.as_tensor(shs[keep], dtype=torch.float32),
        int(g["sh_degree"]),
    )
    print(f"  已写出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
