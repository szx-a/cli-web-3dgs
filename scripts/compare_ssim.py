"""把本项目的 `ssim_simple` 与**社区参考实现**逐点对比。

为什么必须做（用户的批评）：我一直在用自己这套实现得出的数字下结论。
如果 `ssim_simple` 的数值本身偏了，那么"SSIM 项主导损失(61%)"、
"LPIPS 让 SSIM 变差"、以至于此前所有基于 SSIM 的判断**全部不可信**。

两个参照物：
  1. `skimage.metrics.structural_similarity`（本机已有，实现独立、被广泛引用）
  2. `pytorch_msssim`（3DGS 生态常用；若不在则跳过）

判据（可推理出答案）：
  * 渲染 == 真值            -> SSIM 必须 == 1.0
  * 独立随机噪声            -> SSIM 必须很低
  * 与参考实现的差          -> 应远小于两者间的**模型差异**（≈0.2），否则我的实现有问题
  * 常数图（全黑）对全黑    -> 必须 == 1.0（分母有 c1/c2 保护）

用法: python scripts/compare_ssim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import torch

    from gsflow.trainers.gsplat_trainer import ssim_simple

    have_msssim = True
    try:
        from pytorch_msssim import ssim as msssim_ssim
    except Exception:  # noqa: BLE001
        have_msssim = False

    from skimage.metrics import structural_similarity as sk_ssim

    print(f"pytorch_msssim 可用 = {have_msssim}\n")

    rng = np.random.default_rng(0)
    cases = {
        "完全相同": None,
        "轻微噪声(σ=0.02)": 0.02,
        "中等噪声(σ=0.10)": 0.10,
        "强噪声(σ=0.30)": 0.30,
        "纯随机": "rand",
        "全黑对全黑": "black",
    }

    H = W = 256
    base = np.zeros((H, W, 3), np.float32)
    # 造一个"有结构"的图：渐变 + 方块，避免只测噪声
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base[..., 0] = (xx / W) * 0.8 + 0.1
    base[..., 1] = (yy / H) * 0.8 + 0.1
    base[..., 2] = ((xx + yy) / (W + H)) * 0.6
    base[100:180, 60:200] = 0.9

    print(f"{'用例':>18}{'本项目':>10}{'skimage':>10}{'msssim':>10}{'差(本-sk)':>12}")
    print("-" * 62)
    worst = 0.0
    for name, kind in cases.items():
        if kind is None:
            a, b = base, base.copy()
        elif kind == "rand":
            a, b = base, rng.random((H, W, 3)).astype(np.float32)
        elif kind == "black":
            a, b = np.zeros((H, W, 3), np.float32), np.zeros((H, W, 3), np.float32)
        else:
            a = base
            b = np.clip(base + rng.normal(0, kind, base.shape).astype(np.float32), 0, 1)
        ta = torch.from_numpy(a).permute(2, 0, 1)[None]
        tb = torch.from_numpy(b).permute(2, 0, 1)[None]
        mine = float(ssim_simple(ta, tb))
        sk = float(sk_ssim(a, b, channel_axis=2, data_range=1.0))
        ms = float(msssim_ssim(ta, tb, data_range=1.0)) if have_msssim else float("nan")
        d = mine - sk
        if name not in ("完全相同", "全黑对全黑"):
            worst = max(worst, abs(d))
        ms_s = f"{ms:>10.4f}" if have_msssim else f"{'--':>10}"
        print(f"{name:>18}{mine:>10.4f}{sk:>10.4f}{ms_s}{d:>12.4f}")

    print()
    # 已知答案的硬断言
    bad = 0
    ta = torch.from_numpy(base).permute(2, 0, 1)[None]
    v_same = float(ssim_simple(ta, ta))
    if abs(v_same - 1.0) > 1e-4:
        print(f"  **FAIL** 相同图 SSIM={v_same:.6f}，应=1.0"); bad += 1
    else:
        print(f"  [OK] 相同图 SSIM = {v_same:.6f}")
    blk = torch.zeros(1, 3, 64, 64)
    v_blk = float(ssim_simple(blk, blk))
    if abs(v_blk - 1.0) > 1e-4:
        print(f"  **FAIL** 全黑对全黑 SSIM={v_blk:.6f}，应=1.0"); bad += 1
    else:
        print(f"  [OK] 全黑对全黑 SSIM = {v_blk:.6f}")
    if worst > 0.05:
        print(f"  **FAIL** 与 skimage 的最大偏差 {worst:.4f} 过大（>0.05）"); bad += 1
    else:
        print(f"  [OK] 与 skimage 最大偏差 {worst:.4f}（<0.05，可接受）")
    print(f"\n结论: {'ssim_simple 与参考实现一致' if bad == 0 else f'**{bad} 项不一致**'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
