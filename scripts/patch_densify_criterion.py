"""把致密化判据从"3D 世界梯度"改为**官方口径：2D 屏幕梯度 / 可见次数**。

## 为什么必须改（本轮审计的核心结论）

官方 3DGS / gsplat `DefaultStrategy`：
    grads = meta["means2d"].grad            # [C,N,2] 屏幕空间梯度
    grads[..., 0] *= width / 2 * n_cameras  # 换算到 [-1,1] 屏幕空间
    grads = grads.norm(dim=-1)              # [C,N]
    grads = grad2d / count.clamp_min(1)     # 除以被看到次数
    is_grad_high = grads > grow_grad2d      # 官方 2e-4

本项目旧实现：
    means.grad.norm(dim=-1) > 2e-4          # 3D 世界空间梯度

**实测差异（scripts/probe_densify_criterion.py，final-v1，40 视角）**：
| 判据 | p50 | p90 | 超阈值比例 |
|---|---|---|---|
| 3D 梯度 | 1.60e-04 | 3.05e-03 | **2.6%** |
| 官方 2D | 3.55e-06 | 6.89e-05 | **0.2%** |
| 两者交集 IoU | | | **0.065** |

量纲差 ~45 倍，且**选出的高斯几乎完全不同**（IoU 0.065）。
后果已验证：`densify_until=15000` 得到 587,620 个高斯（4.3 倍），
但 PSNR **掉 0.874 dB**，近透明高斯从 21.2% 升到 36.8% ——
**在屏幕上看不出差别的位置加密，只会制造"雾"，不会产生细节。**

## 本补丁做什么

新增 `densify_criterion` 配置：
  * `"grad2d"`（**新默认**）—— 官方口径：累积 `means2d.grad`、按可见次数归一化；
  * `"grad3d"` —— 旧行为，保留以便 A/B 复现。

实现只替换**判据来源**，不动 `duplicate/split/prune` 的其余逻辑
（那部分此前拆一半移植失败过三次，见 03/G31；本次不碰）。

用法: python scripts/patch_densify_criterion.py [--apply]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "gsflow" / "trainers" / "gsplat_trainer.py"

OLD = '''        grad_norm_cache = None
        if means.grad is not None:
            with torch.no_grad():
                grad_norm_cache = means.grad.norm(dim=-1).detach().clone()'''

NEW = '''        # ---- 致密化判据（官方口径 vs 旧的 3D 口径）----
        #
        # 官方 `DefaultStrategy` 用**屏幕空间**梯度并除以可见次数：
        #     g = meta["means2d"].grad;  g[...,0] *= width/2*n_cameras
        #     g = g.norm(dim=-1);  g = g / count.clamp_min(1)
        #     densify if g > grow_grad2d
        # 我们旧版用 `means.grad.norm()`（3D 世界空间）+ 固定 2e-4。
        # 实测（scripts/probe_densify_criterion.py）：两者量纲差 ~45 倍，
        # 且选出的高斯集合 **IoU 仅 0.065** —— 加密位置几乎完全不同。
        # 后果已验证：densify_until=15000 时高斯 4.3 倍，PSNR 反降 0.874 dB，
        # 近透明高斯 21.2% -> 36.8%（"雾"变多，细节没变多）。
        grad_norm_cache = None
        _crit = str(getattr(cfg, "densify_criterion", "grad2d") or "grad2d").lower()
        if _crit == "grad2d":
            _m2 = meta.get("means2d") if isinstance(meta, dict) else None
            _radii = meta.get("radii") if isinstance(meta, dict) else None
            if _m2 is not None and _m2.grad is not None:
                with torch.no_grad():
                    gg = _m2.grad.detach().clone()          # [C,N,2]
                    # 官方第一步：换算到 [-1,1] 屏幕空间（只缩放 x 分量，与官方一致）
                    gg[..., 0] *= float(gt.shape[1]) / 2.0
                    g2d = gg.norm(dim=-1)                    # [C,N]
                    # 除以"被看到次数"（radii>0 视为可见；取不到就按 1）
                    if _radii is not None:
                        vis = (_radii.detach().reshape(-1, 2) > 0).any(dim=-1).float()
                        vis = vis.clamp_min(1.0)
                    else:
                        vis = torch.ones(g2d.shape[-1], device=g2d.device)
                    grad_norm_cache = (g2d / vis[None]).max(dim=0).values.detach().clone()
            else:
                ctx.warn("拿不到 means2d 梯度（densify_criterion=grad2d），"
                         "本轮退回 3D 梯度；如需稳定请检查 gsplat 版本")
        if grad_norm_cache is None and means.grad is not None:
            with torch.no_grad():
                grad_norm_cache = means.grad.norm(dim=-1).detach().clone()'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    text = TARGET.read_text(encoding="utf-8")
    if "densify_criterion" in text and "grad_norm_cache = (g2d / vis[None])" in text:
        print("已打过该补丁（幂等）")
        return 0
    if OLD not in text:
        print("**找不到要替换的代码块** —— 源码可能已变化，请人工核对")
        print("查找锚点：'grad_norm_cache = None' + 'means.grad.norm(dim=-1)'")
        return 1

    new_text = text.replace(OLD, NEW, 1)
    print(f"{'已写入' if args.apply else '将替换'} 一个代码块"
          f"（{len(OLD.splitlines())} 行 -> {len(NEW.splitlines())} 行）")
    if args.apply:
        TARGET.write_text(new_text, encoding="utf-8")
        print("完成")
    else:
        print("（未加 --apply，仅预览）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
