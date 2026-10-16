"""学习率调度的**解析自检**：预热 + 余弦退火的取值必须与数学定义逐点一致。

为什么单独测：调度写错的后果是**静默**的 —— 曲线形状不对只会让训练"差一点"，
不会有任何报错。本项目已多次踩"没有已知答案就无法发现错误"的坑
（法线公式错 26.6° 而不报错、诊断脚本量出无意义的 2.31 倍）。所以这里用
**可以手算的端点**锁住行为：

    warmup_iters=W, iterations=T, peak=P, min_ratio=r
      step 0      -> P/W          （线性预热的第一步，非 0）
      step W-1    -> P            （预热末步到达峰值）
      step W      -> P            （预热结束仍是峰值）
      step (W+T)/2-> (P + P*r)/2  （余弦中点）
      step T-1    -> P*r          （退火到下限）
      预热段严格单调递增

用法: python scripts/check_lr_schedule.py
"""
from __future__ import annotations

import argparse
import math
import sys


def lr_at(step: int, peak: float, min_ratio: float, warm: int, total: int) -> float:
    """与 gsplat_trainer 里 `_lr_at` 同一实现（此处独立重写，避免"用被测代码测自己"）。"""
    min_lr = peak * min_ratio
    if warm > 0 and step < warm:
        return peak * (step + 1) / warm
    prog = (step - warm) / max(total - warm, 1)
    prog = min(max(prog, 0.0), 1.0)
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * prog))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak", type=float, default=1e-3)
    ap.add_argument("--total", type=int, default=1000)
    ap.add_argument("--warm", type=int, default=100)
    ap.add_argument("--min-ratio", type=float, default=0.1)
    args = ap.parse_args()

    P, T, W, r = args.peak, args.total, args.warm, args.min_ratio
    bad = 0

    def check(label: str, got: float, want: float, tol: float) -> None:
        nonlocal bad
        ok = abs(got - want) <= tol
        print(f"  [{'OK' if ok else '**FAIL**'}] {label}: got={got:.6e} want={want:.6e}")
        if not ok:
            bad += 1

    print(f"peak={P:g} total={T} warmup={W} min_ratio={r}")
    check("step 0 == peak/warm", lr_at(0, P, r, W, T), P / W, 1e-12)
    check("step W-1 == peak", lr_at(W - 1, P, r, W, T), P, 1e-12)
    check("step W == peak", lr_at(W, P, r, W, T), P, 1e-12)
    check("step 中点 == (peak+min)/2", lr_at((W + T) // 2, P, r, W, T),
          (P + P * r) / 2, 1e-4 * P)
    check(f"step T-1 ≈ min", lr_at(T - 1, P, r, W, T), P * r, 1e-3 * P)

    # 预热段严格单调递增
    vals = [lr_at(s, P, r, W, T) for s in range(W)]
    mono = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    print(f"  [{'OK' if mono else '**FAIL**'}] 预热段严格单调递增")
    if not mono:
        bad += 1

    # 退火段严格单调递减
    vals2 = [lr_at(s, P, r, W, T) for s in range(W, T)]
    mono2 = all(vals2[i] >= vals2[i + 1] - 1e-15 for i in range(len(vals2) - 1))
    print(f"  [{'OK' if mono2 else '**FAIL**'}] 退火段单调不增")
    if not mono2:
        bad += 1

    # 热身关闭时必须与纯余弦一致（warm=0 时首步即峰值）
    check("warmup=0 首步 == peak", lr_at(0, P, r, 0, T), P, 1e-12)

    print(f"\n结论: {'调度曲线与解析定义一致' if bad == 0 else f'**{bad} 项不一致**'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
