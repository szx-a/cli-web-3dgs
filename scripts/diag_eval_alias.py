"""验证"PSNR 骤降 = 评估视图循环混叠"这一假设。

假设链条：
  1. 训练视图按 `train_idx[step % len(train_idx)]` **循环选取**（行 615）
  2. `len(train_idx)` = 122 - 12 = 110；评估间隔 = 1000
  3. 6000 % 110 = 60，12000 % 110 = 60 -> **两次评估落在同一张训练图**
  4. 1000 % 110 = 10 -> 每 11 次评估（11000 步）完整重复一次
  5. 若该视图"难"（例如相机贴太近 / 视角偏），就会在**同样的步数**上出现同样的低谷
     —— 这与密度控制策略无关（实测内置策略与官方 DefaultStrategy 都在 6000/12000 骤降）

本脚本把"每次评估时对应的训练视图名"与"该视图到场景中心的距离"列出来，
看低谷那几次是否确实指向同一张图 / 特别近或特别远的相机。

用法：env\\venv\\Scripts\\python.exe scripts\\diag_eval_alias.py --job cmp-strategy
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
    ap.add_argument("--job", default="cmp-strategy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iterations", type=int, default=15000)
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--min-holdout", type=int, default=10)
    args = ap.parse_args()

    import torch

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.trainers.gsplat_trainer import find_model_dir

    job = ROOT / "workspace" / args.job
    rec = read_model(find_model_dir(job / "04_train" / "data"))
    imgs = list(rec.images)
    n = len(imgs)
    print(f"视图总数 {n}")

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).tolist()
    n_eval = max(1, n // 10) if n >= args.min_holdout else 0
    eval_idx = set(perm[:n_eval])
    train_idx = [i for i in range(n) if i not in eval_idx]
    psnr_idx = sorted(eval_idx) if eval_idx else list(range(n))
    print(f"留出评估视图 {n_eval} 个 -> {psnr_idx}")
    print(f"训练视图 {len(train_idx)} 个（循环周期 = {len(train_idx)}）")

    ctrs = []
    for im in imgs:
        r, t = im.cam_to_world()
        ctrs.append(np.asarray(t, dtype=np.float64))
    ctrs = np.asarray(ctrs)
    center = ctrs.mean(axis=0)
    dist = np.linalg.norm(ctrs - center, axis=1)

    print(f"\n{'step':>6}{'step%110':>9}{'训练视图名':>18}{'到中心距离':>12}   评估 PSNR")
    print("-" * 62)
    psnr_by_step = {}
    ev = job / "logs" / "events.jsonl"
    if ev.exists():
        import json

        for line in ev.read_text(encoding="utf-8").splitlines():
            try:
                j = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            m = j.get("message", "")
            if "评估 PSNR=" in m:
                try:
                    s = int(m.split("step")[1].split()[0])
                    v = float(m.split("PSNR=")[1].split()[0])
                    psnr_by_step[s] = v
                except Exception:  # noqa: BLE001
                    pass

    for step in range(args.eval_interval, args.iterations + 1, args.eval_interval):
        ti = train_idx[step % len(train_idx)]
        nm = imgs[ti].name
        ps = psnr_by_step.get(step)
        pss = f"{ps:.2f} dB" if ps is not None else "-"
        flag = "  <<< 低谷" if (ps is not None and ps < 20) else ""
        print(f"{step:>6}{step % len(train_idx):>9}{nm:>18}{dist[ti]:>12.3f}   {pss}{flag}")

    # 关键断言：低谷步数是否落在同一个 (step % 110)
    lows = [s for s, v in psnr_by_step.items() if v < 20]
    if lows:
        rems = {s % len(train_idx) for s in lows}
        print(f"\n低谷步数 {lows} 对应的 step%{len(train_idx)} = {sorted(rems)}")
        print("（若这些余数相同，说明低谷就是'恰好评估了同一张难图'，不是训练不稳）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
