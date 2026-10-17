"""检查一个作业的致密化是否真的发生（阈值、克隆/分裂计数、最终高斯数）。

为什么需要：`img3-auto3` 只有 8,681 个高斯、PSNR 19.97 dB，而同数据的 `img3-crop`
是 117,271 / 28.78。差异必须能**从日志里读出来**（阈值、克隆数、分裂数），
而不是靠猜。

用法: python scripts/diag_densify_log.py --job img3-auto3 [--job img3-crop]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", default=[])
    args = ap.parse_args()
    jobs = args.job or ["img3-auto3", "img3-crop"]

    for job in jobs:
        f = ROOT / "workspace" / job / "logs" / "events.jsonl"
        print(f"\n=== {job} ===")
        if not f.exists():
            print("  （没有 events.jsonl）")
            continue
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        warn_false = 0
        dens_lines: list[str] = []
        thrs: set[str] = set()
        clones: list[int] = []
        splits: list[int] = []
        printeds: list[int] = []
        for raw in lines:
            if "未取到 means2d" in raw:
                warn_false += 1
            if "密度" not in raw:
                continue
            try:
                msg = json.loads(raw).get("message", "")
            except Exception:  # noqa: BLE001
                msg = raw
            dens_lines.append(msg)
            m = re.search(r"阈值=([\d.eE+-]+)", msg)
            if m:
                thrs.add(m.group(1))
            m = re.search(r"克隆\s*(\d+)", msg)
            if m:
                clones.append(int(m.group(1)))
            m = re.search(r"分裂\s*(\d+)", msg)
            if m:
                splits.append(int(m.group(1)))
            m = re.search(r"->\s*(\d+)\s*个高斯", msg)
            if m:
                printeds.append(int(m.group(1)))

        print(f"  假告警（未取到 means2d）次数 : {warn_false}")
        print(f"  密度日志条数                 : {len(dens_lines)}")
        print(f"  出现过的阈值                 : {sorted(thrs)}")
        if clones:
            print(f"  克隆数 合计 / 首次 / 末次     : "
                  f"{sum(clones):,} / {clones[0]} / {clones[-1]}")
        if splits:
            print(f"  分裂数 合计 / 首次 / 末次     : "
                  f"{sum(splits):,} / {splits[0]} / {splits[-1]}")
        if printeds:
            print(f"  高斯数 首次 / 末次            : {printeds[0]:,} → {printeds[-1]:,}")
        for msg in dens_lines[:4]:
            print("   · " + msg[:130])

        # 判定
        if not dens_lines:
            print("  ⇒ **完全没有密度日志**：densify_and_prune 没被调用或直接返回")
        elif sum(clones) + sum(splits) == 0:
            print("  ⇒ **克隆与分裂都是 0**：判据阈值远高于实际梯度（或梯度全为 0）")
        else:
            print("  ⇒ 有加密动作，但最终数量偏低的话需看是否被裁剪规则清掉")
    return 0


if __name__ == "__main__":
    sys.exit(main())
