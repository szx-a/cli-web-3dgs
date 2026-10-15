"""打印训练器的真实结构：`train()` 的步骤注释 + 每个阶段的实现函数。

用途：给架构/训练框架文档配图时，图上的名字必须是**代码里真实存在的**，
否则图就成了"看起来专业但查不到"的装饰。这个脚本把事实抓出来。

用法: python scripts/show_trainer_steps.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "gsflow" / "trainers" / "gsplat_trainer.py"

STAGE_KEYS = ("----", "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧")


def main() -> int:
    lines = SRC.read_text(encoding="utf-8", errors="replace").splitlines()
    print("=== 模块级定义（def / class）===")
    for i, l in enumerate(lines, 1):
        if re.match(r"^(def |class |@dataclass)", l):
            print(f"{i:5d}| {l.strip()[:100]}")

    # 定位 train()
    start = next((i for i, l in enumerate(lines) if l.startswith("def train(")), None)
    end = next((i for i, l in enumerate(lines[start + 1:], start + 1)
                if re.match(r"^def ", l)), len(lines))
    print(f"\n=== train() 内部的分段注释（行 {start + 1}..{end}）===")
    for i in range(start, end):
        s = lines[i].strip()
        if s.startswith("#") and any(k in s for k in STAGE_KEYS):
            print(f"{i + 1:5d}| {s[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
