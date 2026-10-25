"""扫描 README 里被 shell/字符串展开破坏的痕迹。

## 背景（2026-10-26，教训）

我此前用 PowerShell **双引号** here-string 改 README，导致：

* `$T`、`$x` 之类被当 **变量** 展开 → 数学里少了变量（如 `$T(u)=...` 变成 `(u)=...`）；
* `$$` 被展开成 **进程号** 或单个 `$` → 公式块退化为普通文字；
* `\\t`、`\\n` 被当 **转义** → 代码路径里混入 TAB/换行（如 `\\tests` 变成 TAB+"ests"）。

本脚本检查：制表符、可疑的孤立数字块、以及 `$` 前后的异常形态。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    bad = 0

    for i, ln in enumerate(lines, start=1):
        # 1) 制表符：正文里出现几乎一定是 `\\t` 被解释
        if "\t" in ln:
            bad += 1
            print(f"  [FAIL] 行 {i}: 含制表符（多半是 \\t 被展开）")
            print(f"         {ln.strip()[:110]}")

        # 2) 形如 `\train.` / `\tests/` 这种“反斜杠 + 小写词 + 路径”的可疑残留
        if re.search(r"[\u4e00-\u9fff]\\[a-z]+[./]", ln):
            bad += 1
            print(f"  [FAIL] 行 {i}: 路径/参数写在反斜杠后且紧跟小写词（可能被转义吃掉过）")
            print(f"         {ln.strip()[:110]}")

        # 3) 行内出现“孤立 $”紧跟中文（典型的 $变量 被吞后残留）
        if re.search(r"\$[\u4e00-\u9fff]", ln):
            bad += 1
            print(f"  [FAIL] 行 {i}: `$` 紧跟中文（数学变量可能被吞）")
            print(f"         {ln.strip()[:110]}")

    if bad:
        print(f"\n共 {bad} 处可疑")
        return 1
    print(f"  {path.name} 未见 shell 展开痕迹（{len(lines)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
