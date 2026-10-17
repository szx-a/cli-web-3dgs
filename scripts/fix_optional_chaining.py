"""把 app.js 里误加的可选链还原（可选链不能出现在赋值左侧）。

## 背景

为了"任何元素缺失都不要让整块功能挂掉"，我给 `$('m-overlay')` 等统一加了
`?.`。但**赋值左侧不能用可选链**：

    $('m-status')?.textContent = 'x'   // SyntaxError: Invalid left-hand side

正确的修法不是加可选链，而是**别把别人的 DOM 删掉**（根因见
`refviewer.js::disposeViewer` 的说明）。所以这里把 `?.` 全部还原。

用法: python scripts/fix_optional_chaining.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "gsflow" / "web" / "static" / "app.js"

IDS = ("m-overlay", "m-image", "m-canvas", "m-status")


def main() -> int:
    t = APP.read_text(encoding="utf-8")
    total = 0
    for eid in IDS:
        before = t
        t = t.replace(f"$('{eid}')?.", f"$('{eid}').")
        total += sum(1 for _ in range(0)) + (1 if t != before else 0)
    APP.write_text(t, encoding="utf-8")
    print(f"  已把 {len(IDS)} 个元素的可选链还原（改了 {total} 个）")
    # 自检：不应再出现 $('x')?.
    left = [e for e in IDS if f"$('{e}')?." in t]
    if left:
        print(f"  **仍有残留: {left}**")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
