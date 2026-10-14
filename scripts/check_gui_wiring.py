"""检查 Web GUI 的表单控件与 app.js 的引用是否一一对应。

背景：GUI 的每个控件都要被 app.js 读取（collectParams）与监听（updateCliPreview），
漏一个就会出现"界面上填了但提交时没带上"的静默失真 —— 这正是 CLI/GUI 同构红线
最容易破的地方。这个脚本用静态检查把它挡住。

用法：env\\venv\\Scripts\\python.exe scripts\\check_gui_wiring.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "src" / "gsflow" / "web" / "static" / "index.html"
JS = ROOT / "src" / "gsflow" / "web" / "static" / "app.js"

# 这些控件不需要被 collectParams/updateCliPreview 处理
IGNORE_PREFIX = ("btn-", "tab-", "panel-", "stage-", "m-")


def main() -> int:
    html = HTML.read_text(encoding="utf-8")
    js = JS.read_text(encoding="utf-8")

    ids = set(re.findall(r'id="([^"]+)"', html))
    used = set(re.findall(r"\$\('([^']+)'\)", js))

    missing = sorted(u for u in used if u not in ids and not u.startswith(IGNORE_PREFIX))
    # 参数控件有两族：`p-*`（新建作业页）与 `me-*`（网格导出页）。
    # 两族都必须被 app.js 读取，否则"界面填了但提交时没带上"。
    PARAM_PREFIX = ("p-", "me-")
    unused = sorted(i for i in ids
                    if i.startswith(PARAM_PREFIX) and i not in used)

    print(f"HTML 控件总数：{len(ids)}")
    print(f"app.js 引用的 id 数：{len(used)}")
    ok = True
    if missing:
        ok = False
        print("\n[错误] app.js 引用了 HTML 中不存在的控件 id：")
        for m in missing:
            print(f"   - {m}")
    else:
        print("\n[OK] app.js 引用的 id 都存在于 HTML")

    if unused:
        ok = False
        print("\n[错误] HTML 里有参数控件没有被 app.js 使用"
              "（会导致'界面填了但没提交'的静默失真）：")
        for u in unused:
            print(f"   - {u}")
    else:
        print("[OK] HTML 里所有参数控件（p-* / me-*）都被 app.js 使用")

    # 再检查 updateCliPreview 的监听列表是否覆盖了所有 p-* 控件
    m = re.search(r"\[([^\]]*?)\]\.forEach\(\(id\) => \{ const el = \$\(id\)", js, re.S)
    if m:
        listened = set(re.findall(r"'([^']+)'", m.group(1)))
        not_listened = sorted(i for i in ids if i.startswith("p-") and i not in listened)
        if not_listened:
            ok = False
            print("\n[警告] 这些 p-* 控件没有绑定 input 事件，改它时 CLI 预览不会实时更新：")
            for n in not_listened:
                print(f"   - {n}")
        else:
            print("[OK] 所有 p-* 控件都绑定了 input 事件（CLI 预览会实时更新）")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
