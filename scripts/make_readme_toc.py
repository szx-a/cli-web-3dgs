"""给 README 生成**可点击目录**（GitHub 上有效）。

为什么用显式锚点而不是 GitHub 自动 slug：中文标题的自动 id 规则（小写、去标点、
空格转连字符）在不同渲染器/版本上不完全一致，目录会静默失效。
这里为每个 `##`/`###` 标题插入一个稳定的 ASCII 锚点：

    <a id="sec-3-15"></a>
    ### 3.15 新测数据与取舍趋势

目录链接写 `#sec-3-15`，确定性 100%。

幂等：已有 `<!-- TOC -->` 时只重写目录块，不重复插锚点。
用法：python scripts/make_readme_toc.py
"""
from __future__ import annotations

import re
from pathlib import Path

P = Path(r"F:\3dgs-and-ai\README.md")
raw = P.read_text(encoding="utf-8")
# 重建式：先清掉旧锚点，保证锚点与目录始终一致、不会重复
import re as _re
raw = _re.sub(r'^<a id="sec[^"]*"></a>\n', '', raw, flags=_re.M)
lines = raw.splitlines()

HEAD_RE = re.compile(r"^(#{2,3})\s+(.*)$")
NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)")

anchors: list[tuple[int, str, str, str]] = []   # (line_idx, level, text, anchor)
used: set[str] = set()
in_fence = False
for i, ln in enumerate(lines):
    if ln.strip().startswith("```"):
        in_fence = not in_fence
        continue
    if in_fence:
        continue
    if ln.strip() == "## 目录":      # 目录自己不进目录
        continue
    m = HEAD_RE.match(ln)
    if not m:
        continue
    level, text = m.group(1), m.group(2).strip()
    if text == "目录":
        continue
    _CN = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
    if text[:1] in _CN:
        text_for_id = str(_CN[text[0]]) + text[1:]
    else:
        text_for_id = text
    num = NUM_RE.match(text_for_id)
    if num:
        base = "sec-" + num.group(1).replace(".", "-")
    else:
        # 无编号的标题：先取 ASCII 部分；若全是中文（如“硬件与约束”）
        # 则用标题哈希后六位，保证唯一且稳定（不会因顺序变化而漂）。
        import hashlib
        ascii_part = re.sub(r"[^0-9a-zA-Z]+", "-", text).strip("-")[:40]
        base = ("sec-" + ascii_part) if ascii_part else ("sec-h" + hashlib.md5(text.encode("utf-8")).hexdigest()[:6])
    aid = base
    n = 2
    while aid in used:
        aid = f"{base}-{n}"
        n += 1
    used.add(aid)
    anchors.append((i, level, text, aid))

# ---- \u63d2\u951a\u70b9\uff08\u4ece\u540e\u5f80\u524d\uff0c\u907f\u514d\u884c\u53f7\u6f02\u79fb\uff1b\u5df2\u6709\u5219\u8df3\u8fc7\uff09----
for idx, level, text, aid in reversed(anchors):
    prev = lines[idx - 1].strip() if idx > 0 else ""
    if prev == f'<a id="{aid}"></a>':
        continue
    lines.insert(idx, f'<a id="{aid}"></a>')

# \u63d2\u5165\u951a\u70b9\u540e\u91cd\u65b0\u89e3\u6790\u4e00\u904d\uff08\u884c\u53f7\u5df2\u53d8\uff09
text_all = "\n".join(lines) + "\n"
HEAD_RE2 = re.compile(r'<a id="([^"]+)"></a>\n(#{2,3})\s+(.*)$', re.M)
entries = [(m.group(1), len(m.group(2)), m.group(3).strip()) for m in HEAD_RE2.finditer(text_all)]

# ---- \u751f\u6210\u76ee\u5f55 ----
toc = ["<!-- TOC -->", "## 目录", ""]
for aid, level, text in entries:
    indent = "" if level == 2 else "  "
    toc.append(f"{indent}- [{text}](#{aid})")
toc += ["", "<!-- /TOC -->"]

block = "\n".join(toc) + "\n"

if "<!-- TOC -->" in text_all:
    text_all = re.sub(r"<!-- TOC -->.*?<!-- /TOC -->\n", block, text_all, flags=re.S)
else:
    # \u653e\u5728\u7b2c\u4e00\u4e2a `## ` \u4e4b\u524d
    m = re.search(r"^## ", text_all, re.M)
    assert m, "README 里没有二级标题，无法安放目录"
    text_all = text_all[:m.start()] + block + "\n" + text_all[m.start():]

P.write_text(text_all, encoding="utf-8")
print(f"  已写入目录：{len(entries)} 个标题（锚点 sec-*）")
for aid, level, text in entries[:6]:
    print(f"    {'  ' if level == 3 else ''}- {text[:48]}  ->  #{aid}")
