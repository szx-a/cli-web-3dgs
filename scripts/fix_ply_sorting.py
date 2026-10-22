"""修掉"按字符串排序取最后一个检查点"的 bug（全局）。

## 这个 bug 有多致命

```python
ply = sorted((job / "04_train" / "point_cloud").glob("*/point_cloud.ply"), key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0))[-1]
```

字符串排序下 `iteration_5000` **排在 `iteration_30000` 之后**：

    sorted(['iteration_10000','iteration_30000','iteration_5000']) ->
    ['iteration_10000', 'iteration_30000', 'iteration_5000']     # 最后一个是 5000！

于是所有"取最终模型"的诊断脚本其实一直在渲染**第 5000 步的中间模型**。
症状：`probe_eval_vs_ply.py` 打印 `iteration_5000`、渲染图过曝带镶边、
逐视角 PSNR 只有 12~14 dB，而 `metrics.json` 写的是 20.97 dB（那是 30000）。

## 修法

统一改成按**父目录名里的数字**排序：

```python
max(<expr>.glob("<pat>"),
    key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0))
```

用法: python scripts/fix_ply_sorting.py [--check]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    # sorted(X.glob("*/point_cloud.ply"), key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0)) —— 字符串排序会把 iteration_5000 排到最后
    (re.compile(r'sorted\((.+?)\.glob\((["\'])\*/point_cloud\.ply\2\)\)'),
     r"sorted(\1.glob(\2*/point_cloud.ply\2), key=lambda p: "
     r"int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0))"),
]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    targets = []
    for d in ("scripts", "src", "tests"):
        targets.extend((ROOT / d).rglob("*.py"))
    changed, total = [], 0
    for p in targets:
        try:
            t = p.read_text(encoding="utf-8")
        except OSError:
            continue
        if "point_cloud.ply" not in t:
            continue
        new = t
        n = 0
        for rx, rep in PATTERNS:
            new, k = rx.subn(rep, new)
            n += k
        if n:
            changed.append((p, n))
            total += n
            if not args.check:
                p.write_text(new, encoding="utf-8")

    print(f"=== 修复检查点排序：{len(changed)} 个文件、{total} 处 ===")
    for p, n in changed:
        print(f"  {p.relative_to(ROOT)}（{n} 处）")
    if not changed:
        print("  已全部修复（没有按字符串排序取最后一个的写法）")
    if args.check:
        print("  --check：未写盘")
        return 1 if total else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
