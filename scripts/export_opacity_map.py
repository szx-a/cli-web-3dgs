"""导出"不透明度治理方案"的映射为 JSON，供 Node 侧一致性检查读取。

为什么拆成两步：本机沙箱下 Node 无法 spawn Python 并捕获其输出
（AGENTS.md 记过：`stdio: 'pipe'` 会 EPERM，实测 `pid: 0`）。
所以先由 Python 把映射写到 JSON，Node 只读文件做纯 JS 侧校验。

用法: python scripts/export_opacity_map.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from gsflow.trainers.opacity_cleanup import CHOICES, opacity_cleanup_params

    data = {c: opacity_cleanup_params(c) for c in CHOICES}
    out = ROOT / "env" / "tmp" / "opacity_map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out}（{len(data)} 个方案）")
    for k, v in data.items():
        print(f"  {k:<12} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
