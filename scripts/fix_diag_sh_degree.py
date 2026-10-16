"""把诊断脚本硬编码的 `sh_degree=0` 改为**数据里真实的** SH 阶数。

事故背景：训练用 `sh_degree=3`（PLY 有 45 个 `f_rest_*`），但导出与 22 个诊断脚本
都硬编码 `sh_degree=0` 渲染 ⇒ 实测**低估 1.91 dB**，且抹掉视角相关外观
（正是感知损失/几何正则要改善的部分）⇒ 用这种尺子测优化手段必然假阴性。

本脚本按两类脚本分别处理：
  A. 已用 `ExportStage._read_ply(...)` 的：返回 dict 现在含 `shs` 与 `sh_degree`，
     在赋值语句后插入一行取阶数，并把 `sh_degree=0` 换成它。
  B. 自己读 PLY 的：把渲染处的 `sh_degree=0` 换成 `0` 会**再次**出错，
     所以这类只**报告**、不自动改（需人工决定是否迁移到 shio）。

用法: python scripts/fix_diag_sh_degree.py [--apply]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SKIP = {"check_sh_export.py", "refactor_read_ply.py", "render_train_views.py",
        "fix_diag_sh_degree.py"}

# 匹配：<var> = ... _read_ply(...)    （可能跨行，取到行尾）
ASSIGN_RE = re.compile(r"^(\s*)(\w+)\s*=\s*[^=\n]*_read_ply\(", re.M)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    done, manual = [], []
    for p in sorted(SCRIPTS.glob("*.py")):
        if p.name in SKIP:
            continue
        text = p.read_text(encoding="utf-8")
        if "sh_degree=0" not in text:
            continue

        m = ASSIGN_RE.search(text)
        if not m:
            manual.append((p.name, "自己读 PLY（未走 _read_ply）—— 建议迁移到 shio"))
            continue

        indent, var = m.group(1), m.group(2)
        # 幂等：已经有阶数变量就不再插入
        if re.search(rf"^\s*_\s*=\s*int\({var}\.get\(\"sh_degree\"", text, re.M):
            manual.append((p.name, "已处理过，跳过"))
            continue
        if '"shs"' not in text and "'shs'" not in text:
            manual.append((p.name, "拿到了 gauss 但没用 gauss[\"shs\"]（需人工确认颜色来源）"))
            continue

        # 1) 插入阶数提取（放在读 PLY 的那条语句之后：找到该语句的结束行）
        lines = text.splitlines(keepends=True)
        idx = next((i for i, l in enumerate(lines) if "_read_ply(" in l), None)
        if idx is None:
            manual.append((p.name, "定位失败"))
            continue
        # 找到该语句结束（括号平衡）
        depth = 0
        end = idx
        for j in range(idx, len(lines)):
            depth += lines[j].count("(") - lines[j].count(")")
            if depth <= 0 and j >= idx:
                end = j
                break
        inject = (f"{indent}# 用数据里真实的 SH 阶数（硬编码 0 会低估约 1.9 dB，见 shio.py）\n"
                  f"{indent}_sh_deg = int({var}.get(\"sh_degree\", 0) or 0)\n")
        lines.insert(end + 1, inject)
        text = "".join(lines)
        n = text.count("sh_degree=0")
        text = text.replace("sh_degree=0", "sh_degree=_sh_deg")
        done.append((p.name, var, n))
        if args.apply:
            p.write_text(text, encoding="utf-8")

    print(f"{'已修改' if args.apply else '将修改'} {len(done)} 个脚本：")
    for name, var, n in done:
        print(f"  {name:<34} 取 {var}['sh_degree'] -> _sh_deg   ({n} 处渲染)")
    if manual:
        print(f"\n需人工处理 {len(manual)} 个：")
        for name, why in manual:
            print(f"  {name:<34} {why}")
    if not args.apply:
        print("\n（未加 --apply，仅预览）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
