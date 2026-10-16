"""把诊断脚本里的「rgb → DC-only 颜色」改为**直接使用完整 SH**。

事故背景：这些脚本从 `ExportStage._read_ply` 拿到了完整 `shs`（含 45 个高阶分量），
却**自己重新造**了一份只有 DC 的颜色：
    shs = ((rgb - 0.5) / 0.28209479177387814).reshape(n, 1, 3)
再配合 `sh_degree=0` 渲染 ⇒ 实测**低估 1.91 dB**，并抹掉视角相关外观
（正是感知损失/几何正则要改善的部分）⇒ 制造假阴性。

关键点：**`shs` 与 `sh_degree` 必须一起改** ——
只改 `sh_degree=3` 而颜色仍是 [N,1,3] 会形状不匹配；
只改颜色而 `sh_degree=0` 会被 gsplat 忽略高阶。

用法: python scripts/fix_diag_sh_full.py [--apply]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SKIP = {"fix_diag_sh_degree.py", "fix_diag_sh_full.py", "refactor_read_ply.py",
        "render_train_views.py", "check_sh_export.py"}

# 匹配"从 rgb 造 DC-only SH"的赋值（变量名任意，右侧含 SH 常数或 0.2820...）
DC_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<var>\w+)\s*=\s*\(\([^)]*\brgb\b[^)]*\)\s*/\s*"
    r"(?:0\.28209479177387814|SH_C0|C0)\)\s*\.reshape\([^)]*\)\s*$",
    re.M,
)


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
        m = DC_RE.search(text)
        if not m:
            manual.append((p.name, "未找到 DC-only 颜色构造式"))
            continue
        indent, var = m.group("indent"), m.group("var")
        # 找到 gauss 变量名（_read_ply 的返回变量）
        gm = re.search(r"^[ \t]*(\w+)\s*=\s*[^=\n]*_read_ply\(", text, re.M)
        if not gm:
            manual.append((p.name, "找不到 gauss 变量"))
            continue
        gauss = gm.group(1)

        new_text = DC_RE.sub(
            f"{indent}# 直接用**完整 SH**（含高阶）：自己从 rgb 造 DC-only 会低估约 1.9 dB\n"
            f"{indent}{var} = __import__('torch').as_tensor("
            f"{gauss}[\"shs\"], dtype=__import__('torch').float32, device=\"cuda\")\n"
            f"{indent}_sh_deg = int({gauss}.get(\"sh_degree\", 0) or 0)",
            text, count=1)
        n = new_text.count("sh_degree=0")
        new_text = new_text.replace("sh_degree=0", "sh_degree=_sh_deg")
        if new_text == text:
            manual.append((p.name, "替换无变化"))
            continue
        done.append((p.name, gauss, var, n))
        if args.apply:
            p.write_text(new_text, encoding="utf-8")

    print(f"{'已修改' if args.apply else '将修改'} {len(done)} 个脚本：")
    for name, gauss, var, n in done:
        print(f"  {name:<32} {var} <- {gauss}['shs']，{n} 处渲染改用真实阶数")
    if manual:
        print(f"\n需人工处理 {len(manual)} 个：")
        for name, why in manual:
            print(f"  {name:<32} {why}")
    if not args.apply:
        print("\n（未加 --apply，仅预览）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
