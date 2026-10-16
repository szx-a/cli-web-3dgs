"""验证 `check_readme_params.py` **在文档确实过时时会报错**（负向测试）。

为什么需要：一个"永远通过"的检查器和没有检查器一样糟，甚至更糟 ——
它给出虚假的安全感。本会话已多次踩"工具本身有问题"（读错 API 键名、
用 HEAD 查 GET 端点、拿环绕视角当训练视角、22 个脚本用 sh_degree=0）。
所以每个检查器都必须证明**它在已知坏输入上会失败**。

做法：把 README 复制到临时文件，把某一行的默认值改成错的值，
让检查器指向该临时文件，断言它返回非 0。

用法: python scripts/selftest_doc_checker.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
README = ROOT / "README.md"
TMPDIR = ROOT / "env" / "tmp"


def run_on(path: Path) -> int:
    """让检查器检查指定文件（通过临时替换 README 实现，检查完恢复）。"""
    backup = README.read_bytes()
    try:
        README.write_bytes(path.read_bytes())
        p = subprocess.run([PY, str(ROOT / "scripts" / "check_readme_params.py")],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=str(ROOT))
        return p.returncode
    finally:
        README.write_bytes(backup)


def main() -> int:
    TMPDIR.mkdir(parents=True, exist_ok=True)
    text = README.read_text(encoding="utf-8")

    print("=== 负向测试：故意改坏 README，检查器必须报错 ===\n")
    bad = 0

    # 用例 1：把 poisson_depth 改成错的值
    f1 = TMPDIR / "readme_bad_depth.md"
    f1.write_text(re.sub(r"(\| `export\.mesh_poisson_depth` \|\s*\*{0,2})8",
                         r"\g<1>9", text), encoding="utf-8")
    rc = run_on(f1)
    ok = rc != 0
    print(f"  [{'OK' if ok else '**FAIL**'}] depth 写成 9（实际 8）⇒ 检查器退出码 {rc}"
          f"（应非 0）")
    bad += 0 if ok else 1

    # 用例 2：删掉一整行
    f2 = TMPDIR / "readme_missing_row.md"
    f2.write_text(re.sub(r"^\| `export\.mesh_crop_quantile`.*\n", "", text, flags=re.M),
                  encoding="utf-8")
    rc = run_on(f2)
    ok = rc != 0
    print(f"  [{'OK' if ok else '**FAIL**'}] 删掉 crop_quantile 行 ⇒ 退出码 {rc}（应非 0）")
    bad += 0 if ok else 1

    # 用例 3：把布尔值写反
    f3 = TMPDIR / "readme_bad_bool.md"
    f3.write_text(re.sub(r"(\| `export\.mesh_texture` \|\s*)false", r"\g<1>true", text),
                  encoding="utf-8")
    rc = run_on(f3)
    ok = rc != 0
    print(f"  [{'OK' if ok else '**FAIL**'}] mesh_texture 写成 true（实际 false）⇒ 退出码 {rc}"
          f"（应非 0）")
    bad += 0 if ok else 1

    # 用例 4：正例 —— 未改动的 README 必须通过
    rc = run_on(README)
    ok = rc == 0
    print(f"  [{'OK' if ok else '**FAIL**'}] 未改动的 README ⇒ 退出码 {rc}（应为 0）")
    bad += 0 if ok else 1

    for f in (f1, f2, f3):
        f.unlink(missing_ok=True)

    print(f"\n结论: {'检查器在坏输入上确实会失败（有效）' if bad == 0 else f'**{bad} 项失败**'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
