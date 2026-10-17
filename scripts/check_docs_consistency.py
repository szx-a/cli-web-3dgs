"""核对文档与代码一致性（组合检查器，供每次改进后一键运行）。

用户要求："每次改进成功先去检查文档过时和不符的更新。"
本脚本把多项检查串起来，并**先各自自检**（避免"检查器本身错了却报通过"）：

  1. `config.DEFAULTS` vs `TrainConfig`         —— 代码内部两处默认值必须一致
  2. `config/config.toml` vs `config.DEFAULTS`  —— 写出来的配置文件不能落后于代码
  3. `README.md` 参数表 vs `config.DEFAULTS`     —— 文档里的默认值不能过时
  4. `docs/*.md` 的 mermaid 图结构               —— 图坏了在源码里看不出来
  5. **关键概念覆盖**（`check_docs_concepts.py`）—— 新机制/新口径必须被写进对应文档

第 5 项是补上一个真实漏洞：只校验**数值**时会出现
"参数默认值全部一致 ✅ 但文档 0 处提到 `psnr_masked` / `refviewer`" 的局面，
于是有人照着文档就会得出错误结论（`docs/03` G123）。

用法: python scripts/check_docs_consistency.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS = [
    ("代码内部（DEFAULTS vs TrainConfig）", "check_config_sync.py"),
    ("配置文件（config.toml vs DEFAULTS）", "check_config_sync.py"),
    ("文档参数表（README vs DEFAULTS）", "check_readme_params.py"),
    ("文档图（mermaid 结构）", "check_mermaid.py"),
    ("文档关键概念覆盖", "check_docs_concepts.py"),
]


def run(script: str) -> tuple[int, str]:
    p = subprocess.run([PY, str(ROOT / "scripts" / script)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out.strip()


def main() -> int:
    print("=== 文档 / 配置一致性检查 ===\n")
    failed = []
    for label, script in STEPS:
        rc, out = run(script)
        # 取最后一个非空行作为摘要
        tail = [l for l in out.splitlines() if l.strip()]
        summary = tail[-1] if tail else "(无输出)"
        flag = "OK  " if rc == 0 else "**FAIL**"
        print(f"  [{flag}] {label:<34} {summary[:90]}")
        if rc != 0:
            failed.append((label, script, out))

    print()
    if not failed:
        print(f"结论: 文档与代码一致（{len(STEPS)} 项检查全通过）")
        return 0
    print(f"结论: **{len(failed)} 项不一致** —— 需要更新文档或配置：")
    for label, script, out in failed:
        print(f"\n--- {label}（{script}）---")
        print("\n".join(out.splitlines()[-14:]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
