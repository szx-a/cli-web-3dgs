"""把 00-环境核查与实施规划.md 里与现状不符的事实修正（幂等）。

为什么单独写脚本：这些字符串里含反引号，用 PowerShell here-string 传递时
反引号会被当成转义符吃掉，替换静默失效（我已因此白跑两次）。
"""
from __future__ import annotations

import sys
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "00-环境核查与实施规划.md"

PAIRS = [
    (
        "**修正**：用已装的 `uv` 拉一个**独立 Python 3.11**，在本工作区建 `.venv`，"
        "与系统 Python 完全隔离（同时满足 AGENTS.md 要求）。",
        "**修正**：用已装的 `uv` 拉一个**独立 Python**（计划 3.11，**实际 3.13.11**），"
        "在本工作区建 `env/venv`，与系统 Python 完全隔离（满足 AGENTS.md 要求）。"
        "源码用 `gsflow-dev.pth` 挂载，**不走 `pip install -e`**（PEP517 在本机报 WinError 5）。",
    ),
    (
        "- 兜底：允许在 `config/pipeline.toml` 里填写任意自备路径（`colmap.exe` / `ffmpeg.exe`）。",
        "- 兜底：允许在 `config/config.toml` 里填写任意自备路径（`colmap.exe` / `ffmpeg.exe`）。",
    ),
    (
        "4. 配置单一来源：`config/pipeline.toml`（无凭据） + `config/credentials.json`"
        "（有凭据，0600/ACL 收紧）。",
        "4. 配置单一来源：`config/config.toml`（无凭据，**已实现**） + "
        "`config/credentials.json`（有凭据，已 gitignore）。",
    ),
    (
        "1. `UV_CACHE_DIR` 指到工作区，用 uv 建 `.venv`（Python 3.11）。",
        "1. `UV_CACHE_DIR` 指到工作区，用 uv 建 `env/venv`（**实际 Python 3.13.11**）。",
    ),
    (
        "- `config/pipeline.toml` + 凭据读写（GUI/CLI 共用）。",
        "- `config/config.toml` + 凭据读写（GUI/CLI 共用）—— **已实现**"
        "（`credentials.py` 统一入口）。",
    ),
    (
        "│   ├─ pipeline.toml                (可提交)",
        "│   ├─ config.toml                  (可提交)   ← 已实现",
    ),
    (
        "| uv | ✅ 0.11.19 | 可用它拉取独立 Python 3.11 解释器（推荐） |",
        "| uv | ✅ 0.11.19 | 用它拉独立解释器（**实际 3.13.11**；"
        "torch 2.11+cu128 与 gsplat 1.5.3 均可用）|",
    ),
]


def main() -> int:
    t = DOC.read_text(encoding="utf-8")
    changed = 0
    missing = []
    for old, new in PAIRS:
        if new in t:
            continue
        if old in t:
            t = t.replace(old, new, 1)
            changed += 1
        else:
            missing.append(old[:48])
    DOC.write_text(t, encoding="utf-8")
    print(f"  替换 {changed} 处")
    for m in missing:
        print(f"  **未命中**: {m}")
    left = [s for s in ("pipeline.toml", "Python 3.11") if s in t]
    print(f"  残留: {left if left else '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
