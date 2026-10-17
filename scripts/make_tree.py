"""生成文件树文档（`docs/04-文件树.md`），可重复运行、不手工维护。

要点：
* 只列**有意义的**内容：跳过 `.git` / `__pycache__` / `env`（虚拟环境）/
  `workspace` 里成千上万的产物，但把它们的**规模与用途**用一行说明，避免"看不出体量"。
* Python 文件带**行数**，便于判断模块分量。
* 结尾附**架构红线**与模块职责，让文件树能当索引用。

用法: python scripts/make_tree.py [--out docs/04-文件树.md]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 目录 -> 是否递归展开、说明
DIR_NOTES = {
    "src/gsflow": "流程核心（不依赖 UI）",
    "src/gsflow/pipeline": "编排：阶段注册、上下文、事件、作业状态",
    "src/gsflow/stages": "六个阶段：frames/mask/colmap/train/export/metrics",
    "src/gsflow/trainers": "训练后端（gsplat 默认）",
    "src/gsflow/gsdata": "数据 IO 与几何（COLMAP 读写、网格提取）",
    "src/gsflow/segment": "可插拔分割后端",
    "src/gsflow/tools": "外部工具适配（COLMAP/FFmpeg 定位与调用）",
    "src/gsflow/web": "FastAPI Web GUI（+ static 前端）",
    "src/gsflow/web/static": "前端（无构建步骤的纯 JS/CSS）",
    "src/gsflow/web/static/vendor": "离线第三方前端库（three.js + gaussian-splats-3d）",
    "scripts": "运维/诊断/实验脚本（不参与产品运行）",
    "tests": "pytest 回归",
    "docs": "文档",
    "config": "config.toml（写配置即改行为）",
    "data": "输入素材（视频）",
    "tools": "第三方可执行（COLMAP / FFmpeg），已解压免安装",
    "models": "模型权重（分割后端等）",
    "env": "Python 虚拟环境与临时目录（不入库）",
    "workspace": "每个作业一个目录：01..05 阶段产物 + logs",
    "downloads": "下载缓存",
    "logs": "运行日志",
}
SKIP_NAMES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
              "node_modules", ".ipynb_checkpoints"}
# 这些目录只统计、不展开（体量太大或内容无信息量）
COLLAPSE = {"env", "workspace", "downloads", "logs", "models", "tools/ffmpeg", "tools/colmap",
            # vendor 里是两个 ~600KB~1.2MB 的压缩库，展开没信息量，但**必须出现在树里**：
            # 高斯渲染完全依赖它们（离线可用），早期版本的文件树漏掉了这一层。
            "src/gsflow/web/static/vendor"}


def py_lines(p: Path) -> int:
    try:
        return len(p.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return 0


def dir_size(p: Path) -> tuple[int, int]:
    files = dirs = 0
    try:
        for x in p.rglob("*"):
            if any(s in x.parts for s in SKIP_NAMES):
                continue
            if x.is_file():
                files += 1
            elif x.is_dir():
                dirs += 1
    except OSError:
        pass
    return files, dirs


def walk(p: Path, prefix: str, rel: str, out: list[str], max_depth: int, depth: int = 0) -> None:
    if depth > max_depth:
        return
    try:
        items = [x for x in p.iterdir() if x.name not in SKIP_NAMES]
    except OSError:
        return
    items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
    for i, x in enumerate(items):
        last = i == len(items) - 1
        branch = "└── " if last else "├── "
        r = f"{rel}/{x.name}" if rel else x.name
        note = DIR_NOTES.get(r, "")
        if x.is_dir():
            # `tools/` 下面是第三方可执行（COLMAP 几百个文件），整棵子树只统计不展开；
            # 否则 `--depth` 一大，文件树会多出 600+ 行噪声，把真正的代码结构淹掉。
            collapse = (x.name in COLLAPSE or r in COLLAPSE
                        or r == "tools" or r.startswith("tools/"))
            if collapse:
                f, d = dir_size(x)
                out.append(f"{prefix}{branch}{x.name}/   "
                           f"[{f} 文件, {d} 目录]{('  —— ' + note) if note else ''}")
            else:
                out.append(f"{prefix}{branch}{x.name}/"
                           f"{('  —— ' + note) if note else ''}")
                walk(x, prefix + ("    " if last else "│   "), r, out, max_depth, depth + 1)
        else:
            if x.suffix == ".py":
                out.append(f"{prefix}{branch}{x.name}   ({py_lines(x)} 行)"
                           f"{('  —— ' + note) if note else ''}")
            else:
                kb = x.stat().st_size / 1024
                size = f"{kb:.1f} KB" if kb < 1024 else f"{kb / 1024:.1f} MB"
                out.append(f"{prefix}{branch}{x.name}   ({size})"
                           f"{('  —— ' + note) if note else ''}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/04-文件树.md")
    ap.add_argument("--depth", type=int, default=5)
    args = ap.parse_args()

    lines: list[str] = []
    lines.append("# 04 · 文件树")
    lines.append("")
    lines.append("> 本文件由 `python scripts\\make_tree.py` **自动生成**，请勿手工编辑。")
    lines.append("> Python 文件标注行数；`env/` `workspace/` 等只统计不展开。")
    lines.append("")
    lines.append("```text")
    lines.append("3dgs-and-ai/")
    body: list[str] = []
    walk(ROOT, "", "", body, args.depth)
    lines.extend(body)
    lines.append("```")
    lines.append("")

    # 训练包单独一张（最常被查阅）
    lines.append("## 训练框架（`src/gsflow/trainers/`）")
    lines.append("")
    lines.append("```text")
    lines.append("trainers/")
    body2: list[str] = []
    walk(ROOT / "src" / "gsflow" / "trainers", "", "", body2, 2)
    lines.extend(body2)
    lines.append("```")
    lines.append("")

    # 模块职责（人工维护，改动少）
    lines.append("## 模块职责")
    lines.append("")
    lines.append("| 模块 | 职责 | 关键约束 |")
    lines.append("|---|---|---|")
    lines.append("| `pipeline/` | 阶段注册、编排、事件总线、作业状态 | **不依赖 UI**；阶段间只通过磁盘产物 + `job.json` 通信 |")
    lines.append("| `stages/` | frames → mask → colmap → train → export → metrics | 每个阶段可单独重跑，产物落 `0N_*` 目录 |")
    lines.append("| `trainers/` | 训练后端（gsplat 默认 / official / external） | 实现 `base.py` 协议即可插拔 |")
    lines.append("| `gsdata/` | COLMAP 读写、深度反投影、泊松/TSDF 网格提取 | 纯几何，不依赖训练栈 |")
    lines.append("| `segment/` | 分割后端（rembg/classical/FastSAM/SAM2/SAM3） | 实现 `Segmenter` 协议 |")
    lines.append("| `web/` | FastAPI + 静态前端 | 与 CLI **共用同一 Orchestrator**（同构红线） |")
    lines.append("| `web/static/refviewer.js` | 高斯显示的**适配层** | 渲染像素交给 "
                 "vendored `gaussian-splats-3d`，本文件不含任何着色器代码（`docs/03` G116/G121）|")
    lines.append("| `web/static/viewer.js` | WebGL1 查看器：glb / **普通** ply（网格、点云）| "
                 "遇到 3DGS PLY 或 `.splat` **明确报错**并指向 refviewer |")
    lines.append("| `web/static/splat.legacy.js` | 已下线的自研泼溅器（归档）| **不被 index.html 加载**；"
                 "保留仅供追溯 |")
    lines.append("| `cli.py` | 命令行入口 | 与 GUI 等价，GUI 会打印等价 CLI 命令 |")
    lines.append("")

    out = ROOT / args.out
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成 {out}  ({len(lines)} 行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
