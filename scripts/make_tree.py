"""生成项目文件树（Markdown），便于梳理工程结构。

用法：
    env\\venv\\Scripts\\python.exe scripts\\make_tree.py                 # 打印到屏幕
    env\\venv\\Scripts\\python.exe scripts\\make_tree.py --out docs\\04-文件树.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 这些目录不入树（体积大 / 运行期产物），但会在末尾单独说明
SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
    "env", "downloads", "workspace", "models", "logs", "tools",
    # 用户素材目录：内容是多视角视频/图片，不属于源码结构
    "data",
    # 打包元数据（pip install -e 失败时留下的中间产物）
    "gsflow.egg-info",
}
SKIP_EXT = {".pyc", ".pyo", ".part", ".lnk"}

# 每个顶层目录/文件的一句话说明
NOTES = {
    "AGENTS.md": "给 AI 协作者的既定事实与红线（必读）",
    "README.md": "项目总览、快速开始、目录说明",
    "pyproject.toml": "包元数据与可选依赖（train / segment / dev）",
    "requirements.txt": "核心运行依赖",
    "requirements-gpu.txt": "GPU 与分割后端的可选依赖（torch cu128 等）",
    ".gitignore": "忽略运行期产物与凭据",
    "config": "配置与凭据（credentials.json 不入库）",
    "docs": "环境核查 / 架构 / 操作手册 / 踩坑记录 / 文件树",
    "scripts": "环境搭建、工具下载、验证脚本、启动器",
    "src": "流程核心源码（gsflow 包）",
    "tests": "pytest 测试（适配本机沙箱的临时目录方案）",
    "tools": "自带的 colmap / ffmpeg / 便携 CUDA（体积大，不入树）",
    "workspace": "每个作业一个子目录：01_images_raw … 06_reports",
}


def render(root: Path) -> str:
    lines: list[str] = []

    def walk(d: Path, prefix: str = "") -> None:
        entries = sorted(
            (p for p in d.iterdir() if p.name not in SKIP_DIRS and p.suffix not in SKIP_EXT),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
        for i, p in enumerate(entries):
            last = i == len(entries) - 1
            branch = "└── " if last else "├── "
            note = ""
            if prefix == "" and p.name in NOTES:
                note = f"   # {NOTES[p.name]}"
            elif p.name == "__init__.py":
                note = "   # 包导出"
            lines.append(f"{prefix}{branch}{p.name}{'/' if p.is_dir() else ''}{note}")
            if p.is_dir():
                walk(p, prefix + ("    " if last else "│   "))

    lines.append(f"{root.name}/")
    walk(root)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="写入的文件；留空则打印")
    args = ap.parse_args()
    tree = render(ROOT)

    extra = """
## 未入树的目录（体积大或运行期产物）

| 目录 | 说明 |
|---|---|
| `env/` | 虚拟环境（`venv/`）、缓存、临时目录、便携 CUDA 解包中间产物 |
| `tools/` | 自带外部工具：`colmap/`（COLMAP 3.12）、`ffmpeg/`（FFmpeg 9.0.2）、`cuda/`（便携 CUDA 12.8，358MB） |
| `downloads/` | 下载的安装包（COLMAP / FFmpeg / CUDA 3.2GB） |
| `models/` | 模型权重：FastSAM、rembg 的 ONNX、HF 缓存 |
| `workspace/` | 每个作业的产物，结构见 `docs/02-操作手册.md` |
| `logs/` | 全局日志：环境自检报告、CUDA 安装日志、g++ 构建日志 |

## 作业产物结构（workspace/<job_id>/）

```
<job_id>/
├─ job.json                          参数快照 + 各阶段状态（唯一状态源）
├─ logs/events.jsonl                 全部事件（CLI 与 GUI 共用）
├─ logs/colmap.log 等                外部工具的原始输出
├─ 01_images_raw/                    抽帧或导入后的原始图 + _frames.json
├─ 02_masks/                         掩码（两种命名）+ _overlay/ 效果图 + manifest.json
├─ 02_images_clean/                  ★ 清理后的图（COLMAP 的输入）
├─ 03_colmap/                        database.db / sparse/ / sparse_txt/ / report.json
├─ 04_train/                         data/（去畸变）+ point_cloud/iteration_N/point_cloud.ply
├─ 05_export/                        model.glb / model.obj / viewer.html / preview_*.png
└─ 06_reports/                       report.md + report.json
```
"""
    text = "# gsflow 项目文件树\n\n> 由 `scripts/make_tree.py` 生成，改动目录结构后可重新生成。\n\n```\n" \
           + tree + "\n```\n" + extra

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"已写入 {out}")
    else:
        print(tree)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
