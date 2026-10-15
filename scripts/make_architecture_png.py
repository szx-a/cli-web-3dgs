"""生成架构总览 PNG（不依赖 mermaid 渲染器，纯 matplotlib）。

为什么单独出 PNG：md 里的 mermaid 需要 GitHub / VS Code / Typora 等**支持 mermaid 的阅读器**
才显示；直接打开 md 或转发给别人时图是看不见的。PNG 一定能看。

图上出现的**函数名/文件名都取自代码实际定义**
（可用 `scripts/show_trainer_steps.py` 复核），因此这张图也能当索引用。

用法: python scripts/make_architecture_png.py [--out docs/architecture.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def pick_font() -> tuple[str, bool]:
    """选一个能显示中文的字体；找不到就退化为英文标签（避免方框乱码）。"""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if cand in names:
            return cand, True
    return "DejaVu Sans", False


FONT, HAS_CJK = pick_font()
plt.rcParams["font.family"] = FONT
plt.rcParams["axes.unicode_minus"] = False


def L(zh: str, en: str) -> str:
    return zh if HAS_CJK else en


def box(ax, x, y, w, h, text, fc, ec="#334155", fs=9, weight="normal", tc="#0f172a"):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.018",
        facecolor=fc, edgecolor=ec, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc, linespacing=1.45)


def arrow(ax, p1, p2, color="#64748b", lw=1.4, ls="-"):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, shrinkA=2, shrinkB=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/architecture.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(15, 10), dpi=150)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.982, L("3DGS 全流程工程 · 架构总览", "3DGS Pipeline - Architecture Overview"),
            ha="center", va="top", fontsize=17, fontweight="bold", color="#0b1220")
    ax.text(0.5, 0.947, L("一套流程核心，两个入口（CLI / Web）；阶段之间只通过磁盘产物通信",
                          "One core, two front-ends; stages communicate via disk artifacts only"),
            ha="center", va="top", fontsize=10, color="#475569")

    # ---------------- 入口层 ----------------
    box(ax, 0.05, 0.855, 0.42, 0.062,
        L("CLI    cli.py (argparse)\n控制台事件 sink", "CLI    cli.py (argparse)"),
        "#e0f2fe", fs=10, weight="bold")
    box(ax, 0.53, 0.855, 0.42, 0.062,
        L("Web GUI    web/app.py (FastAPI + SSE)\nweb/static/{app.js, viewer.js}（无构建步骤）",
          "Web GUI    web/app.py (FastAPI + SSE)"),
        "#e0f2fe", fs=10, weight="bold")

    # ---------------- 编排层 ----------------
    box(ax, 0.05, 0.715, 0.90, 0.108,
        L("编排层    pipeline/orchestrator.py: JobSpec / Orchestrator · list_jobs\n"
          "pipeline/context.py: StageContext（受控世界：cfg / cfgt / 事件 / 阶段目录）\n"
          "events.py: EventBus（JSONL 落盘 + 控制台 sink）    ·    pipeline/registry.py: 阶段注册与依赖闭包",
          "Orchestration: orchestrator / context / events / registry"),
        "#fef9c3", fs=9)

    # ---------------- 阶段层 ----------------
    stages = [
        ("① frames\nstages/frames.py", "1 frames"),
        ("② mask\nstages/mask.py", "2 mask"),
        ("③ colmap\nstages/colmap.py", "3 colmap"),
        ("④ train\nstages/train.py", "4 train"),
        ("⑤ export\nstages/export.py", "5 export"),
        ("⑥ metrics\nstages/metrics.py", "6 metrics"),
    ]
    xs = [0.048 + i * 0.153 for i in range(6)]
    w = 0.142
    for i, (zh, en) in enumerate(stages):
        fc = "#bbf7d0" if i == 3 else "#dcfce7"
        box(ax, xs[i], 0.598, w, 0.070, L(zh, en), fc, fs=8.5, weight="bold")
        if i:
            arrow(ax, (xs[i - 1] + w, 0.633), (xs[i], 0.633))
    ax.text(0.5, 0.577, L("六个阶段顺序执行；产物已存在则自动跳过，任一阶段可单独重跑",
                          "six sequential stages; each re-runnable"),
            ha="center", va="top", fontsize=9, color="#475569", style="italic")

    # ---------------- 能力层 ----------------
    caps = [
        ("segment/", L("分割后端\nbase.py 协议\nclassical / rembg\nfastsam / sam2 / sam3",
                       "segmentation backends")),
        ("trainers/", L("训练后端\nbase.py 协议\ngsplat_backend（默认）\nofficial / external",
                        "training backends")),
        ("gsdata/", L("colmap_io.py\n读写 + 去畸变\nmesh_export.py\n反投影 / 泊松 / TSDF",
                      "COLMAP IO, mesh export")),
        ("tools/", L("locate.py 定位\nrunner.py 执行包装\ndoctor.py 自检",
                     "tool adapters")),
        ("imageops.py", L("掩码形态学\n连通域\n黑背景合成",
                          "mask morphology")),
    ]
    cw = 0.174
    for i, (name, desc) in enumerate(caps):
        x = 0.048 + i * (cw + 0.0115)
        box(ax, x, 0.412, cw, 0.118, f"{name}\n{desc}", "#f1f5f9", fs=8)

    # ---------------- 产物层 ----------------
    box(ax, 0.048, 0.232, 0.352, 0.118,
        L("作业目录   workspace/<job_id>/\n"
          "01_images_raw  02_masks  02_images_clean\n"
          "03_colmap  04_train  05_export  06_reports\n"
          "+ job.json（唯一状态源）",
          "workspace/<job_id>/  01..06 + job.json"),
        "#ede9fe", fs=8.5)
    box(ax, 0.425, 0.232, 0.252, 0.118,
        L("三种查看形态\n① 点云 GLB（3DGS 原始形态）\n② 网格 PLY / GLB / OBJ\n③ 带贴图网格 GLB",
          "three view modes"),
        "#fce7f3", fs=8.5)
    box(ax, 0.702, 0.232, 0.248, 0.118,
        L("离线单文件查看器\nviewer_standalone_*.html\n内嵌 viewer.js + base64 模型\n双击即看，不需要服务",
          "standalone viewer (single HTML)"),
        "#fce7f3", fs=8.5)
    arrow(ax, (0.400, 0.291), (0.425, 0.291))
    arrow(ax, (0.677, 0.291), (0.702, 0.291))

    # ---------------- 红线 ----------------
    box(ax, 0.048, 0.058, 0.435, 0.130,
        L("红线 ①  流程核心（pipeline / stages / segment / trainers）\n"
          "不得 import web 或 cli\n→ 换掉 GUI 不影响流程",
          "Rule 1: core must not import web/cli"),
        "#ffe4e6", ec="#9f1239", fs=8.5)
    box(ax, 0.515, 0.058, 0.435, 0.130,
        L("红线 ②  阶段之间不传内存对象\n只传磁盘路径 + job.json\n→ 关掉 GUI 明天接着跑；两个入口行为一致",
          "Rule 2: stages talk via disk + job.json"),
        "#ffe4e6", ec="#9f1239", fs=8.5)

    # ---------------- 连线 ----------------
    arrow(ax, (0.26, 0.855), (0.26, 0.824), color="#94a3b8")
    arrow(ax, (0.74, 0.855), (0.74, 0.824), color="#94a3b8")
    arrow(ax, (0.50, 0.715), (0.50, 0.669), color="#94a3b8")
    arrow(ax, (0.50, 0.598), (0.50, 0.531), color="#94a3b8")
    # 标注放**左侧留白**：放在箭头附近会压在阶段框上（第一版就是这样重叠的）
    ax.text(0.014, 0.634, L("被编排器调度", "scheduled"), fontsize=8,
            color="#64748b", va="center", ha="left")
    arrow(ax, (0.50, 0.412), (0.50, 0.351), color="#94a3b8")
    ax.text(0.014, 0.533, L("各阶段读写产物", "read/write"), fontsize=8,
            color="#64748b", va="center", ha="left")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    note = "" if HAS_CJK else "  —— 未找到中文字体，已改用英文标签"
    print(f"已生成 {out}  (字体: {FONT}{note})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
