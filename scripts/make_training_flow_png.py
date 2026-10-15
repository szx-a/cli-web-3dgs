"""生成训练流程 PNG（`docs/training_flow.png`），纯 matplotlib，不依赖 mermaid 渲染器。

图上出现的函数名都取自 `src/gsflow/trainers/gsplat_trainer.py` 的**实际定义**
（可用 `scripts/show_trainer_steps.py` 复核），所以这张图也能当索引用。

⚠️ 版式教训：**不要手工填 y 坐标**。前几版靠肉眼调数值，反复出现"框底被切、
说明文字压到下一行框上"。现在改为**自上而下的游标布局**：每块高度由内容行数算出，
`y` 由游标累减，块间固定间距 —— 增加内容也不会互相重叠。

用法: python scripts/make_training_flow_png.py [--out docs/training_flow.png]
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


class Cursor:
    """自上而下的布局游标：`take(h)` 前进并返回该块底部的 y（轴坐标 y 向上）。"""

    def __init__(self, top: float, gap: float = 0.016):
        self.y = top
        self.gap = gap

    def take(self, h: float) -> float:
        self.y -= h
        bottom = self.y
        self.y -= self.gap
        return bottom


def box(ax, x, y, w, h, text, fc, ec="#334155", fs=8.5, weight="normal", tc="#0f172a"):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.014",
        facecolor=fc, edgecolor=ec, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc, linespacing=1.5)


def arrow(ax, p1, p2, color="#64748b", lw=1.4, ls="-"):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, shrinkA=1, shrinkB=1))


def h_for(lines: int, fs: float = 8.0) -> float:
    """按文本行数估算块高（含上下留白）。"""
    return 0.026 + lines * 0.0215 * (fs / 8.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/training_flow.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(15.5, 11), dpi=150)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.985, L("训练框架 · 流程与顺序红线",
                          "Training Framework - Flow and Ordering Rules"),
            ha="center", va="top", fontsize=17, fontweight="bold", color="#0b1220")
    ax.text(0.5, 0.953, L("src/gsflow/trainers/gsplat_trainer.py    train(cfg, ctx)"
                          "    —— 括号内为实际函数名 / 参数名",
                          "gsplat_trainer.py: train(cfg, ctx)"),
            ha="center", va="top", fontsize=9.5, color="#475569")

    cur = Cursor(top=0.931)

    # ---------------- ①②③ 准备行 ----------------
    prep = [
        ("#e0f2fe", "① 数据准备\n"
                    "find_model_dir()  按“注册图像最多”选 sparse/<best>\n"
                    "load_colmap_data()  读 K / c2w / 稀疏点 → viewmat = inv(c2w)\n"
                    "_load_images()  缩放 + 掩码，K 同步缩放（只缩小不放大）"),
        ("#dcfce7", "② 参数初始化\n"
                    "means = COLMAP 稀疏点（超 cap_max 随机下采样）\n"
                    "_init_scales_nn(k=3) → log(sqrt(mean‖p−knn‖²))，再 repeat(1,3)\n"
                    "quats=单位   opac=logit(0.1)   shs=稀疏点颜色"),
        ("#fef9c3", "③ 优化器\n"
                    "_LR = {means 1.6e-4, scales 5e-3, quats 1e-3,\n"
                    "        opacities lr_opacities, shs 2.5e-3}\n"
                    "schedule_lrs() 手动实现 ExponentialLR\n"
                    "（不能用 torch 调度器对象：密度控制会替换参数对象）"),
    ]
    h_prep = h_for(5, 7.6)
    y_prep = cur.take(h_prep)
    xs = (0.030, 0.3535, 0.677)
    w3 = 0.293
    for i, (fc, txt) in enumerate(prep):
        box(ax, xs[i], y_prep, w3, h_prep, L(txt, txt), fc, fs=7.6)
        if i:
            arrow(ax, (xs[i] - 0.0235, y_prep + h_prep / 2), (xs[i], y_prep + h_prep / 2))

    # ---------------- ④ 训练循环 ----------------
    steps = [
        "取视角\ntrain_idx[step % N]",
        "光栅化\nrasterization()\nrender_mode=RGB",
        "合成背景\nrender += bg·(1−α)",
        "损失\nL1 + SSIM\n(mask_loss_mode)",
        "可选正则\nopacity/scale\n2DGS(默认关)",
        "反向\nbackward()\n+ 缓存 ‖means2d.grad‖",
        "密度控制\n克隆/分裂/裁剪\n(或官方 Strategy)",
        "评估\nevaluate()\n+ 保存最佳检查点",
        "重置不透明度\nreset（可关）",
    ]
    n = len(steps)
    bw, bgap = 0.0925, 0.0075
    notes = [
        ("#9a3412", "顺序红线：密度控制 → 评估/保存 → 不透明度重置。重置必须在保存之后，"
                     "且最后一步跳过重置；\n否则“最终检查点”存的是刚被重置的状态"
                     "（不透明度被钳到 logit(prune_opa*2)，导出近乎全白）。"),
        ("#1e3a8a", "损失口径：mask_loss_mode=composite（默认）把 GT 掩码外合成为背景色、"
                     "在全图求损失；\ninside 是旧行为（掩码外无监督 → 背景里乱长高斯、该空洞处破碎）。"),
    ]
    h_steps = h_for(4, 7.2)
    h_notes = 2 * 0.030
    h_loop = 0.032 + h_steps + 0.016 + 0.020 + h_notes + 0.020
    y_loop = cur.take(h_loop)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.022, y_loop), 0.956, h_loop, boxstyle="round,pad=0.006,rounding_size=0.016",
        facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.4, linestyle="--"))
    ax.text(0.5, y_loop + h_loop - 0.014, L("④ 训练循环   for step in range(iterations)",
                                             "4 Training loop"),
            ha="center", va="top", fontsize=10.5, fontweight="bold", color="#0f172a")

    y_s = y_loop + h_loop - 0.032 - h_steps
    for i, s in enumerate(steps):
        x = 0.036 + i * (bw + bgap)
        fc = "#ffedd5" if i in (3, 6, 8) else "#ffffff"
        ec = "#c2410c" if i in (3, 6, 8) else "#334155"
        box(ax, x, y_s, bw, h_steps, L(s, s), fc, ec=ec, fs=7.2)
        if i:
            arrow(ax, (x - bgap, y_s + h_steps / 2), (x, y_s + h_steps / 2))
    # 循环回箭头（步骤条下方）
    y_ret = y_s - 0.014
    arrow(ax, (0.955, y_s), (0.955, y_ret), color="#c2410c", ls="--", lw=1.3)
    ax.annotate("", xy=(0.115, y_ret), xytext=(0.955, y_ret),
                arrowprops=dict(arrowstyle="-|>", color="#c2410c", lw=1.3,
                                linestyle="--", shrinkA=0, shrinkB=0))
    arrow(ax, (0.115, y_ret), (0.115, y_s), color="#c2410c", ls="--", lw=1.3)
    ax.text(0.52, y_ret, L("下一步（循环）", "next"), ha="center", va="center",
            fontsize=7.5, color="#c2410c",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="#f8fafc", edgecolor="none"))
    # 两条说明（放在回箭头下方）
    for k, (color, txt) in enumerate(notes):
        ax.text(0.5, y_ret - 0.014 - k * 0.030, L(txt, txt), ha="center", va="top",
                fontsize=7.6, color=color)

    # ---------------- ⑤⑥⑦ 产物行 ----------------
    prod = [
        ("#ede9fe", "⑤ 每个评估点保存\n"
                    "save_ply() → point_cloud/iteration_N/point_cloud.ply\n"
                    "metrics.json 记录 best_ply / best_psnr / best_step"),
        ("#ede9fe", "⑥ 收尾\n"
                    "再按 min_opacity_final 删一遍\n"
                    "（Faster-GS 的 MIN_OPACITY_AFTER_TRAINING）\n"
                    "导出阶段优先用 best_ply，避开被重置污染的末次快照"),
        ("#f1f5f9", "⑦ 可插拔后端\n"
                    "trainers/base.py 协议：\n"
                    "run(ctx, data_dir, out_dir, iterations) → dict\n"
                    "gsplat（默认）/ official / external，新增后端不必改 CLI/GUI"),
    ]
    h_prod = h_for(5, 7.6)
    y_prod = cur.take(h_prod)
    for i, (fc, txt) in enumerate(prod):
        box(ax, xs[i], y_prod, w3, h_prod, L(txt, txt), fc, fs=7.6)
        if i:
            arrow(ax, (xs[i] - 0.0235, y_prod + h_prod / 2), (xs[i], y_prod + h_prod / 2))

    # ---------------- 底线提醒 ----------------
    h_bot = h_for(3, 7.8)
    y_bot = cur.take(h_bot)
    box(ax, 0.030, y_bot, 0.94, h_bot,
        L("对比实验纪律：比较两个训练配置，必须让它们共用同一份 04_train/data —— "
          "否则测到的主要是 COLMAP 的运行间差异\n"
          "（实测同一数据集两次 COLMAP 给出 720×720 vs 722×722、fx 853.62 vs 849.34）。"
          "工具：scripts/ab_loss_mode.py、run_export_direct.py",
          "Experimental discipline: share one 04_train/data when comparing configs"),
        "#ffe4e6", ec="#9f1239", fs=7.8)

    print(f"版面底部 y = {y_bot:.3f}（应 >= 0；为负说明内容超出画布）")

    # 画布高度按**实际内容**裁掉底部空白：内容占到 y_bot，标题占顶部 ~0.048。
    used = 1.0 - y_bot + 0.03
    fig.set_size_inches(15.5, max(6.0, 11.0 * used), forward=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    note = "" if HAS_CJK else "  —— 未找到中文字体，已改用英文标签"
    print(f"已生成 {out}  (字体: {FONT}{note})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
