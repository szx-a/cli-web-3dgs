"""生成"本轮新测数据"的趋势图：LPIPS / PSNR 取舍 + 三档配置对比。

数据全部来自本会话实测（img5，8 个留出视角，论文对齐 LPIPS）。
输出 `docs/figs/fig_tradeoff.png`，供 README §3.15 引用。

用法：python scripts/make_readme_figures_v2.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "docs" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# (标签, LPIPS, PSNR(m), 高斯数, 颜色, 说明)
DATA = [
    ("solid 60k\n(builtin, 旧预设)", 0.2116, 35.38, 405_629, "#8c8c8c"),
    ("official 60k\n(builtin)", 0.2076, 34.93, 465_388, "#2b6cb0"),
    ("official 60k\n(official+AbsGS)", 0.2095, 35.65, 600_000, "#2f9e44"),
    ("mcmc 60k\n(SGLD)", 0.2053, 32.32, 600_000, "#c92a2a"),
    ("mcmc 60k\n(noise 1e5)", 0.2074, 32.33, 600_000, "#e8590c"),
    ("100k 步\n(builtin)", 0.2131, 36.06, 600_412, "#7048e8"),
]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

# ---- (1) \u53d6\u820d\u6563\u70b9\uff1a\u8d8a\u5de6\u4e0a\u8d8a\u597d ----
ax = axes[0]
xs = [d[1] for d in DATA]
ys = [d[2] for d in DATA]
for label, lp, ps, n, color in DATA:
    ax.scatter(lp, ps, s=110, color=color, zorder=3, edgecolor="white", linewidth=1.2)
    ax.annotate(label, (lp, ps), textcoords="offset points", xytext=(9, 6),
                fontsize=7.5, color="#333333")
ax.set_xlabel("LPIPS（越低越好）")
ax.set_ylabel("掩码 PSNR / dB（越高越好）")
ax.set_title("(a) 配置取舍：左上角最好（左=LPIPS 低，上=PSNR 高）")
ax.grid(alpha=0.25, linestyle=":")

# ---- (2) LPIPS \u6761\u5f62\uff08\u6309\u5347\u5e8f\uff09----
ax = axes[1]
order = sorted(DATA, key=lambda d: d[1])
labels = [d[0].replace("\n", " ") for d in order]
vals = [d[1] for d in order]
colors = [d[4] for d in order]
bars = ax.barh(range(len(order)), vals, color=colors, alpha=0.88)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(labels, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlim(0.19, 0.22)
ax.set_xlabel("LPIPS（越低越好）")
ax.set_title("(b) LPIPS 排序")
for i, v in enumerate(vals):
    ax.text(v + 0.0004, i, f"{v:.4f}", va="center", fontsize=7.5)
ax.grid(axis="x", alpha=0.25, linestyle=":")

# ---- (3) PSNR \u6761\u5f62\uff08\u6309\u964d\u5e8f\uff09----
ax = axes[2]
order = sorted(DATA, key=lambda d: -d[2])
labels = [d[0].replace("\n", " ") for d in order]
vals = [d[2] for d in order]
colors = [d[4] for d in order]
ax.barh(range(len(order)), vals, color=colors, alpha=0.88)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(labels, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlim(30, 37)
ax.set_xlabel("掩码 PSNR / dB（越高越好）")
ax.set_title("(c) PSNR 排序")
for i, v in enumerate(vals):
    ax.text(v + 0.08, i, f"{v:.2f}", va="center", fontsize=7.5)
ax.grid(axis="x", alpha=0.25, linestyle=":")

fig.suptitle("img5 实测：三档配置的 LPIPS / PSNR 取舍（2026-11-01，8 留出视角）", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = FIGS / "fig_tradeoff.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"  已写 {out}（{out.stat().st_size/1024:.0f} KB）")
