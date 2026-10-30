"""生成 README 用的对比曲线图（数据全部来自本仓库实测记录）。

四张图：
1. `fig_grad_dose.png`   —— λ_grad 剂量-响应（PSNR/LPIPS + 高频指标），15k，8 留出视角
2. `fig_depthvar_dose.png` —— λ_depth_var 剂量-响应（深度方差/p90 + LPIPS），15k
3. `fig_60k_compare.png` —— 60k 三配置对比（v4 / solid-v2 / solid-v3）
4. `fig_lpips_cap.png`   —— LPIPS 权重 × 容量 2×2 标定，30k

数据出处：docs/03 G162、G164、G166、G167 与 README 3.9/3.10。
坐标轴标签用英文 —— 避免不同机器缺 CJK 字体导致豆腐块。

用法：python scripts/make_readme_figures.py
输出：docs/figs/*.png（同时打印每张图的落盘路径）
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COL = {"psnr": "#1f77b4", "lpips": "#d62728", "hf": "#2ca02c", "var": "#9467bd", "gray": "#555555"}


def _save(fig, name: str) -> None:
    p = OUT / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  已写出 {p.relative_to(ROOT)}")


# ---------------------------------------------------------------- 1) λ_grad
def fig_grad_dose() -> None:
    lam = [0.0, 0.1, 0.3, 1.0]
    psnr = [29.19, 29.91, 31.22, 33.94]
    lpips = [0.2407, 0.2350, 0.2248, 0.2041]
    hf_corr = [0.750, 0.785, 0.807, 0.839]
    hf_ratio = [1.201, 1.094, 1.033, 0.989]
    gauss = [224885, 236700, 257712, 325441]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    x = np.arange(len(lam))
    ax1.plot(x, psnr, "o-", color=COL["psnr"], label="masked PSNR (dB)")
    ax1.set_ylabel("masked PSNR (dB)", color=COL["psnr"])
    ax1.tick_params(axis="y", labelcolor=COL["psnr"])
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{v:g}" for v in lam])
    ax1.set_xlabel(r"$\lambda_{\mathrm{grad}}$")
    ax1b = ax1.twinx()
    ax1b.plot(x, lpips, "s--", color=COL["lpips"], label="LPIPS")
    ax1b.set_ylabel("LPIPS (lower is better)", color=COL["lpips"])
    ax1b.tick_params(axis="y", labelcolor=COL["lpips"])
    ax1b.spines["top"].set_visible(False)
    ax1.set_title("Detail-gradient weight: fidelity", fontsize=10)

    ax2.plot(x, hf_corr, "o-", color=COL["hf"], label="high-freq correlation")
    ax2.plot(x, hf_ratio, "s--", color=COL["gray"], label="high-freq energy ratio")
    ax2.axhline(1.0, color=COL["gray"], lw=0.8, ls=":", label="ratio = 1 (ideal)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{v:g}" for v in lam])
    ax2.set_xlabel(r"$\lambda_{\mathrm{grad}}$")
    ax2.set_title("Detail-gradient weight: texture", fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    ax2.set_ylim(0.9, 1.25)
    _save(fig, "fig_grad_dose.png")


# ---------------------------------------------------------------- 2) λ_depth_var
def fig_depthvar_dose() -> None:
    lam = [0.0, 0.3, 1.0, 3.0]
    var = [0.00371, 0.00241, 0.00150, 0.00089]
    p90 = [0.0297, 0.0109, 0.0055, 0.0032]
    lpips = [0.2401, 0.2410, 0.2408, 0.2414]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    x = np.arange(len(lam))
    ax1.semilogy(x, var, "o-", color=COL["var"], label="median depth variance")
    ax1.semilogy(x, p90, "s--", color=COL["gray"], label="p90 depth variance")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{v:g}" for v in lam])
    ax1.set_xlabel(r"$\lambda_{\mathrm{var}}$")
    ax1.set_ylabel("depth variance (scene units$^2$)")
    ax1.set_title("Depth-variance weight: geometric spread", fontsize=10)
    ax1.legend(fontsize=8, frameon=False)

    ax2.plot(x, lpips, "o-", color=COL["lpips"])
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{v:g}" for v in lam])
    ax2.set_xlabel(r"$\lambda_{\mathrm{var}}$")
    ax2.set_ylabel("LPIPS")
    ax2.set_ylim(min(lpips) - 0.002, max(lpips) + 0.002)
    ax2.set_title("Depth-variance weight: cost is in the noise (15k)", fontsize=10)
    _save(fig, "fig_depthvar_dose.png")


# ---------------------------------------------------------------- 3) 60k 对比
def fig_60k_compare() -> None:
    names = ["v4\n(no grad)", "solid-v2\n(depth-var)", "solid-v3\n(grad 1.0)"]
    psnr = [30.56, 30.54, 35.38]
    lpips = [0.2353, 0.2399, 0.2116]
    var = [0.01448, 0.00186, 0.00224]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    x = np.arange(len(names))
    axes[0].bar(x, psnr, color=COL["psnr"], width=0.55)
    axes[0].set_title("masked PSNR (dB)", fontsize=10)
    axes[1].bar(x, lpips, color=COL["lpips"], width=0.55)
    axes[1].set_title("LPIPS (lower is better)", fontsize=10)
    axes[2].bar(x, var, color=COL["var"], width=0.55)
    axes[2].set_title("median depth variance", fontsize=10)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
    for ax, vals, fmt in zip(axes, (psnr, lpips, var), ("{:.2f}", "{:.4f}", "{:.5f}")):
        for xi, v in zip(x, vals):
            ax.annotate(fmt.format(v), (xi, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8)
    fig.suptitle("60k training: 8 held-out views, same probe", fontsize=10)
    _save(fig, "fig_60k_compare.png")


# ---------------------------------------------------------------- 4) LPIPS × cap
def fig_lpips_cap() -> None:
    caps = ["200k", "400k"]
    lam = ["0.05", "0.10"]
    psnr = np.array([[30.07, 30.00], [30.17, 30.14]])
    lpips = np.array([[0.2381, 0.2388], [0.2387, 0.2386]])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for ax, data, title, cmap, fmt in (
        (axes[0], psnr, "masked PSNR (dB)", "Blues", "{:.2f}"),
        (axes[1], lpips, "LPIPS (lower is better)", "Reds_r", "{:.4f}"),
    ):
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(lam)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(caps)
        ax.set_xlabel(r"$\lambda_{\mathrm{lpips}}$")
        ax.set_ylabel("cap_max")
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                        color="black", fontsize=9)
    fig.suptitle("LPIPS weight x capacity (30k, 8 held-out views)", fontsize=10)
    _save(fig, "fig_lpips_cap.png")


if __name__ == "__main__":
    fig_grad_dose()
    fig_depthvar_dose()
    fig_60k_compare()
    fig_lpips_cap()
