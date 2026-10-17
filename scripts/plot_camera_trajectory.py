"""把数据集的相机轨迹画出来 —— 判断"角度缺口/冗余"是不是算法假象。

为什么需要：`inspect_dataset.py` 用**方位角分箱**统计覆盖与冗余。
如果相机不是简单绕行（例如"来回摆动"或"走了一段直线"），
分箱统计会得出误导性结论。**一张轨迹图能立刻分辨**。

输出 PNG（俯视图 + 侧视图 + 方位角分布直方图），标出：
  * 相机中心（散点）与被摄物体中心
  * 最大角度空隙
  * 方位角覆盖区间
  * 相机高度分布

用法: scripts\\gpu_python.cmd scripts\\plot_camera_trajectory.py --job img3-01 --compare maskloss-v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


def pick_font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for c in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if c in names:
            return c
    return "DejaVu Sans"


plt.rcParams["font.family"] = pick_font()
plt.rcParams["axes.unicode_minus"] = False


def load_centers(job: Path):
    from gsflow.gsdata.colmap_io import read_model

    cands = sorted({p.parent for p in (job / "03_colmap").rglob("cameras.bin")})
    best = None
    for c in cands:
        try:
            r = read_model(c)
        except Exception:  # noqa: BLE001
            continue
        if best is None or len(r.images) > len(best[1].images):
            best = (c, r)
    if best is None:
        return None, None
    rec = best[1]
    cs = []
    for im in rec.images:
        r, t = im.cam_to_world()
        cs.append(t)
    return np.asarray(cs), best[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--compare", default="")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    jobs = [args.job] + [j.strip() for j in args.compare.split(",") if j.strip()]
    data = {}
    for j in jobs:
        cs, md = load_centers(ROOT / "workspace" / j)
        if cs is None:
            print(f"  [跳过] {j}: 找不到 COLMAP 模型")
            continue
        data[j] = cs
        print(f"  {j}: {len(cs)} 个相机中心（来自 {md.relative_to(ROOT / 'workspace' / j)}）")

    if not data:
        print("没有可用数据")
        return 1

    fig = plt.figure(figsize=(15, 4.6 * len(data)), dpi=120)
    n = len(data)
    for idx, (job, cs) in enumerate(data.items()):
        c = cs.mean(0)
        v = cs - c
        az = (np.degrees(np.arctan2(v[:, 2], v[:, 0])) + 360) % 360
        r = np.linalg.norm(v, axis=1)
        azs = np.sort(az)
        gaps = np.diff(np.concatenate([azs, [azs[0] + 360]]))
        gi = int(gaps.argmax())
        gap = float(gaps.max())

        # 俯视图
        ax = fig.add_subplot(n, 3, idx * 3 + 1)
        ax.scatter(cs[:, 0], cs[:, 2], s=6, c=az, cmap="twilight", vmin=0, vmax=360)
        ax.scatter([c[0]], [c[2]], marker="x", s=120, c="red", label="中心")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_title(f"{job} 俯视图（颜色=方位角）")
        ax.axis("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # 方位角分布（找缺口）
        ax2 = fig.add_subplot(n, 3, idx * 3 + 2)
        ax2.hist(az, bins=np.arange(0, 370, 10), edgecolor="k", alpha=0.7)
        a_start, a_end = azs[gi], azs[(gi + 1) % len(azs)]
        ax2.set_xlabel("方位角 (°)")
        ax2.set_ylabel("帧数")
        ax2.set_title(f"方位角分布（最大缺口 {gap:.1f}°：{a_start:.0f}°→{a_end:.0f}°）")
        ax2.axvspan(a_start, a_end, color="red", alpha=0.12)
        ax2.grid(alpha=0.3)

        # 侧视图（高度分布）
        ax3 = fig.add_subplot(n, 3, idx * 3 + 3)
        ax3.scatter(np.arange(len(cs)), cs[:, 1], s=8, alpha=0.7)
        ax3.set_xlabel("帧序号")
        ax3.set_ylabel("相机高度 Y")
        ax3.set_title(f"高度分布（跨度 {cs[:, 1].max()-cs[:, 1].min():.2f}，"
                      f"半径 {r.mean():.2f}）")
        ax3.grid(alpha=0.3)

        occ = int((np.bincount(np.floor(az / 10).astype(int)) > 0).sum())
        print(f"  {job}: 覆盖 {360-gap:.1f}°/360°（{occ} 个扇区），缺口 {gap:.1f}°，"
              f"冗余 {len(cs)/max(occ,1):.1f}×，半径 {r.mean():.2f}")

    fig.tight_layout()
    out = ROOT / "docs" / "camera_trajectory.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n已生成 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
