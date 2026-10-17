"""估算"本机 + 本数据"能达到的质量上限，避免在不可能的目标上打转。

用户反馈质量不足（点云糊、网格与纹理"灾难"）。在继续调优之前必须先回答：
**是优化不够，还是数据/硬件本身限制了上限？**

本脚本做四件事（都是可算的）：
  1. **有效前景像素**：720×720 的画面里，物体占多少像素？
     这决定训练的"信息量"，也决定 720p 是否够用。
  2. **每高斯覆盖的像素**：高斯数 / 前景像素。覆盖过大 ⇒ 容量不足。
  3. **角度冗余**：同一方位角被拍了多少次？（相机绕物体转，冗余高则有效信息少）
  4. **显存与高斯数的关系**：按本机 8GB 估算可容纳的高斯上限。

用法: scripts\\gpu_python.cmd scripts\\estimate_quality_ceiling.py --job final-v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="final-v1")
    ap.add_argument("--data-from", default="maskloss-v2", help="提供 masks 的作业")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    from PIL import Image

    from gsflow.trainers.gsplat_trainer import load_colmap_data

    job = ROOT / "workspace" / args.job
    data_dir = job / "04_train" / "data"
    metrics = json.loads((job / "04_train" / "metrics.json").read_text(encoding="utf-8"))
    n_gauss = int(metrics["gaussians"])
    n_views = int(metrics["train_views"])
    W, H = metrics["resolution"]
    print(f"作业 {args.job}：{n_views} 视图 @ {W}x{H}，高斯 {n_gauss}")

    # ---- 1) 前景像素占比（用掩码；没有就用真值图的非黑像素）----
    mask_dir = ROOT / "workspace" / args.data_from / "02_masks"
    img_dir = data_dir / "images"
    fracs = []
    if mask_dir.exists():
        for p in sorted(mask_dir.glob("*.png"))[:40]:
            with Image.open(p) as im:
                m = np.asarray(im.convert("L")) > 127
                fracs.append(float(m.mean()))
        src = f"掩码 {mask_dir.relative_to(ROOT)}"
    else:
        for p in sorted(img_dir.glob("*"))[:40]:
            with Image.open(p) as im:
                a = np.asarray(im.convert("RGB"))
                fracs.append(float((a.max(axis=2) > 12).mean()))
        src = "真值非黑像素（无掩码时）"
    fg = float(np.mean(fracs)) if fracs else 0.0
    print(f"\n① 前景占比（{src}）：均值 {fg*100:.1f}%（范围 {min(fracs)*100:.1f}%~{max(fracs)*100:.1f}%）")
    fg_px_per_view = int(fg * W * H)
    print(f"   每视图有效前景像素 ≈ {fg_px_per_view:,}")

    # ---- 2) 每高斯覆盖的前景像素 ----
    if fg_px_per_view > 0:
        per_g = fg_px_per_view / max(n_gauss, 1)
        print(f"\n② 每高斯对应前景像素 ≈ {per_g:.2f}")
        print("   （社区经验：该值 > 1 说明高斯太少；< 0.3 说明足够密）")

    # ---- 3) 角度冗余 ----
    d = load_colmap_data(data_dir, downscale=1, white_background=False, flip_camera=False)
    centers = d["c2ws"][:, :3, 3]
    c = centers.mean(0)
    v = centers - c
    az = (np.degrees(np.arctan2(v[:, 2], v[:, 0])) + 360) % 360
    # 按 10° 分箱统计"每个方位角被拍了几次"
    bins = np.floor(az / 10).astype(int)
    counts = np.bincount(bins)
    occupied = counts[counts > 0]
    print(f"\n③ 角度冗余：{n_views} 张覆盖 {len(occupied)} 个 10° 方位扇区")
    print(f"   每扇区平均 {occupied.mean():.1f} 张（最多 {occupied.max()} 张）")
    print(f"   ⇒ 冗余比 {n_views / max(len(occupied), 1):.1f}× "
          f"（同一角度被重复拍摄的次数；越高说明有效信息越少）")
    gap = np.diff(np.concatenate([np.sort(az), [np.sort(az)[0] + 360]])).max()
    print(f"   最大角度空隙 {gap:.1f}° ⇒ 覆盖 {360-gap:.1f}°/360°")

    # ---- 4) 显存上限估算 ----
    # 每个高斯参数：xyz3 + scale3 + rot4 + opacity1 + sh(deg3)=(3+45)=48 float
    # 训练时还要 Adam 两个动量 + 梯度 ⇒ 约 3 倍参数量；再加光栅化中间量。
    floats_per_g = 3 + 3 + 4 + 1 + 48          # = 59
    bytes_per_g = floats_per_g * 4 * 3          # 参数+两动量 ≈ 3x
    cap_8gb = int(8 * 1024**3 * 0.55 / bytes_per_g)   # 预留 55% 给光栅化/激活
    print(f"\n④ 显存估算：每高斯约 {bytes_per_g} 字节（含 Adam 动量）")
    print(f"   8GB 显存（预留 55%）≈ 可容纳 {cap_8gb:,} 个高斯")
    print(f"   当前 {n_gauss:,} ⇒ 余量 {cap_8gb/max(n_gauss,1):.1f}×")

    print("\n=== 结论 ===")
    verdict = []
    if fg_px_per_view < 150_000:
        verdict.append(f"前景像素仅 {fg_px_per_view:,}（720p 里物体占比 {fg*100:.0f}%）"
                       f" ⇒ **分辨率不是瓶颈**，但物体细节的像素信息有限")
    if n_gauss < fg_px_per_view * 0.3:
        verdict.append(f"高斯 {n_gauss:,} 远少于前景像素 {fg_px_per_view:,}"
                       f" ⇒ **容量不足是主因之一**")
    if n_views / max(len(occupied), 1) > 2.0:
        verdict.append(f"角度冗余 {n_views/max(len(occupied),1):.1f}×"
                       f" ⇒ 大量视角在拍同一面，**有效信息远少于张数**")
    if gap > 30:
        verdict.append(f"缺 {gap:.0f}° 视角 ⇒ 那部分是**外推**，必然糊/破碎")
    if not verdict:
        verdict.append("未发现明显的数据/容量上限")
    for s in verdict:
        print(f"  * {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
