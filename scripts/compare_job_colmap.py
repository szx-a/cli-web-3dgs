"""对比两个作业的 COLMAP/去畸变数据，用于判断"分辨率差异"是否真实存在。

背景：两个作业在**统一 8 视角**下 PSNR 差 0.67 dB（20.51 vs 19.84），
日志显示它们最终训练分辨率分别是 720x720 与 722x722。
若 COLMAP 声明的图像尺寸/内参本就不同，那 0.67 dB 就不该归因于"分辨率选择"。

用法: python scripts/compare_job_colmap.py --jobs maskloss-v2,resauto-v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    args = ap.parse_args()

    from gsflow.trainers.gsplat_trainer import load_colmap_data

    for job in [j.strip() for j in args.jobs.split(",") if j.strip()]:
        d = ROOT / "workspace" / job / "04_train" / "data"
        if not d.exists():
            print(f"{job}: 缺少 {d}")
            continue
        data = load_colmap_data(d, downscale=1, white_background=False, flip_camera=False)
        K = np.asarray(data["ks"][0])
        ws = np.asarray(data["widths"])
        hs = np.asarray(data["heights"])
        print(f"{job}:")
        print(f"  图像 {len(data['names'])} 张   声明尺寸 {ws[0]}x{hs[0]}"
              f"（所有图一致={bool((ws == ws[0]).all() and (hs == hs[0]).all())}）")
        print(f"  内参 fx={K[0,0]:.4f} fy={K[1,1]:.4f} cx={K[0,2]:.4f} cy={K[1,2]:.4f}")
        print(f"  稀疏点 {len(data['points'])}")
        # 去畸变目录里图像的真实像素尺寸
        imgs = sorted((d / "images").glob("*"))
        if imgs:
            from PIL import Image
            with Image.open(imgs[0]) as im:
                print(f"  实际图像 {imgs[0].name} 像素 {im.size[0]}x{im.size[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
