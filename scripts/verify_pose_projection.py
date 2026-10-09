"""决定性诊断：把 COLMAP 稀疏点**投影**到真值图像上，看点是否落在纹理上。

为什么之前那个"用稀疏点当高斯渲染再算 PSNR"的方法不成立：
只有 1846 个点、每个渲染成一枚小圆点，覆盖不到整张图，
所以**无论位姿对错 PSNR 都很低**（实测两种约定都是 ~10 dB），没有判别力。

正确做法：把点投影成像素坐标，然后检查这些像素处的**图像梯度/边缘**是否明显。
  * 位姿正确 -> 点落在棋盘格角点等纹理丰富处 -> 该处梯度大
  * 位姿错误 -> 点会落到平坦背景（深灰）上 -> 该处梯度接近 0

用法：scripts\\gpu_python.cmd scripts\\verify_pose_projection.py --data <去畸变数据目录>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow import paths  # noqa: E402

paths.apply_runtime_env()

from gsflow.trainers.gsplat_trainer import load_colmap_data  # noqa: E402


def image_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--views", type=int, default=6, help="抽样多少个视图做统计")
    args = ap.parse_args()

    from PIL import Image

    data_dir = Path(args.data)
    img_dir = data_dir / "images"
    all_imgs = sorted(img_dir.iterdir())

    print(f"{'约定':<12}{'平均梯度@点':>14}{'图像平均梯度':>14}{'比值':>10}{'点在图内占比':>14}")
    print("-" * 68)
    summary = []
    for flip in (False, True):
        d = load_colmap_data(data_dir, flip_camera=flip)
        names = d["names"]
        c2ws = d["c2ws"]
        ks = d["ks"]
        pts = d["points"]
        hgt = d["heights"]
        wid = d["widths"]

        ratios, covers, grads_at, grads_all = [], [], [], []
        step = max(1, len(names) // args.views)
        for i in range(0, len(names), step):
            p = img_dir / names[i]
            if not p.exists():
                cands = list(img_dir.glob(Path(names[i]).stem + ".*"))
                if not cands:
                    continue
                p = cands[0]
            with Image.open(p) as im:
                im = im.convert("L")
                if max(im.size) > 400:
                    s = 400 / max(im.size)
                    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
                gray = np.asarray(im, dtype=np.float32)
            H, W = gray.shape
            grad = image_gradient_magnitude(gray)

            # 世界点 -> 相机 -> 像素
            c2w = c2ws[i].astype(np.float64)
            w2c = np.linalg.inv(c2w)
            K = ks[i].astype(np.float64)
            # 注意：K 是按原图分辨率给的，需要随缩放同步
            sx, sy = W / wid[i], H / hgt[i]
            Ks = K.copy()
            Ks[0, 0] *= sx
            Ks[1, 1] *= sy
            Ks[0, 2] *= sx
            Ks[1, 2] *= sy

            pc = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
            front = pc[:, 2] > 1e-6
            proj = (Ks @ pc[front].T).T
            uv = proj[:, :2] / proj[:, 2:3]
            u = np.round(uv[:, 0]).astype(int)
            v = np.round(uv[:, 1]).astype(int)
            inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            covers.append(float(inside.mean()))
            if inside.sum() < 10:
                continue
            g_at = float(grad[v[inside], u[inside]].mean())
            grads_at.append(g_at)
            grads_all.append(float(grad.mean()))

        if not grads_at:
            print(f"{'不翻转' if not flip else '翻 y':<12} 无有效视图")
            continue
        ga = float(np.mean(grads_at))
        gm = float(np.mean(grads_all))
        cov = float(np.mean(covers))
        tag = "不翻转" if not flip else "翻 y"
        print(f"{tag:<12}{ga:>14.2f}{gm:>14.2f}{ga/max(gm,1e-6):>10.2f}{cov*100:>13.1f}%")
        summary.append((tag, ga / max(gm, 1e-6), cov))

    if summary:
        best = max(summary, key=lambda s: s[1])
        print(f"\n结论：'{best[0]}' 的点落在纹理上（梯度比 {best[1]:.2f}x，图内占比 {best[2]*100:.1f}%）")
        print("比值 > 1 说明点确实落在有纹理的地方；两种约定都接近 1 则说明测试仍无判别力。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
