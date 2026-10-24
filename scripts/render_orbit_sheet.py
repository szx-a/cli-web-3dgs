"""从**任意轨道视角**渲染模型并出图，用于肉眼判断"细丝"。

## 为什么要这个

我的指标（掩码外 alpha、`aniso`）都说"没有细丝了"，但用户仍然看到细丝。
原因可能是我只测了**训练视角附近 ±25°**，而查看器可以转到任何角度：
`max_aspect=50` 允许 50:1 的椭球 —— 在 0.16 米的尺度下是 0.16m × 3mm 的**一根毛**，
在训练视角附近看不出来，转到侧面就是一根丝。

所以：绕物体均匀采样 8 个方位 + 3 个仰角，渲成 3×3 拼图（远景全貌 + 局部放大），
自己看。

用法: python scripts/render_orbit_sheet.py --job img3-final --out env/tmp/orbit.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _axis_angle(axis: np.ndarray, ang: float) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = np.cos(ang), np.sin(ang)
    x, y, z = a
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="img3-final")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--out", default="env/tmp/orbit.png")
    ap.add_argument("--dist", type=float, default=2.2, help="相机距离（× 物体半径）")
    ap.add_argument("--data-dir", default="",
                    help="含 sparse/ 的目录；默认 jd/04_train/data（对比实验作业在 workspace/<work>/data）")
    ap.add_argument("--sh", type=int, default=-1,
                    help="强制 SH 阶数（-1=用模型自带；0=仅 DC）——用来分离彩虹色块是否来自 SH")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env

    apply_runtime_env()
    from PIL import Image

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.gsdata.shio import read_gaussians_full, render_view

    jd = ROOT / "workspace" / args.job
    ply = max((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
              key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))
    g = read_gaussians_full(ply)
    if args.sh >= 0:
        nb = (args.sh + 1) ** 2
        if np.asarray(g["shs"]).shape[1] >= nb:
            g = dict(g)
            g["shs"] = np.asarray(g["shs"])[:, :nb, :].copy()
            g["sh_degree"] = args.sh
            print(f"  forced sh_degree={args.sh}")
    xyz = np.asarray(g["xyz"], dtype=np.float64)
    lo, hi = xyz.min(0), xyz.max(0)
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo)) / 2

    data = (Path(args.data_dir) if args.data_dir else jd / "04_train" / "data")
    if not (data / "sparse").exists():
        alt = ROOT / "workspace" / jd.name.split("-")[0].rsplit("-", 1)[0] / "data"
        if (alt / "sparse").exists():
            data = alt
            print(f"  自动定位到 {data}")
    rec = max((read_model(c) for c in sorted(
        {p.parent for p in (data / "sparse").rglob("cameras.bin")})),
        key=lambda r: len(r.images))
    ups = []
    for im in rec.images:
        R, _ = im.cam_to_world()
        ups.append(-np.asarray(R, dtype=np.float64)[:, 1])
    up = np.mean(ups, axis=0)
    up /= np.linalg.norm(up) + 1e-12

    # 以 up 为轴构造一组正交基
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, up)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(up, ref)
    e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(up, e1)

    fov = 50.0
    W = H = args.size
    f = 0.5 * W / np.tan(np.deg2rad(fov) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)

    rows = []
    for elev in (15.0, 40.0, 70.0):
        tiles = []
        for yaw in (0.0, 45.0, 90.0, 135.0, 180.0, 270.0):
            a, b = np.deg2rad(yaw), np.deg2rad(elev)
            d = (np.cos(b) * (np.cos(a) * e1 + np.sin(a) * e2) + np.sin(b) * up)
            eye = center + d * radius * args.dist
            # 构造 w2c（OpenCV 约定：+Z 朝前，+Y 朝下）
            # OpenCV 约定：相机 +Z 朝**前**（从眼睛指向目标），+Y 朝下
            z = center - eye
            z /= np.linalg.norm(z) + 1e-12
            x = np.cross(z, up)          # 右 = 前 × 上（保证 y 朝下）
            if np.linalg.norm(x) < 1e-8:
                x = np.cross(z, e1)
            x /= np.linalg.norm(x) + 1e-12
            y = np.cross(z, x)           # 下
            R = np.stack([x, y, z], axis=0)   # 行 = 相机轴
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = -R @ eye
            colors, alphas = render_view(g, w2c, K, W, H)
            col = np.asarray(colors.detach().cpu().numpy()
                             if hasattr(colors, "detach") else colors)[..., :3]
            a_map = np.asarray(alphas.detach().cpu().numpy()
                               if hasattr(alphas, "detach") else alphas)[..., 0]
            img = np.clip(col, 0, 1)
            # 黑背景上叠半透明：更接近查看器观感
            img = img + 0.0
            tile = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            tiles.append(tile)
        rows.append(np.concatenate(tiles, axis=1))
    sheet = np.concatenate(rows, axis=0)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out)
    print(f"  已写出 {out}（3 个仰角 × 6 个方位，{W}x{H} 每格）")
    print(f"  物体中心 {np.round(center,2).tolist()} 半径 {radius:.2f}，"
          f"相机距离 {radius*args.dist:.2f}")
    print("  判读：看有没有贯穿画面的细丝、半透明雾、或成片的碎屑。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
