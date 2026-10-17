"""判定"亮背景"到底是**高斯**还是**渲染器的背景色**。

## 为什么要查

我先后推断"背景被训练进去了"，并用 `mask_loss_mode=inside` 做了受控实验：

    composite（默认）: 亮权重 61.2%，外层亮权重 91.0%
    inside          : 亮权重 59.6%，外层亮权重 87.8%   ← 几乎没变

⇒ **改损失口径不能去掉那片亮区**，说明我原来的归因不完整。

关键问题：`preview_*.png` 里那片浅灰背景，到底是

  a) 高斯渲出来的（alpha ≈ 1），还是
  b) gsplat 渲染器的**背景色**（alpha ≈ 0，只是被合成成了浅色）？

`shio.render_view` 返回 `(colors, alphas)`，直接看 alpha 就能分辨：
背景处 alpha≈0 ⇒ 那里**没有几何**，看到的只是合成色。

用法: python scripts/diag_alpha_background.py --job img3-final
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
    ap.add_argument("--job", default="img3-final")
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--data-dir", default="",
                    help="相机位姿/掩码所在的数据目录（默认在作业目录下找）")
    args = ap.parse_args()

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.gsdata.shio import read_gaussians_full

    # ⚠️ 先确认 gsplat 的 CUDA 内核**已就绪**，否则 `render_view` 会触发一次
    # JIT 编译（本机实测 80~140 秒），而本机在 GBK 代码页下探测编译器版本会直接崩：
    #     UnicodeDecodeError: 'cp1' codec can't decode ...
    #
    # 判据要与项目里的 `export._splat_kernel_ready()` **完全一致**：
    #   1) `paths.apply_runtime_env()` 已把预编译内核注册进 `sys.modules`
    #      （它会注册 `gsplat_cuda` 与 `gsplat.csrc`，让 gsplat 跳过编译分支）；
    #   2) `gsplat.cuda._backend._C` 不为 None。
    # 我第一版写成 `hasattr(gsplat.cuda, "_C")` —— 那是错的属性路径，
    # 于是永远"未就绪"，把一条本来能跑的检查挡掉了。
    from gsflow.paths import apply_runtime_env

    apply_runtime_env()
    import sys as _sys

    if _sys.modules.get("gsplat.csrc") is None:
        print("gsplat CUDA 内核未注册（apply_runtime_env 未生效），本检查跳过。")
        return 0
    try:
        from gsplat.cuda._backend import _C as _kernel

        if _kernel is None:
            print("gsplat CUDA 内核未就绪（会触发 JIT 编译），本检查跳过。")
            print("  提示：先跑一次训练/导出（内核会构建并缓存），再执行本脚本。")
            return 0
    except Exception as exc:  # noqa: BLE001
        print(f"gsplat 不可用（{type(exc).__name__}），本检查跳过。")
        return 0

    from gsflow.gsdata.shio import render_view

    jd = ROOT / "workspace" / args.job
    plys = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"))
    if not plys:
        print("没有训练产物")
        return 1
    g = read_gaussians_full(plys[-1])

    # 相机位姿可能在 03_colmap/sparse，也可能随训练数据复制到 04_train/data/sparse
    roots = [jd / "03_colmap" / "sparse", jd / "04_train" / "data" / "sparse"]
    if args.data_dir:
        roots.insert(0, Path(args.data_dir) / "sparse")
    cands = []
    for r in roots:
        if r.exists():
            cands.extend(sorted({p.parent for p in r.rglob("cameras.bin")}))
    cands = sorted(set(cands))
    best = None
    for c in cands:
        rec = read_model(c)
        if best is None or len(rec.images) > len(best.images):
            best = rec
    if best is None:
        print("没有相机位姿")
        return 1

    print(f"=== alpha 判定：{args.job} ===")
    n = min(args.views, len(best.images))
    step = max(1, len(best.images) // n)
    picks = list(range(0, len(best.images), step))[:n]
    im0 = best.images[picks[0]]
    cam0 = best.cameras[im0.camera_id]          # Camera 在 Reconstruction.cameras 里
    W, H = cam0.width, cam0.height
    # 用较小分辨率，够判断 alpha 分布
    W, H = min(W, 640), min(H, int(640 * cam0.height / max(cam0.width, 1)))

    for k in picks:
        im = best.images[k]
        cam = best.cameras[im.camera_id]
        # ⚠️ `cam_to_world()` 返回 (R 3x3, 相机中心 3)，**不是** 4x4 矩阵。
        # 我第一次当 4x4 用，gsplat 直接 assert 失败：
        #     viewmats.shape == (C, 4, 4)，实际 (1, 3, 3)
        R, center = im.cam_to_world()
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = np.asarray(R).T
        w2c[:3, 3] = -np.asarray(R).T @ np.asarray(center)
        K = np.asarray(cam.intrinsics()).copy()
        sx = W / cam.width
        sy = H / cam.height
        K[0, :] *= sx
        K[1, :] *= sy
        colors, alphas = render_view(g, w2c, K, W, H)
        a = np.asarray(alphas.detach().cpu().numpy()
                       if hasattr(alphas, "detach") else alphas)[..., 0]
        c = np.asarray(colors.detach().cpu().numpy()
                       if hasattr(colors, "detach") else colors)
        if c.ndim == 3 and c.shape[-1] >= 3:
            c = c[..., :3]
        lum = c.mean(axis=2)
        empty = a < 0.05
        print(f"\n  视角 {im.name}（{W}x{H}）")
        print(f"    alpha: 中位 {np.median(a):.3f}  >0.5 占比 {(a>0.5).mean()*100:.1f}%  "
              f"<0.05 占比 {empty.mean()*100:.1f}%")
        if empty.any():
            print(f"    **空区**（alpha<0.05）平均亮度 {lum[empty].mean():.3f}"
                  f"  —— 若也很亮，说明亮背景来自渲染器而非高斯")
        solid = a > 0.5
        if solid.any():
            print(f"    实心区平均亮度 {lum[solid].mean():.3f}")
        bright = lum > 0.8
        if bright.any():
            print(f"    亮度>0.8 的像素: {bright.mean()*100:.1f}%，"
                  f"其中 alpha<0.05 的占 {(bright & empty).sum()/max(bright.sum(),1)*100:.1f}%")

        # ---- 掩码内/外分开统计：直接看"背景有没有被抑制" ----
        from PIL import Image as _Im

        mdata = (Path(args.data_dir) / "masks") if args.data_dir \
            else (jd / "04_train" / "data" / "masks")
        mp = None
        for cand in (mdata / (im.name + ".png"),
                     mdata / (Path(im.name).stem + ".png")):
            if cand.exists():
                mp = cand
                break
        if mp is not None:
            with _Im.open(mp) as mm:
                msk = np.asarray(mm.convert("L").resize((W, H), _Im.NEAREST)) > 127
            if msk.any() and (~msk).any():
                print(f"    掩码内: 亮度 {lum[msk].mean():.3f}  alpha {a[msk].mean():.3f}")
                print(f"    掩码外: 亮度 {lum[~msk].mean():.3f}  alpha {a[~msk].mean():.3f}"
                      f"   ← 若掩码生效，这里应接近 **黑**（合成损失把掩码外 GT 涂黑）")

    print("\n  判读：若'亮度>0.8 的像素'里大部分 alpha<0.05 ⇒ 亮背景**不是高斯**，"
          "\n        而是渲染器/合成时的背景色 ⇒ 观感问题的修法在查看器（背景/色调），"
          "\n        而不是训练损失。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
