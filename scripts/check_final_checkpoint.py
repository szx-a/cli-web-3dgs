"""检查"最终检查点"是否被不透明度重置污染（启用重置为默认后必须常做）。

## 背景
`opacity_reset_interval` 现在默认 **3000**。若重置发生在保存之后的那一刻，
最终 PLY 会是一堆 op≈0.01 的高斯 ⇒ 渲染近乎全黑（历史事故：90.6% 的高斯被钉在 0.01，
逐视图 PSNR 仅 ~10 dB）。trainer 有两条防线（重置在保存之后、最后一步跳过重置），
这个脚本用来**实证**它们有效。

## 判据（任一不满足即 WARN）
1. `op` 中位数 ≥ 0.02（被重置过的模型会掉到 ~0.01）；
2. `op` 最大值 ≥ 0.3（正常训练应有相当一部分较实的高斯）；
3. 从一个训练相机渲染，掩码内 PSNR ≥ 22 dB（污染时会掉到 ~10 dB）。

用法：python scripts/check_final_checkpoint.py --job img5-absgs4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow.gsdata.shio import read_gaussians_full  # noqa: E402
from gsflow.paths import apply_runtime_env  # noqa: E402

apply_runtime_env()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    args = ap.parse_args()

    jd = ROOT / "workspace" / args.job
    plys = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"))
    if not plys:
        print(f"  {args.job}: 没有检查点")
        return 1
    ply = plys[-1]
    g = read_gaussians_full(ply)
    op = np.asarray(g["opacity"], np.float64).reshape(-1)
    if op.max() > 1.5:
        op = 1.0 / (1.0 + np.exp(-op))
    p50, pmax = float(np.median(op)), float(op.max())
    print(f"  {args.job}  {ply.parent.name}  n={len(op):,}")
    print(f"    op p50={p50:.4f}  max={pmax:.4f}  >0.5 \u5360\u6bd4={float((op>0.5).mean())*100:.1f}%")

    ok = True
    if p50 < 0.02:
        print("    ⚠ op 中位 < 0.02：**疑似最终检查点被重置污染**")
        ok = False
    if pmax < 0.3:
        print("    ⚠ op 最大值 < 0.3：模型几乎没有实心高斯")
        ok = False

    # \u6e32\u67d3\u4e00\u5e27\u505a\u4fdd\u5e95\uff08\u9700 CUDA\uff09
    try:
        import torch  # noqa: F401
        from PIL import Image

        from gsflow.gsdata.colmap_io import read_model
        from gsflow.gsdata.shio import render_view

        sp = jd / "04_train" / "data" / "sparse"
        rec = read_model(sp / "0" if (sp / "0" / "cameras.bin").exists() else sp)
        im = sorted(rec.images, key=lambda x: x.name)[0]
        cam = rec.cameras[im.camera_id]
        R, c = im.cam_to_world()
        R = np.asarray(R, float)
        c = np.asarray(c, float)
        w2c = np.eye(4)
        w2c[:3, :3] = R.T
        w2c[:3, 3] = -R.T @ c
        K = np.asarray(cam.intrinsics(), float)
        col, _ = render_view(g, w2c, K, cam.width, cam.height)
        img = np.clip(np.asarray(col.detach().cpu() if hasattr(col, "detach") else col)[..., :3], 0, 1)
        with Image.open(jd / "04_train" / "data" / "images" / im.name) as gi:
            gt = np.asarray(gi.convert("RGB"), dtype=np.float32) / 255.0
        if gt.shape[:2] == img.shape[:2]:
            # ⚠ 掩码口径必须与探针一致：优先 `masks/*.png` 前景掩码（约 35 dB）；
            # 退回 "gt 非黑" 时天然只有约 17–18 dB（含轮廓过渡带），阈值也要相应调低。
            mask_p = jd / "04_train" / "data" / "masks" / (Path(im.name).stem + ".png")
            m = None
            if mask_p.exists():
                mm = np.asarray(Image.open(mask_p).convert("L")) > 127
                if mm.shape == gt.shape[:2]:
                    m = mm
            if m is None:
                m = gt.max(axis=2) > 6 / 255.0
            thr = 22.0 if mask_p.exists() else 14.0
            kind = "前景掩码" if mask_p.exists() else "gt 非黑"
            mse = float(((img - gt) ** 2)[m].mean()) if m.any() else float("nan")
            psnr = 99.0 if mse <= 1e-12 else 10 * np.log10(1 / mse)
            print(f"    训练相机渲染 PSNR = {psnr:.2f} dB（{kind}，阈值 {thr:.0f} dB）")
            if psnr < thr:
                print(f"    ⚠ PSNR < {thr:.0f} dB：**该检查点很可能不可用**（历史上重置污染时约 10 dB）")
                ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"    \uff08\u6e32\u67d3\u68c0\u67e5\u8df3\u8fc7\uff1a{type(exc).__name__}: {exc}\uff09")

    print("    \u7ed3\u8bba:", "\u901a\u8fc7" if ok else "**\u9700\u68c0\u67e5**")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
