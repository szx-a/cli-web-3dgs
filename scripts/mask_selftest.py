"""分割后端自检：用合成图（已知前景形状）量化验证掩码质量。

做法：生成“彩色主体 + 杂乱背景”的合成图，主体形状已知（圆/方块），
于是可以用 IoU 客观评价各分割后端的效果，而不是靠肉眼看效果图。

用法：
    env\\venv\\Scripts\\python.exe scripts\\mask_selftest.py
    env\\venv\\Scripts\\python.exe scripts\\mask_selftest.py --backend rembg --shape circle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 分割后端（ultralytics/rembg）的配置与权重目录必须在导入前指到工作区
try:
    from gsflow import paths as _paths

    _paths.apply_runtime_env()
except Exception:
    pass


def make_sample(shape: str, size: int = 512, seed: int = 3) -> tuple[Image.Image, np.ndarray]:
    """返回 (合成图, 前景 GT 掩码)。背景刻意做成高对比纹理，考验分割鲁棒性。"""
    rng = np.random.default_rng(seed)
    bg = np.zeros((size, size, 3), dtype=np.uint8)
    # 条纹 + 噪点背景
    for y in range(size):
        bg[y, :, :] = 40 + (y % 40) * 3
    bg = np.clip(bg + rng.normal(0, 12, bg.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)

    gt = Image.new("L", (size, size), 0)
    gtd = ImageDraw.Draw(gt)
    cx, cy, r = size // 2, size // 2, size // 4

    if shape == "circle":
        box = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(box, fill=(220, 90, 40), outline=(255, 255, 255), width=4)
        gtd.ellipse(box, fill=255)
        # 内部纹理，避免纯色导致分割模型“看不出物体”
        for i in range(8):
            off = -r + i * (2 * r / 8) + 6
            draw.line([cx - r + 8, cy + off, cx + r - 8, cy + off], fill=(150, 40, 20), width=3)
    else:
        box = [cx - r, cy - int(r * 1.3), cx + r, cy + int(r * 1.3)]
        draw.rectangle(box, fill=(60, 120, 220), outline=(255, 255, 255), width=4)
        gtd.rectangle(box, fill=255)
        for i in range(6):
            x = cx - r + 12 + i * (2 * r - 24) / 6
            draw.line([x, cy - int(r * 1.3) + 10, x, cy + int(r * 1.3) - 10], fill=(30, 70, 160), width=4)

    img = img.filter(ImageFilter.GaussianBlur(0.8))
    return img, np.asarray(gt, dtype=np.uint8) > 127


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum()) or 1.0
    return inter / union


def main() -> int:
    ap = argparse.ArgumentParser(description="分割后端量化自检（IoU vs 已知 GT）")
    ap.add_argument("--backend", default="", help="只测指定后端；默认测全部可用后端")
    ap.add_argument("--shape", default="circle", choices=["circle", "rect"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default=str(ROOT / "env" / "mask-selftest"))
    args = ap.parse_args()

    from gsflow.errors import GsflowError
    from gsflow.imageops import dilate, largest_component, load_rgb, save_overlay
    from gsflow.segment import SegOptions, create

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    img, gt = make_sample(args.shape, args.size)
    img_path = out_dir / f"sample_{args.shape}.png"
    img.save(img_path)
    save_overlay(np.asarray(img), gt.astype(np.float32), out_dir / f"gt_{args.shape}.jpg")
    print(f"合成样本：{img_path}  前景占比 {gt.mean():.1%}")

    backends = [args.backend] if args.backend else ["rembg", "fastsam", "sam2", "sam3"]
    any_ok = False
    for name in backends:
        try:
            seg = create(name, SegOptions(prompt="object", keep_largest=True, imgsz=640))
            masks = seg.segment([img_path])
        except GsflowError as exc:
            print(f"[{name:<8}] 跳过：{str(exc)[:120]}")
            continue
        except Exception as exc:  # 权重下载失败、显存不足等
            print(f"[{name:<8}] 失败：{type(exc).__name__}: {str(exc)[:120]}")
            continue

        mask = masks.get(img_path.name)
        if mask is None:
            print(f"[{name:<8}] 没有返回掩码")
            continue
        if mask.shape != gt.shape:
            from PIL import Image as PILImage

            mask = np.asarray(
                PILImage.fromarray((mask * 255).astype(np.uint8)).resize((gt.shape[1], gt.shape[0]), PILImage.NEAREST)
            ) > 127
        score = iou(mask, gt)
        area = mask.mean()
        any_ok = True
        rgb = load_rgb(img_path)
        save_overlay(rgb, mask.astype(np.float32), out_dir / f"pred_{name}_{args.shape}.jpg")
        verdict = "良好" if score > 0.85 else ("可用" if score > 0.7 else "偏差较大")
        print(f"[{name:<8}] IoU={score:.3f}  前景占比={area:.1%}  → {verdict}"
              f"   （效果图 {out_dir / f'pred_{name}_{args.shape}.jpg'}）")

    if not any_ok:
        print("\n没有任何后端可用。先安装：powershell -ExecutionPolicy Bypass -File scripts\\bootstrap.ps1 -Segment")
        return 1
    print(f"\n产物目录：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
