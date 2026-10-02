"""单阶段验证：用合成图直接跑 frames → mask（不需要真实数据集）。

用途：验证 mask 阶段的完整契约（掩码 PNG + 黑背景清理图 + overlay + manifest），
并给出可量化结论（前景占比、与 GT 的 IoU）。

用法：
    env\\venv\\Scripts\\python.exe scripts\\verify_mask_stage.py
    env\\venv\\Scripts\\python.exe scripts\\verify_mask_stage.py --backend classical --dilate 6 --feather 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _make_sample(shape: str, size: int, seed: int):
    """内联实现，避免脚本间互相导入带来的路径问题。"""
    from PIL import Image, ImageDraw, ImageFilter

    rng = np.random.default_rng(seed)
    bg = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        bg[y, :, :] = 40 + (y % 40) * 3
    bg = np.clip(bg + rng.normal(0, 12, bg.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)
    gt = Image.new("L", (size, size), 0)
    gtd = ImageDraw.Draw(gt)
    cx = cy = size // 2
    r = size // 4
    if shape == "rect":
        box = [cx - r, cy - int(r * 1.3), cx + r, cy + int(r * 1.3)]
        draw.rectangle(box, fill=(60, 120, 220), outline=(255, 255, 255), width=4)
        gtd.rectangle(box, fill=255)
    else:
        box = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(box, fill=(220, 90, 40), outline=(255, 255, 255), width=4)
        gtd.ellipse(box, fill=255)
        for i in range(8):
            off = -r + i * (2 * r / 8) + 6
            draw.line([cx - r + 8, cy + off, cx + r - 8, cy + off], fill=(150, 40, 20), width=3)
    return img.filter(ImageFilter.GaussianBlur(0.8)), np.asarray(gt, dtype=np.uint8) > 127


def main() -> int:
    ap = argparse.ArgumentParser(description="验证 mask 阶段的完整契约")
    ap.add_argument("--backend", default="classical")
    ap.add_argument("--shape", default="circle", choices=["circle", "rect"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--dilate", type=int, default=4)
    ap.add_argument("--feather", type=int, default=2)
    ap.add_argument("--root", default=str(ROOT / "env" / "verify-mask"))
    args = ap.parse_args()

    from gsflow.config import load as load_settings
    from gsflow.events import EventBus
    from gsflow.pipeline.context import StageContext
    from gsflow.stages import MaskStage, register_all  # noqa: F401

    root = Path(args.root)
    job_dir = root / "job"
    raw = job_dir / "01_images_raw"
    raw.mkdir(parents=True, exist_ok=True)
    for p in raw.iterdir():
        p.unlink()

    gt_masks = {}
    for i in range(args.count):
        img, gt = _make_sample(args.shape, args.size, seed=10 + i)
        name = f"{i:05d}.jpg"
        img.save(raw / name, quality=95)
        gt_masks[name] = gt
    print(f"已生成合成图 {args.count} 张 -> {raw}（GT 前景占比 {np.mean([g.mean() for g in gt_masks.values()]):.1%}）")

    settings = load_settings()
    settings.set("project.workspace_dir", str(root / "workspace"))
    settings.set("mask.backend", args.backend)
    settings.set("mask.dilate_px", args.dilate)
    settings.set("mask.feather_px", args.feather)
    ctx = StageContext(job_id="verify", job_dir=job_dir, settings=settings, events=EventBus(),
                       params={"mask": {"backend": args.backend, "dilate_px": args.dilate,
                                        "feather_px": args.feather}})
    ctx.bind_stage("mask")

    result = MaskStage().execute(ctx)
    print(f"\n阶段状态：{result.status}  {result.message}")
    print(f"耗时：{result.seconds:.2f}s")

    mask_dir = job_dir / "02_masks"
    clean_dir = job_dir / "02_images_clean"
    manifest = json.loads((mask_dir / "manifest.json").read_text(encoding="utf-8"))

    # 契约检查
    problems = []
    if not (mask_dir / "manifest.json").exists():
        problems.append("缺少 manifest.json")
    if not (mask_dir / "_overlay").exists():
        problems.append("缺少 _overlay 效果图目录")
    if len(manifest["items"]) != args.count:
        problems.append(f"manifest 条数 {len(manifest['items'])} != 输入 {args.count}")
    for item in manifest["items"]:
        if not (mask_dir / item["mask"]).exists():
            problems.append(f"缺掩码 {item['mask']}")
        if not (clean_dir / item["clean"]).exists():
            problems.append(f"缺清理图 {item['clean']}")

    # 质量：与 GT 的 IoU
    from PIL import Image

    ious = []
    black_ok = 0
    for name, gt in gt_masks.items():
        pred_p = mask_dir / (Path(name).stem + ".png")
        with Image.open(pred_p) as m:
            pred = np.asarray(m.convert("L"), dtype=np.uint8) > 127
        inter = float(np.logical_and(pred, gt).sum())
        union = float(np.logical_or(pred, gt).sum()) or 1.0
        ious.append(inter / union)

        with Image.open(clean_dir / (Path(name).stem + ".jpg")) as c:
            arr = np.asarray(c.convert("RGB"))
        # 背景应接近全黑：检查 GT 之外的像素亮度
        outside = arr[~gt]
        if outside.size and outside.mean() < 12:
            black_ok += 1

    print(f"\n=== 契约检查 ===")
    print("全部通过" if not problems else "\n".join(f"  ✗ {p}" for p in problems))
    print(f"=== 质量 ===")
    print(f"  与 GT 的平均 IoU：{float(np.mean(ious)):.4f}（每张 {[round(x,3) for x in ious]}）")
    print(f"  背景确认为黑（亮度<12）的帧：{black_ok}/{args.count}")
    print(f"  平均前景占比：{manifest['summary']['mean_foreground_ratio']:.1%}")
    print(f"\n产物：{job_dir}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
