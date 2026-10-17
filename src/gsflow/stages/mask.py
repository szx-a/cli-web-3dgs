"""阶段 2：AI 前景分割与背景清理（本项目的核心预处理）。

输入：01_images_raw/
产物：
  02_masks/*.png        二值掩码（可喂 COLMAP 的 mask_path 忽略动态物体）
  02_images_clean/*.jpg 清理后图像（背景纯黑，或按 mode 做纹理补全）
  02_masks/_overlay/    GUI/人工抽检用的效果图
  02_masks/manifest.json 每张图的统计与后端信息
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .. import imageops
from ..errors import StageError
from ..pipeline.base import Artifact, Stage, StageResult
from ..pipeline.context import StageContext
from ..segment import SegOptions, create

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class MaskStage(Stage):
    name = "mask"
    title = "② AI 前景分割 / 背景清理"
    description = "用可插拔分割后端生成掩码，输出纯黑背景（或补全）的清理图，供 COLMAP 使用"
    requires = ("frames",)
    inputs = (Artifact("01_images_raw", "原始图片"),)
    outputs = (
        Artifact("02_masks", "二值掩码"),
        Artifact("02_images_clean", "清理后的图片"),
    )

    def run(self, ctx: StageContext) -> StageResult:
        src_dir = ctx.job_dir / "01_images_raw"
        images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
        if not images:
            raise StageError(self.name, f"{src_dir} 中没有图片", "先跑 frames 阶段")

        mask_dir = ctx.stage_dir("02_masks")
        clean_dir = ctx.stage_dir("02_images_clean")
        overlay_dir = mask_dir / "_overlay"
        overlay_dir.mkdir(exist_ok=True)

        backend_name = str(ctx.cfg("backend", "fastsam"))
        mode = str(ctx.cfg("mode", "black"))
        # 分割失败时的策略：keep=原图直通（下游仍可用）| skip=丢弃该帧
        on_failure = str(ctx.cfg("on_failure", "keep")).lower()
        if on_failure not in ("keep", "skip"):
            on_failure = "keep"
        dilate = int(ctx.cfg("dilate_px", 0) or 0)
        feather_px = int(ctx.cfg("feather_px", 0) or 0)
        min_area_ratio = float(ctx.cfg("min_area_ratio", 0.002) or 0.0)
        keep_largest = bool(ctx.cfg("keep_largest", True))

        prompt = str(ctx.params.get("mask.prompt", ctx.params.get("prompt", "")) or "")
        points_raw = ctx.params.get("mask.points") or []
        points: list[tuple[float, float]] = []
        for p in points_raw if isinstance(points_raw, list) else []:
            try:
                if isinstance(p, (list, tuple)) and len(p) == 2:
                    points.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError):
                continue

        opts = SegOptions(
            prompt=prompt,
            points=points,
            min_area_ratio=min_area_ratio,
            keep_largest=keep_largest,
            imgsz=int(ctx.cfg("imgsz", 1024) or 1024),
            conf=float(ctx.cfg("conf", 0.25) or 0.25),
            batch_size=int(ctx.cfg("batch_size", 4) or 4),
            # 后端相关选项：`or None` 之类的空值统一交给后端自己取默认
            # （踩坑：曾把空字符串 model="" 原样传给 rembg，导致
            #  "No session class found for model ''"）
            extra={"model": (ctx.params.get("mask.model") or "").strip()},
        )

        ctx.info(f"分割后端 = {backend_name}，清理模式 = {mode}，共 {len(images)} 张图")
        t0 = time.time()
        segmenter = create(backend_name, opts)

        last_report = {"t": 0.0}

        def on_progress(i: int, total: int, fname: str) -> None:
            # 限流：每 0.5s 或每 5 张报一次，避免刷屏
            now = time.time()
            if now - last_report["t"] > 0.5 or i == total:
                last_report["t"] = now
                ctx.info(f"分割 {i}/{total}：{fname}", progress=i / max(total, 1))

        masks = segmenter.segment(images, on_progress=on_progress)

        manifest: list[dict] = []
        failures: list[str] = []
        fallbacks: list[str] = []
        total_fg = 0.0
        for idx, img in enumerate(images, 1):
            mask = masks.get(img.name)
            if mask is None or not mask.any():
                # 空掩码：不整单失败，但要记录。默认“原图直通”，
                # 否则下游 COLMAP 会拿到一张全黑图（等价于丢帧）。
                failures.append(img.name)
                if on_failure == "skip":
                    ctx.warn(f"未检出前景，已跳过：{img.name}")
                    continue
                from PIL import Image as PILImage

                with PILImage.open(img) as im:
                    im.convert("RGB").save(clean_dir / f"{img.stem}.jpg", quality=95)
                # 回退帧也要写两种命名的掩码，否则 COLMAP 会对它报 MASK_ERROR
                ones = np.ones((1, 1), dtype=bool)
                imageops.save_mask_png(ones, mask_dir / f"{img.name}.png")
                imageops.save_mask_png(ones, mask_dir / f"{img.stem}.png")
                fallbacks.append(img.name)
                continue
            if dilate > 0:
                mask = imageops.dilate(mask, dilate)
            alpha = imageops.feather(mask, feather_px) if feather_px > 0 else mask.astype(np.float32)
            rgb = imageops.load_rgb(img)
            # `keep` 模式：**保留原始背景**，只把掩码单独输出。
            #
            # 为什么需要（转台拍法的关键）：物体在转台上转、相机固定时，
            # 静态背景是"相机确实没动"的唯一证据。合成黑底会把背景信息抹掉，
            # 重建只能靠"物体自身的旋转"解释像素 ⇒ 把物体转当成相机转 ⇒
            # 光照/遮挡不自洽 ⇒ 细节糊、网格碎。
            # 实测（scripts/detect_turntable_capture.py）：
            #   img3-01 背景帧间差异 0.34（比值 0.013）⇒ 转台；
            #   maskloss-v2 17.88（比值 0.489）⇒ 相机绕行。
            # 转台素材应用 `mask.mode=keep`，让背景参与监督，导出阶段再裁出物体。
            if mode == "keep":
                clean = rgb
            elif mode == "black":
                clean = imageops.apply_black_background(rgb, alpha)
            else:
                clean = imageops.apply_alpha(rgb, alpha, background=0)
            out_img = clean_dir / f"{img.stem}.jpg"
            from PIL import Image

            Image.fromarray(clean).save(out_img, quality=95)
            # 两种命名都写：{stem}.png 便于我们自己读写，
            # {image_name}.png 是 COLMAP 的规则（否则报 MASK_ERROR 拒绝该帧）
            imageops.save_mask_png(mask, mask_dir / f"{img.name}.png")
            imageops.save_mask_png(mask, mask_dir / f"{img.stem}.png")
            if idx <= 50 or idx % 25 == 0:
                imageops.save_overlay(rgb, alpha, overlay_dir / f"{img.stem}.jpg")

            stats = imageops.mask_stats(mask)
            total_fg += stats["ratio"]
            manifest.append(
                {
                    "image": img.name,
                    "mask": f"{img.stem}.png",
                    "clean": out_img.name,
                    "size": list(rgb.shape[:2]),
                    **stats,
                }
            )
            if idx % 10 == 0 or idx == len(images):
                ctx.info(f"清理 {idx}/{len(images)}", progress=0.5 + 0.5 * idx / max(len(images), 1))

        summary = {
            "backend": backend_name,
            "mode": mode,
            "prompt": prompt,
            "points": points,
            "count": len(images),
            "ok": len(manifest),
            "failed": failures,
            "fallback_kept_original": fallbacks,
            "mean_foreground_ratio": round(total_fg / max(len(manifest), 1), 4),
            "elapsed_s": round(time.time() - t0, 1),
            "params": {"dilate_px": dilate, "feather_px": feather_px,
                       "min_area_ratio": min_area_ratio, "keep_largest": keep_largest,
                       "on_failure": on_failure},
        }
        (mask_dir / "manifest.json").write_text(json.dumps({"summary": summary, "items": manifest},
                                                          ensure_ascii=False, indent=2), encoding="utf-8")

        if failures:
            ctx.warn(f"{len(failures)} 张图未检出前景：{failures[:10]}")
        if fallbacks:
            ctx.warn(f"{len(fallbacks)} 张已回退为“原图直通”（这些图不做背景清理）")
        if not manifest:
            if fallbacks:
                ctx.warn("所有图都未检出前景，已全部回退为原图直通；建议换后端或补提示词")
                return StageResult(
                    artifacts=[str(clean_dir)],
                    metrics=summary,
                    message=f"未能分割，已回退原图直通（{len(fallbacks)} 张）",
                )
            raise StageError(
                self.name,
                "所有图片都没有检出前景，且已按 skip 策略全部丢弃",
                "换后端（--set mask.backend=rembg）或提供提示（--set mask.prompt=目标名称 / mask.points）",
            )
        return StageResult(
            artifacts=[str(mask_dir), str(clean_dir)],
            metrics=summary,
            message=f"{len(manifest)}/{len(images)} 张清理完成，平均前景占比 {summary['mean_foreground_ratio']:.1%}",
        )
