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

    def _resolve_capture_mode(self, ctx: StageContext) -> tuple[str, str]:
        """解析 `capture.mode`，返回 (实际模式, 说明)。

        `auto` 时用背景帧间差异自动判定；判定前需要有抽帧产物
        （`01_images_raw`），若还没有就回退 orbit 并说明原因。
        """
        from ..gsdata.capture_mode import detect_from_frames, detect_without_masks
        from ..gsdata.capture_mode_params import resolve_capture_mode

        chosen = str(ctx.params.get("capture.mode", "auto") or "auto").strip().lower()
        detected = None
        try:
            # ⚠️ 两个检测函数都接受**作业根目录**（内部拼 `01_images_raw`），
            # 不是 `01_images_raw` 本身。我第一版传了后者，
            # 于是内部拼成 `01_images_raw/01_images_raw` ⇒ 不存在 ⇒ 永远回退 orbit。
            # 同一个错误我在测试脚本里也犯了一次，排查花了两轮。
            if (ctx.job_dir / "01_images_raw").exists():
                # 优先用掩码判据（更准）；拿不到掩码时用"画面静止比例"兜底。
                # 为什么需要兜底：本阶段本身就是**产出掩码**的地方，
                # 首次运行时还没有掩码，否则会出现"缺掩码 ⇒ 无法判定 ⇒ 回退 orbit"
                # 的死循环（实测过）。
                analysis = detect_from_frames(ctx.job_dir, max_pairs=30)
                if analysis is None:
                    analysis = detect_without_masks(ctx.job_dir, max_pairs=24)
                if analysis is not None:
                    detected = analysis.mode
                    ctx.info(f"拍摄模式自动判定：{analysis.reason}")
        except Exception as exc:  # noqa: BLE001
            ctx.warn(f"拍摄模式自动判定失败（{type(exc).__name__}: {exc}），按未判定处理")
        mode, note = resolve_capture_mode(chosen, detected)
        return mode, note

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
        # 掩码收紧（默认 0）：实测本数据掩码整体偏大，会学出轮廓白边（docs/03 G134）
        mode = str(ctx.cfg("mode", "black"))
        # ---- 拍摄模式（capture.mode）在这里**落地** ----
        # 为什么要在此处解析：`capture.mode` 若不参与任何阶段，就只是"配置了但没执行"。
        # 两种拍法的实质差别是"要不要保留背景"：
        #   turntable（物体转、相机固定）⇒ 必须保留背景，否则重建只能靠物体自身旋转解释像素，
        #     把物体转当成相机转 ⇒ 光照/遮挡不自洽（实测：保留背景后 PSNR 20.74→28.78）。
        #   orbit（相机绕行）⇒ 抠背景 + 黑底合成为安全。
        # `auto` 用 `gsdata/capture_mode.detect_from_frames` 按背景帧间差异自动判定
        # （实测两类素材比值 0.015 vs 0.489，相差 34 倍）。
        cap_mode, cap_note = self._resolve_capture_mode(ctx)
        if cap_mode == "turntable" and mode != "keep":
            ctx.info(f"拍摄模式=turntable ⇒ 自动改用 mask.mode=keep（原为 {mode}）：{cap_note}")
            mode = "keep"
        elif cap_mode == "orbit" and mode == "keep":
            ctx.warn(f"拍摄模式=orbit 但显式指定了 mask.mode=keep；"
                     f"相机在绕行时保留背景会引入游离高斯，建议用 black")
        ctx.info(f"拍摄模式 = {cap_mode}（{cap_note}）")
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

        # ---- 退化检测 + 自动兜底（2026-10-24，鲁棒性）----
        #
        # 事故：在"已经裁好的帧"上跑 fastsam，**每张都返回空掩码**，日志里
        # "平均前景占比 0.0%" 却继续往下跑，直到 COLMAP 才崩。属于**静默失败**。
        # 这里在落盘前先算前景比例：若几乎为 0（或几乎全图），换 classical 后端重试一次；
        # 仍退化就**明确报错**，而不是把空掩码传给下游。
        _fg_ratios = [float(m.mean()) for m in masks.values() if m is not None and m.size]
        _mean_fg = float(sum(_fg_ratios) / max(len(_fg_ratios), 1))
        # ⚠️ mode=keep（转台拍法）时掩码本就是全 1，不能当退化
        # 阈值收紧（2026-10-24）：实测坏掩码往往不是“全 0”而是“大面积包含背景”，
        # 前景占比会偏大（如 80~95%），而下游不会报错。所以把合理区间收到 2%~90%。
        if mode != "keep" and _fg_ratios and (_mean_fg < 0.02 or _mean_fg > 0.90):
            ctx.warn(f"分割退化：平均前景占比 {_mean_fg*100:.2f}%"
                     f"（{len(_fg_ratios)} 张）⇒ 换 classical 后端重试一次")
            try:
                segmenter2 = create("classical", opts)
                masks2 = segmenter2.segment(images, on_progress=on_progress)
                _fg2 = [float(m.mean()) for m in masks2.values() if m is not None and m.size]
                _mean2 = float(sum(_fg2) / max(len(_fg2), 1)) if _fg2 else 0.0
                if 0.02 <= _mean2 <= 0.90:
                    ctx.warn(f"classical 兜底成功：平均前景 {_mean2*100:.1f}%"
                             f"（{backend_name} 为 {_mean_fg*100:.2f}%）")
                    masks, backend_name, _mean_fg = masks2, "classical", _mean2
                else:
                    # ⚠️ 降级而不是失败（2026-10-24）：两个后端都找不到前景时，
                    # 说明“这批图本身就不适合做掩码”。硬失败会把整单卡死；
                    # 更稳当的行为是**降级为 keep**（全图参与）并报得非常响。
                    ctx.warn(
                        f"分割两个后端都没找到前景（{backend_name} {_mean_fg*100:.2f}%、"
                        f"classical {_mean2*100:.2f}%）⇒ **降级为 keep**（全图参与、不做背景处理）")
                    mode = "keep"
                    masks = {}
            except StageError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise StageError(
                    self.name,
                    f"分割退化且兜底失败：{backend_name} 前景 {_mean_fg*100:.2f}%",
                    f"原因：{type(exc).__name__}: {exc}。"
                    "可改用 mask.backend=rembg，或对已裁好的帧跳过 mask 阶段") from exc

        # ---- 两条拍摄模式判据的一致性复核（2026-10-24）----
        #
        # 实测：同一份输入，**无掩码判据**判成 turntable（保留背景），
        # **有掩码判据**判成 orbit（抠黑）。两条路径结论不一致，而走哪条取决于执行顺序
        # ⇒ 同一命令重跑会得到不同结果（img3-v2 掩码内 21.96/0.9267 vs img3-v7 21.00/0.9485）。
        # 这里在掩码产生后再用掩码判据复核一次；不一致就告警并写进 capture_mode.json。
        try:
            from ..gsdata.capture_mode import detect_from_frames as _detect_m

            _alt = _detect_m(ctx.job_dir, max_pairs=20)
            if _alt is not None and _alt.mode != cap_mode:
                ctx.warn(
                    f"拍摄模式两条判据不一致：无掩码={cap_mode}、有掩码={_alt.mode}"
                    f"（{_alt.reason[:60]}）。建议用 --capture-mode 显式指定，"
                    f"否则结果会随执行顺序变化")
                cap_note = (cap_note or "") + f"｜有掩码判据为 {_alt.mode}"
        except Exception as _exc:  # noqa: BLE001
            ctx.warn(f"有掩码判据复核跳过：{type(_exc).__name__}")

        # ---- 掩码自动收紧（2026-10-24，鲁棒性）----
        #
        # 实测：分割后端产出的掩码经常“整体偏大”——掩码内靠边那圈实际是背景。
        #   img3-v2 rim_ratio=**0.70**、img3-v4 rim_ratio=**1.89**（0 为贴合），
        # 两个作业的掩码都不合格；尤其 v4 掩码把大片空白圈成前景。
        #
        # 做法：在采样帧上搜索最小腐蚀 k（0..6），使平均 rim_ratio ≤ 0.30 且前景 ≥ 15%。
        # 若背景本身很暗（外部亮度 <0.25）则无法用亮度判据，退回配置值。
        _erode = int(ctx.cfg("erode_px", 0) or 0)
        auto_erode = 0
        try:
            from PIL import Image as _PILImage
            from scipy import ndimage as _nd2

            _sample_idx = list(range(0, len(images), max(1, len(images) // 12)))[:12]
            _best = None
            for _k in range(0, 7):
                _ratios = []
                for _i in _sample_idx:
                    _img_path = images[_i]
                    _m = masks.get(_img_path.name)
                    if _m is None or not np.asarray(_m).any():
                        continue
                    _mb = np.asarray(_m) > (127 if np.asarray(_m).dtype.kind in "iu" and np.asarray(_m).max() > 1 else 0.5)
                    if _k:
                        _mb = _nd2.binary_erosion(_mb, iterations=_k)
                    if not _mb.any():
                        break
                    with _PILImage.open(_img_path) as _im:
                        _g = np.asarray(_im.convert("L"), dtype=np.float64) / 255.0
                    if _g.shape != _mb.shape:
                        continue
                    _inner = _nd2.binary_erosion(_mb, iterations=4)
                    _rim = _mb & ~_inner
                    _outer = _nd2.binary_dilation(_mb, iterations=4) & ~_mb
                    if not (_inner.any() and _rim.any() and _outer.any()):
                        continue
                    _im_v, _rm_v, _ou_v = _g[_inner].mean(), _g[_rim].mean(), _g[_outer].mean()
                    if abs(_ou_v - _im_v) < 0.05 or _ou_v < 0.25:
                        # 背景与前景对比不足 / 背景很暗 ⇒ 该帧不可用于判据
                        continue
                    _ratios.append((_rm_v - _im_v) / abs(_ou_v - _im_v))
                if _ratios:
                    _mean_r = float(np.mean(_ratios))
                    _fg = float(np.mean([np.asarray(masks[images[_i].name]).astype(bool).mean()
                                         for _i in _sample_idx if masks.get(images[_i].name) is not None])) if _sample_idx else 0.0
                    if _mean_r <= 0.30 and _fg >= 0.15:
                        _best = (_k, _mean_r)
                        break
                    if _best is None or _mean_r < _best[1]:
                        _best = (_k, _mean_r)
            if _best and _best[0] > 0:
                auto_erode = _best[0]
                ctx.warn(f"掩码自动收紧：需腐蚀 **{auto_erode}px**"
                         f"才能把 rim_ratio 降到 {_best[1]:.2f}（≤ 0.30 为合格）")
        except Exception as _exc:  # noqa: BLE001
            ctx.warn(f"掩码自动收紧跳过：{type(_exc).__name__}")
        _erode = max(_erode, auto_erode)

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
            # ---- 掩码收紧（`mask.erode_px`，默认 0 = 不改行为）----
            #
            # 实测（docs/03 G134）：本数据集的掩码**整体偏大**——掩码内靠边 4px 环的
            # 真值亮度 0.89，几乎等于掩码外背景 0.91，而掩码内部只有 0.43~0.51。
            # 训练会把这一圈亮背景当物体 ⇒ 渲染出**轮廓白边/发光**
            # （`rim_bright` 0.295；收紧到 4px 后 0.037）。
            #
            # 为什么放在**写盘**而不是损失里：训练与评估必须用同一套掩码，
            # 否则等于"训练不管边界、评估却考核边界"（实测会让 SSIM 从 0.948 掉到 0.863）。
            if _erode > 0:
                from scipy import ndimage as _nd

                # ⚠️ dtype 必须分流（2026-10-24 自己踩的坑）：
                # 分割后端返回的是 **bool**（True=前景），旧写法 > 127 会把
                # True(=1) 全判成 False ⇒ **掩码全黑**。当时日志只报
                # “平均前景 0.0%”，直到 COLMAP 才发现。
                _m_arr = np.asarray(mask)
                if _m_arr.dtype == np.bool_:
                    m_bool = _m_arr
                elif np.issubdtype(_m_arr.dtype, np.floating):
                    m_bool = _m_arr > (0.5 if float(_m_arr.max()) <= 1.0 else 127.0)
                else:
                    m_bool = _m_arr > 127
                if m_bool.any():
                    m_bool = _nd.binary_erosion(m_bool, iterations=_erode)
                mask = m_bool.astype(np.uint8) * 255
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
            # **实际生效**的清理模式与拍摄模式。
            # 为什么要显式记录：`capture.mode=turntable` 会把 `mask.mode` 自动改成 keep，
            # 但 `job.json` 的 `settings_snapshot` 记录的是**用户请求**（black），
            # 于是"快照"与"实际行为"不一致 —— 排查时曾因此误判 keep 没生效。
            # 这里把解析后的真实值落到阶段输出，作为**唯一可信的行为记录**。
            "mask_mode_effective": mode,
            "capture_mode_effective": cap_mode,
            "capture_mode_note": cap_note,
        }
        # 再写一份独立文件，便于脚本不解析 manifest 就能读到
        (mask_dir / "capture_mode.json").write_text(json.dumps({
            "capture_mode": cap_mode,
            "capture_mode_note": cap_note,
            "mask_mode_effective": mode,
            "mask_mode_requested": str(ctx.cfg("mode", "black")),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
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
