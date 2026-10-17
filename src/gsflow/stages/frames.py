"""阶段 1：视频抽帧 / 图片导入。

产物：<job>/01_images_raw/*.jpg（+ _frames.json 元数据）
后续所有阶段都只吃这个目录，所以视频与图片输入在这里被归一化。

注意：元数据文件名必须带下划线前缀（_frames.json），否则 COLMAP 会把它
当成图像去读，报 "BITMAP_ERROR" 并浪费一次探测。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..errors import StageError
from ..tools.locate import find_tool, probe_version, run_tool
from ..pipeline.base import Artifact, Stage, StageResult
from ..pipeline.context import StageContext

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv", ".webm", ".mts"}
META_NAME = "_frames.json"


def _fps_mode_flag(ffmpeg_path: str) -> str:
    """返回当前 ffmpeg 支持的帧同步选项名（`-fps_mode` 或旧的 `-vsync`）。

    踩坑：`-vsync` 在 ffmpeg 5.0 起废弃、**9.0 已移除**，老代码传它会在真实视频上
    直接抽帧失败（"Unrecognized option 'vsync'"）。本项目此前只用预置图片测试，
    从未走过视频路径，所以直到跑真实素材才暴露。
    这里探测一次 ffmpeg 的帮助输出，按版本选名字，避免绑死某个版本。
    """
    import functools
    import subprocess

    @functools.lru_cache(maxsize=4)
    def _probe(path: str) -> str:
        try:
            out = subprocess.run(
                [path, "-hide_banner", "-h", "full"],
                capture_output=True, text=True, errors="ignore", timeout=30,
            )
            text = (out.stdout or "") + (out.stderr or "")
            if "-fps_mode" in text:
                return "-fps_mode"
        except Exception:  # noqa: BLE001
            pass
        return "-vsync"

    return _probe(ffmpeg_path)


class FramesStage(Stage):
    name = "frames"
    title = "① 抽帧 / 图片导入"
    description = "把视频按帧率或目标张数抽成图片，或直接导入图片目录，统一到 01_images_raw/"
    requires = ()
    inputs = ()
    outputs = (Artifact("01_images_raw", "归一化后的原始图片"),)

    def run(self, ctx: StageContext) -> StageResult:
        out_dir = ctx.stage_dir("01_images_raw")
        spec = ctx.spec
        src = (spec.get("images_dir") or spec.get("input_path") or "").strip()
        if not src:
            raise StageError(self.name, "未提供输入路径", "用 --images <目录> 或 --video <文件> 指定输入")
        src_path = Path(src.replace('"', ""))
        if not src_path.exists():
            raise StageError(self.name, f"输入不存在：{src_path}", "检查路径是否含中文空格或被引号包住")

        if src_path.is_dir():
            return self._import_images(ctx, src_path, out_dir)
        if src_path.suffix.lower() in VIDEO_EXT:
            return self._extract_video(ctx, src_path, out_dir)
        if src_path.suffix.lower() in IMAGE_EXT:
            return self._import_images(ctx, src_path.parent, out_dir, only=src_path.name)
        raise StageError(self.name, f"无法识别的输入类型：{src_path.name}", "支持视频文件、图片目录、单张图片")

    # ------------------------------------------------------------ 图片输入
    def _import_images(self, ctx: StageContext, src: Path, out_dir: Path, only: str | None = None) -> StageResult:
        files = [src / only] if only else sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT)
        if not files:
            raise StageError(self.name, f"目录中没有图片：{src}", "确认扩展名是 jpg/png 等常见格式")

        long_side = int(ctx.cfg("long_side", 0) or 0)
        # 可选裁剪 `x,y,w,h`（在缩放之前裁）—— 见 `_resize_to` 的说明。
        # 物体在画面里很小时，裁掉空白等于把有效分辨率提高数倍，比调参更直接。
        crop = str(ctx.cfg("crop", "") or "").strip()
        if crop:
            ctx.info(f"按 crop={crop} 裁掉空白（x,y,w,h）")
        if out_dir.exists():
            for p in out_dir.iterdir():
                if p.is_file():
                    p.unlink()
        copied = 0
        for i, f in enumerate(files):
            dst = out_dir / f"{i:05d}{f.suffix.lower()}"
            if long_side > 0 or crop:
                self._resize_to(f, dst, long_side, crop)
            else:
                shutil.copy2(f, dst)
            copied += 1
            if copied % 20 == 0 or copied == len(files):
                ctx.info(f"导入图片 {copied}/{len(files)}", progress=copied / max(len(files), 1))
        metrics = {"source": str(src), "mode": "images", "count": copied}
        (out_dir / META_NAME).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return StageResult(
            artifacts=[str(out_dir)],
            metrics=metrics,
            message=f"导入 {copied} 张图片",
        )

    # ------------------------------------------------------------ 视频输入
    def _extract_video(self, ctx: StageContext, src: Path, out_dir: Path) -> StageResult:
        ffmpeg = find_tool("ffmpeg", str(ctx.cfg("tools.ffmpeg", "") or ""))
        ffprobe = find_tool("ffprobe", str(ctx.cfg("tools.ffprobe", "") or ""))
        assert ffmpeg and ffprobe
        ctx.info(f"ffmpeg: {probe_version(ffmpeg.path)}")

        duration = self._probe_duration(ffprobe.path, src)
        mode = str(ctx.cfg("mode", "fps"))
        long_side = int(ctx.cfg("long_side", 0) or 0)
        fps = float(ctx.cfg("fps", 2.0) or 2.0)
        count = int(ctx.cfg("count", 120) or 120)

        vf = []
        # 可选裁剪（视频路径用 ffmpeg 的 crop 滤镜）。crop 必须在 scale **之前**，
        # 否则裁剪坐标是相对缩放后尺寸的，会全裁歪。
        crop_v = str(ctx.cfg("crop", "") or "").strip()
        if crop_v:
            try:
                cx, cy, cw, ch = [int(v) for v in crop_v.split(",")]
                # 保证偶数尺寸（H.264 要求），且不越界
                cw, ch = cw - (cw % 2), ch - (ch % 2)
                if cw > 1 and ch > 1:
                    vf.append(f"crop={cw}:{ch}:{cx}:{cy}")
                    ctx.info(f"按 crop={cx},{cy},{cw},{ch} 裁掉空白（在缩放之前）")
            except (ValueError, TypeError):
                ctx.warn(f"crop 参数格式不对（应为 x,y,w,h），忽略：{crop_v!r}")
        if mode == "count":
            if duration <= 0:
                raise StageError(self.name, "无法读取视频时长，无法按张数抽帧", "改用 mode=fps 抽帧")
            fps = max(count / duration, 0.01)
            ctx.info(f"视频时长 {duration:.1f}s，按 {count} 张均匀抽帧（等效 {fps:.3f} fps）")
        if long_side > 0:
            # 长边缩放，保持偶数尺寸（H.264 编码器要求）
            vf.append(f"scale='if(gt(iw,ih),{long_side},-2)':'if(gt(iw,ih),-2,{long_side})'")
        if out_dir.exists():
            for p in out_dir.iterdir():
                if p.is_file():
                    p.unlink()
        cmd = [ffmpeg.path, "-hide_banner", "-loglevel", "warning", "-y", "-i", str(src)]
        # `-r` 必须在两种模式都加：count 模式只是先把"目标张数"换算成等效 fps，
        # 真正让 ffmpeg 降采样的是这个 `-r`。
        # 踩坑：早期把它写在 `if mode == "fps"` 里，于是 count 模式算出 fps 却没用上，
        # 结果 23 秒 30fps 的视频被判成"抽 120 张"实际抽出 **698 张（全部帧）**，
        # 后续 mask 阶段白白处理了近 6 倍数据。
        cmd += ["-r", f"{fps}"]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        # 帧同步选项的坑（两层）：
        #   1) ffmpeg 5.0 起 `-vsync` 废弃、**9.0 已彻底移除**，传它会直接
        #      "Unrecognized option 'vsync'"。新名字是 `-fps_mode`。
        #   2) 更重要的是**取值必须匹配**：一旦用了 `-r`（我们要按目标张数降采样），
        #      就只能配 CFR，配 `passthrough` 会报
        #      "One of -r/-fpsmax was specified together a non-CFR -vsync/-fps_mode.
        #       This is contradictory."
        #      （第一版就是踩了这个：没加 -r 时 passthrough 能跑，加了 -r 就冲突。）
        # 所以这里统一用 `cfr`：既有确定的帧率、又不会静默丢掉我们要的帧。
        cmd += ["-qscale:v", "2", _fps_mode_flag(ffmpeg.path), "cfr"]
        cmd += [str(out_dir / "%05d.jpg")]
        log = ctx.logs_dir() / "ffmpeg_frames.log"
        ctx.info("开始抽帧 ...")
        code = run_tool(cmd, log_file=log, on_line=lambda l: ctx.emit(l, level="debug"))
        if code != 0:
            raise StageError(self.name, f"ffmpeg 抽帧失败（exit={code}）", f"详见日志 {log}")
        frames = sorted(out_dir.glob("*.jpg"))
        if not frames:
            raise StageError(self.name, "抽帧结果为空", "检查视频编码或降低 fps")

        max_frames = int(ctx.cfg("max_frames", 0) or 0)
        if max_frames and len(frames) > max_frames:
            keep = {frames[round(i * (len(frames) - 1) / (max_frames - 1))] for i in range(max_frames)}
            for f in frames:
                if f not in keep:
                    f.unlink()
            frames = sorted(keep)
            ctx.info(f"按 max_frames 均匀保留 {len(frames)} 张")

        blur = float(ctx.cfg("blur_threshold", 0.0) or 0.0)
        dropped = 0
        if blur > 0:
            dropped = self._drop_blurry(ctx, frames, blur, out_dir)
            frames = sorted(out_dir.glob("*.jpg"))
        metrics = {
            "source": str(src),
            "mode": mode,
            "fps": fps,
            "duration_s": round(duration, 2),
            "count": len(frames),
            "dropped_blurry": dropped,
            "long_side": long_side,
        }
        (out_dir / META_NAME).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return StageResult(artifacts=[str(out_dir)], metrics=metrics,
                           message=f"抽帧 {len(frames)} 张（丢弃模糊 {dropped} 张）")

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _probe_duration(ffprobe: str, src: Path) -> float:
        from ..tools import runner

        code, out = runner.probe(
            ffprobe,
            ["-v", "error", "-show_entries", "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", str(src)],
            timeout=40,
        )
        try:
            return float((out or "0").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return 0.0

    @staticmethod
    def _resize_to(src: Path, dst: Path, long_side: int, crop: str = "") -> None:
        """缩放（只缩不放），可选先按 `crop = x,y,w,h` 裁剪。

        **为什么需要裁剪**：物体在画面里很小时，网络容量与高斯预算大量浪费在空白上。
        实测 IMG3（576×1280）：牌坊只占 **8.5%** 像素、联合包围盒仅 16.7%；
        按包围盒裁到 489×316 后前景占比升到 **79.6%（9.4 倍）**，
        等效上等于把训练分辨率提高数倍 —— 这比调任何损失/正则都直接。
        （用 `scripts/measure_foreground_bbox.py` 求联合包围盒，它保证任何一帧都不被裁掉。）
        """
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise StageError("frames", "缺少 Pillow，无法缩放", "pip install pillow") from exc
        with Image.open(src) as im:
            if crop:
                try:
                    cx, cy, cw, ch = [int(v) for v in str(crop).split(",")]
                    W0, H0 = im.size
                    cx = max(0, min(cx, W0 - 1))
                    cy = max(0, min(cy, H0 - 1))
                    cw = max(1, min(cw, W0 - cx))
                    ch = max(1, min(ch, H0 - cy))
                    im = im.crop((cx, cy, cx + cw, cy + ch))
                except (ValueError, TypeError):
                    pass          # 解析失败就不裁（保持原行为）
            w, h = im.size
            if long_side <= 0 or max(w, h) <= long_side:
                im.convert("RGB").save(dst, quality=95)
                return
            scale = long_side / max(w, h)
            im = im.convert("RGB").resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            im.save(dst, quality=95)

    def _drop_blurry(self, ctx: StageContext, frames: list[Path], threshold: float, out_dir: Path) -> int:
        """按梯度方差剔除模糊帧（纯 numpy/PIL，不依赖 OpenCV）。"""
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            return 0
        dropped = 0
        kept: list[float] = []
        for f in frames:
            with Image.open(f) as im:
                arr = np.asarray(im.convert("L").resize((256, 256)), dtype="float32")
            gy, gx = np.gradient(arr)
            var = float((gx ** 2 + gy ** 2).mean())
            kept.append(var)
            if var < threshold:
                f.unlink()
                dropped += 1
        if kept:
            ctx.info(f"清晰度方差：min={min(kept):.1f} mean={sum(kept)/len(kept):.1f} max={max(kept):.1f}")
        return dropped
