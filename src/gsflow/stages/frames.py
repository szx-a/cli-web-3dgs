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
        if out_dir.exists():
            for p in out_dir.iterdir():
                if p.is_file():
                    p.unlink()
        copied = 0
        for i, f in enumerate(files):
            dst = out_dir / f"{i:05d}{f.suffix.lower()}"
            if long_side > 0:
                self._resize_to(f, dst, long_side)
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
        if mode == "fps":
            cmd += ["-r", f"{fps}"]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        cmd += ["-qscale:v", "2", "-vsync", "0", str(out_dir / "%05d.jpg")]
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
    def _resize_to(src: Path, dst: Path, long_side: int) -> None:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise StageError("frames", "缺少 Pillow，无法缩放", "pip install pillow") from exc
        with Image.open(src) as im:
            w, h = im.size
            if max(w, h) <= long_side:
                im.save(dst)
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
