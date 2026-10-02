"""阶段 6：质量度量与报告。

输入：03_colmap 的 TXT 模型 + 02_images_clean
产物：06_reports/report.md、report.json、colmap_quality.json

这一阶段刻意保持“轻量 + 无强依赖”：没有 torch 也能出注册率、
重叠度、掩码占比这些流程级指标；装了 torch 才额外算 PSNR/SSIM。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..pipeline.base import Artifact, Stage, StageResult
from ..pipeline.context import StageContext

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class MetricsStage(Stage):
    name = "metrics"
    title = "⑥ 质量度量与报告"
    description = "汇总注册率/重投影误差/掩码占比/训练指标，生成 markdown 报告"
    requires = ("train",)
    optional = True
    inputs = ()
    outputs = (Artifact("06_reports/report.md", "人类可读报告"),)

    def run(self, ctx: StageContext) -> StageResult:
        out_dir = ctx.stage_dir("06_reports")
        report: dict = {"job_id": ctx.job_id, "stages": {}, "warnings": []}

        # --- 各阶段 metrics.json 汇总 ---
        for stage_name, rels in (
            ("frames", ("01_images_raw/_frames.json", "01_images_raw/frames.json")),
            ("mask", ("02_masks/manifest.json",)),
            ("colmap", ("03_colmap/report.json",)),
            ("train", ("04_train/metrics.json",)),
            ("export", ("05_export/manifest.json",)),
        ):
            p = next((ctx.job_dir / r for r in rels if (ctx.job_dir / r).exists()), None)
            if p is None:
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            report["stages"][stage_name] = data.get("summary", data)

        # --- 输入体检：清晰度与重叠度 ---
        raw_dir = ctx.job_dir / "01_images_raw"
        clean_dir = ctx.job_dir / "02_images_clean"
        mask_dir = ctx.job_dir / "02_masks"
        report["input"] = self._input_stats(raw_dir, clean_dir, mask_dir, ctx)

        # --- 训练渲染指标（有 torch 才算）---
        report["render"] = self._render_metrics(ctx)

        # --- 结论与建议 ---
        report["suggestions"] = self._suggest(report)

        (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        md = self._to_markdown(report)
        (out_dir / "report.md").write_text(md, encoding="utf-8")
        ctx.success("报告已生成：06_reports/report.md")
        for s in report["suggestions"]:
            ctx.warn(s)
        return StageResult(artifacts=[str(out_dir / "report.md"), str(out_dir / "report.json")],
                           metrics={"suggestions": report["suggestions"]},
                           message=f"{len(report['suggestions'])} 条改进建议")

    # ------------------------------------------------------------------ 统计
    def _input_stats(self, raw_dir: Path, clean_dir: Path, mask_dir: Path, ctx: StageContext) -> dict:
        raw = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in IMAGE_EXT) if raw_dir.exists() else []
        clean = sorted(p for p in clean_dir.iterdir() if p.suffix.lower() in IMAGE_EXT) if clean_dir.exists() else []
        stats: dict = {"raw_images": len(raw), "clean_images": len(clean)}

        if raw:
            try:
                from PIL import Image

                with Image.open(raw[0]) as im:
                    stats["first_image_size"] = list(im.size)
            except Exception:
                pass

        # 清晰度（拉普拉斯方差近似）
        if raw:
            try:
                from PIL import Image

                variances = []
                for p in raw[:: max(1, len(raw) // 40)]:
                    with Image.open(p) as im:
                        a = np.asarray(im.convert("L").resize((256, 256)), dtype="float32")
                    gy, gx = np.gradient(a)
                    variances.append(float((gx ** 2 + gy ** 2).mean()))
                stats["sharpness"] = {
                    "min": round(min(variances), 1),
                    "mean": round(float(np.mean(variances)), 1),
                    "max": round(max(variances), 1),
                }
                if np.mean(variances) < 5:
                    stats.setdefault("notes", []).append("整体偏模糊：建议提高快门速度/增加光照，或开启 frames.blur_threshold 剔除")
            except Exception:
                pass

        # 掩码占比
        masks = sorted(mask_dir.glob("*.png")) if mask_dir.exists() else []
        if masks:
            try:
                from PIL import Image

                ratios = []
                for p in masks[:: max(1, len(masks) // 40)]:
                    with Image.open(p) as im:
                        m = np.asarray(im.convert("L"), dtype=np.uint8) > 127
                    ratios.append(float(m.mean()))
                stats["mask"] = {
                    "count": len(masks),
                    "mean_foreground_ratio": round(float(np.mean(ratios)), 4),
                    "min": round(float(np.min(ratios)), 4),
                    "max": round(float(np.max(ratios)), 4),
                }
                if np.mean(ratios) < 0.05:
                    stats.setdefault("notes", []).append("前景占比过低，可能是分割后端选错或提示不到位")
                if np.mean(ratios) > 0.9:
                    stats.setdefault("notes", []).append("前景几乎占满全图，掩码可能没起作用")
            except Exception:
                pass

        # 相邻帧重叠估计（用图像缩略图的像素差近似，给出“是否拍得太稀”的提示）
        if len(clean) >= 4:
            try:
                from PIL import Image

                diffs = []
                step = max(1, len(clean) // 20)
                prev = None
                for p in clean[::step]:
                    with Image.open(p) as im:
                        a = np.asarray(im.convert("L").resize((64, 64)), dtype="float32")
                    if prev is not None:
                        diffs.append(float(np.abs(a - prev).mean()))
                    prev = a
                stats["adjacent_frame_diff"] = {
                    "mean": round(float(np.mean(diffs)), 2),
                    "max": round(float(np.max(diffs)), 2),
                }
                if np.mean(diffs) > 42:
                    stats.setdefault("notes", []).append("相邻帧差异大：抽帧太稀，建议提高 fps 重跑以增加视角重叠")
            except Exception:
                pass
        return stats

    def _render_metrics(self, ctx: StageContext) -> dict:
        """如果训练阶段已经算了 PSNR，就沿用；否则不引入重依赖。"""
        train_metrics = ctx.job_dir / "04_train" / "metrics.json"
        if not train_metrics.exists():
            return {"available": False, "reason": "尚未训练"}
        try:
            data = json.loads(train_metrics.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"available": False, "reason": "训练指标不可读"}
        return {
            "available": data.get("psnr") is not None,
            "psnr": data.get("psnr"),
            "gaussians": data.get("gaussians"),
            "resolution": data.get("resolution"),
            "elapsed_min": data.get("elapsed_min"),
            "psnr_curve": data.get("psnr_curve", []),
        }

    def _suggest(self, report: dict) -> list[str]:
        out: list[str] = []
        colmap = report["stages"].get("colmap", {})
        ratio = colmap.get("registered_ratio")
        if ratio is not None:
            if ratio < 0.6:
                out.append(f"COLMAP 注册率只有 {ratio:.0%}：建议提高抽帧密度、改用 exhaustive 匹配，或检查是否有大量模糊/重复帧")
            elif ratio < 0.85:
                out.append(f"COLMAP 注册率 {ratio:.0%} 偏低：可尝试调大 colmap.sequential_overlap 或降低 mask.dilate_px")
        err = colmap.get("mean_reproj_error")
        if err is not None and err > 1.5:
            out.append(f"平均重投影误差 {err:.2f}px 偏大：标定不稳定，可能是掩码过激或场景纹理太弱")
        render = report.get("render", {})
        psnr = render.get("psnr")
        if psnr is not None:
            if psnr < 20:
                out.append(f"PSNR {psnr:.1f}dB 偏低：常见原因是位姿不准、掩码吃掉纹理，或迭代不足")
            elif psnr > 30:
                out.append(f"PSNR {psnr:.1f}dB 表现良好")
        mask_stats = report["input"].get("mask")
        if mask_stats and mask_stats["mean_foreground_ratio"] > 0.85:
            out.append("前景占比过高：确认分割后端是否真的在分割（例如 FastSAM 的 prompt 未命中）")
        return out

    # ------------------------------------------------------------------ 报告
    def _to_markdown(self, report: dict) -> str:
        lines = [f"# 重建报告 · 作业 {report['job_id']}", ""]
        inp = report.get("input", {})
        lines += ["## 输入体检", ""]
        lines.append(f"- 原始图片：{inp.get('raw_images')} 张；清理后图片：{inp.get('clean_images')} 张")
        if inp.get("first_image_size"):
            lines.append(f"- 首图尺寸：{inp['first_image_size'][0]}x{inp['first_image_size'][1]}")
        if inp.get("sharpness"):
            s = inp["sharpness"]
            lines.append(f"- 清晰度（梯度方差）：min {s['min']} / mean {s['mean']} / max {s['max']}")
        if inp.get("mask"):
            m = inp["mask"]
            lines.append(f"- 掩码前景占比：均值 {m['mean_foreground_ratio']:.1%}（{m['min']:.1%} ~ {m['max']:.1%}）")
        if inp.get("adjacent_frame_diff"):
            lines.append(f"- 相邻帧差异均值：{inp['adjacent_frame_diff']['mean']}（越大说明视角间隔越大）")
        for note in inp.get("notes", []):
            lines.append(f"- ⚠️ {note}")
        lines.append("")

        lines += ["## 各阶段指标", ""]
        for name, data in report["stages"].items():
            lines.append(f"### {name}")
            lines.append("```json")
            lines.append(json.dumps(data, ensure_ascii=False, indent=2)[:1800])
            lines.append("```")
            lines.append("")

        render = report.get("render", {})
        if render.get("psnr") is not None:
            lines += ["## 渲染质量", "", f"- PSNR：{render['psnr']} dB（{render.get('resolution')}）",
                      f"- 高斯数量：{render.get('gaussians')}", f"- 训练耗时：{render.get('elapsed_min')} 分钟", ""]

        lines += ["## 改进建议", ""]
        for s in report.get("suggestions", []) or ["暂无"]:
            lines.append(f"- {s}")
        lines.append("")
        return "\n".join(lines)
