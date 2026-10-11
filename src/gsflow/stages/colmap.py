"""阶段 3：COLMAP 稀疏重建（相机位姿恢复）。

输入：02_images_clean/（清理后的黑背景图，可选配合 02_masks/）
产物：
  03_colmap/database.db
  03_colmap/sparse/0/{cameras,images,points3D}.bin
  03_colmap/sparse/0/*.txt（方便排障与报告）
  03_colmap/report.json（注册率、平均重投影误差、匹配统计）

踩坑要点：
  * Windows 下 COLMAP 输出不是 UTF-8，run_tool 已强制 errors='ignore'；
  * 单场景务必关掉 multiple_models，否则会得到稀疏/0、稀疏/1 多份模型；
  * 用 mask 时只有部分 COLMAP 版本支持 --ImageReader.mask_path，
    这里用“能力探测 + 自动回退到纯黑背景图”的方式规避。
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from ..errors import StageError
from ..pipeline.base import Artifact, Stage, StageResult
from ..pipeline.context import StageContext
from ..tools import runner
from ..tools.locate import find_tool, run_tool

MODEL_DIR = "sparse/0"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ColmapStage(Stage):
    name = "colmap"
    title = "③ COLMAP 稀疏重建"
    description = "特征提取 -> 匹配 -> 稀疏重建 -> 导出，恢复相机内外参与稀疏点云"
    requires = ("frames",)
    inputs = (Artifact("01_images_raw", "原始图片（缺少清理图时回退用它）"),)
    outputs = (Artifact(f"03_colmap/{MODEL_DIR}/cameras.bin", "相机内参"),)

    # ------------------------------------------------------------------ 主流程
    def run(self, ctx: StageContext) -> StageResult:
        colmap = find_tool("colmap", str(ctx.cfg("tools.colmap", "") or ""))
        assert colmap
        base = ctx.stage_dir("03_colmap")
        work = base / "work"
        work.mkdir(exist_ok=True)
        db = work / "database.db"
        log = ctx.logs_dir() / "colmap.log"

        images_dir = self._pick_images_dir(ctx)
        mask_dir = ctx.job_dir / "02_masks"
        mask_dir_arg = self._pick_mask_dir(ctx, images_dir, mask_dir) if bool(ctx.cfg("use_masks", True)) else None

        ctx.info(f"COLMAP: {colmap.path}")
        ctx.info(f"图像目录：{images_dir}" + (f"  mask：{mask_dir_arg}" if mask_dir_arg else "（不使用 mask）"))

        # 1) 特征提取
        self._run_colmap(
            ctx, colmap.path, log,
            [
                "feature_extractor",
                "--database_path", str(db),
                "--image_path", str(images_dir),
                *(["--ImageReader.mask_path", str(mask_dir_arg)] if mask_dir_arg else []),
                "--ImageReader.camera_model", str(ctx.cfg("cameras", "SIMPLE_RADIAL")),
                "--ImageReader.single_camera", "1" if bool(ctx.cfg("single_camera", True)) else "0",
                "--SiftExtraction.max_image_size", str(int(ctx.cfg("max_image_size", 2048))),
                "--SiftExtraction.max_num_features", str(int(ctx.cfg("max_num_features", 8192))),
                "--SiftExtraction.use_gpu", "1" if bool(ctx.cfg("use_gpu", True)) else "0",
            ],
            "特征提取",
        )

        # 2) 匹配
        matcher = self._choose_matcher(ctx, images_dir)
        ctx.info(f"匹配模式：{matcher}")
        if matcher == "sequential":
            # 踩坑：开启 loop_detection 但没有 vocabulary tree 时，COLMAP 3.12 会在
            # "Generating image pairs with vocabulary tree..." 处直接崩溃
            # （Windows 原生异常 0xC0000409 / exit 3221226505，与 GPU 无关）。
            # 因此只有显式配置了 vocab_tree_path 才启用闭环检测。
            vocab_tree = str(ctx.cfg("vocab_tree_path", "") or "").strip()
            use_loop = bool(vocab_tree) and Path(vocab_tree).exists()
            if bool(vocab_tree) and not use_loop:
                ctx.warn(f"配置的 vocab_tree_path 不存在，已关闭闭环检测：{vocab_tree}")
            args = [
                "sequential_matcher",
                "--database_path", str(db),
                "--SiftMatching.use_gpu", "1" if bool(ctx.cfg("use_gpu", True)) else "0",
                "--SiftMatching.max_num_matches", str(int(ctx.cfg("max_num_matches", 32768) or 32768)),
                "--SequentialMatching.overlap", str(int(ctx.cfg("sequential_overlap", 12))),
                "--SiftMatching.guided_matching", "1" if bool(ctx.cfg("guided_matching", True)) else "0",
            ]
            if use_loop:
                args += ["--SequentialMatching.loop_detection", "1",
                         "--SequentialMatching.vocab_tree_path", vocab_tree]
                ctx.info("已启用闭环检测（vocabulary tree 存在）")
            self._run_colmap(ctx, colmap.path, log, args, "序列匹配")
        else:
            args = [
                "exhaustive_matcher" if matcher == "exhaustive" else "vocab_tree_matcher",
                "--database_path", str(db),
                "--SiftMatching.use_gpu", "1" if bool(ctx.cfg("use_gpu", True)) else "0",
                "--SiftMatching.max_num_matches", str(int(ctx.cfg("max_num_matches", 32768) or 32768)),
                "--SiftMatching.guided_matching", "1" if bool(ctx.cfg("guided_matching", True)) else "0",
            ]
            self._run_colmap(ctx, colmap.path, log, args, "全局匹配")

        # 3) 稀疏重建
        sparse = base / "sparse"
        if sparse.exists():
            shutil.rmtree(sparse, ignore_errors=True)
        sparse.mkdir(parents=True, exist_ok=True)
        self._run_colmap(
            ctx, colmap.path, log,
            [
                "mapper",
                "--database_path", str(db),
                "--image_path", str(images_dir),
                "--output_path", str(sparse),
                "--Mapper.min_num_matches", str(int(ctx.cfg("min_num_matches", 32))),
                "--Mapper.multiple_models", "1" if bool(ctx.cfg("multiple_models", False)) else "0",
                "--Mapper.ba_global_function_tolerance", "0.000001",
            ],
            "稀疏重建",
            allow_fail=True,
        )

        model = self._find_model(sparse)
        if model is None:
            raise StageError(
                self.name,
                "稀疏重建没有产生任何模型（0 个注册视图）",
                "常见原因：① 图像重叠不足（提高抽帧密度/降低 fps）② 黑背景掩码把纹理也吃掉了"
                "（试 mask.mode=inpaint 或 mask.dilate_px 调大）③ 纯色/反光/无纹理场景"
                "（增加纹理或改用 exhaustive 匹配）。可查看 logs/colmap.log。",
            )

        # 4) 导出文本 + 文本模型便于之后解析
        text_dir = base / "sparse_txt"
        text_dir.mkdir(exist_ok=True)
        self._run_colmap(
            ctx, colmap.path, log,
            ["model_converter", "--input_path", str(model), "--output_path", str(text_dir), "--output_type", "TXT"],
            "导出文本模型",
        )

        # 5) 质量体检
        report = self._quality_report(ctx, images_dir, text_dir)
        (base / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.info(
            f"注册率 {report['registered_ratio']:.1%}（{report['registered']}/{report['total_images']}），"
            f"平均重投影误差 {report['mean_reproj_error']:.3f}px，稀疏点 {report['points3D']}"
        )
        min_ratio = float(ctx.cfg("quality_min_registered_ratio", 0.75) or 0.0)
        max_err = float(ctx.cfg("quality_max_reproj_error", 2.0) or 0.0)
        if report["registered_ratio"] < min_ratio:
            ctx.warn(
                f"注册率低于阈值 {min_ratio:.0%}，后续 3DGS 质量可能受影响；"
                "建议提高抽帧密度或改用 exhaustive 匹配后重跑 colmap 阶段。"
            )
        if report["mean_reproj_error"] > max_err:
            ctx.warn(f"平均重投影误差 > {max_err}px，标定可能不稳定。")

        return StageResult(
            artifacts=[str(model), str(text_dir), str(base / "report.json")],
            metrics=report,
            message=f"注册 {report['registered']}/{report['total_images']} 张，稀疏点 {report['points3D']}",
        )

    # ------------------------------------------------------------------ 工具
    def _pick_images_dir(self, ctx: StageContext) -> Path:
        """优先用清理后的图；没有就回退到原始图（只清理不重建也能跑）。"""
        clean = ctx.job_dir / "02_images_clean"
        if clean.exists() and any(p.is_file() for p in clean.iterdir()):
            return clean
        raw = ctx.job_dir / "01_images_raw"
        if raw.exists() and any(p.is_file() for p in raw.iterdir()):
            ctx.warn("未找到清理后的图像（02_images_clean），回退用原始图像做 COLMAP；"
                     "若场景有动态物体，建议先跑 mask 阶段。")
            return raw
        raise StageError(self.name, "找不到任何可用图像", "先跑 frames / mask 阶段")

    def _run_colmap(self, ctx: StageContext, exe: str, log: Path, args: list[str], label: str,
                    allow_fail: bool = False) -> int:
        subcommand = args[0] if args else ""
        dropped: list[str] = []

        def on_drop(opt: str, _value) -> None:
            dropped.append(opt)

        # 关键：不同 COLMAP 版本选项名不同，这里按 --help 探测结果过滤，
        # 避免 "unrecognised option" 直接让阶段失败。
        safe_args = runner.filter_options(exe, subcommand, args, on_drop=on_drop)
        if dropped:
            ctx.warn(f"当前 COLMAP 不支持这些选项，已跳过：{', '.join(sorted(set(dropped)))}")

        cmd = [exe] + safe_args
        if "--help" in safe_args:
            raise StageError(self.name, "内部错误：参数中混入了 --help")
        ctx.info(f"{label} ...")

        def on_line(line: str) -> None:
            low = line.lower()
            if any(k in low for k in ("error", "failed", "warning", "registered", "elapsed")):
                ctx.emit(line, level="debug")

        code = run_tool(cmd, log_file=log, on_line=on_line)
        if code != 0 and not allow_fail:
            raise StageError(
                self.name,
                f"{label} 失败（exit={code}{'，看起来是原生崩溃' if code < 0 or code > 2**31 - 1 else ''}）",
                f"常见原因：① 未安装/版本不匹配的 vocabulary tree 却开了 loop_detection；"
                f"② 显存不足（把 colmap.use_gpu 设为 0）；③ 图像路径含中文/空格。日志：{log}",
            )
        if code != 0:
            ctx.warn(f"{label} 返回非零（exit={code}），继续尝试后续步骤（mapper 可能仍产出模型）")
        return code

    def _pick_mask_dir(self, ctx: StageContext, images_dir: Path, mask_dir: Path) -> Path | None:
        """探测当前 COLMAP 是否支持 mask_path；不支持则返回 None（用纯黑图兜底）。"""
        from ..tools import locate, runner

        info = locate.find_tool("colmap", str(ctx.cfg("tools.colmap", "") or ""), required=False)
        if info is None:
            return None
        code, out = runner.probe(info.path, ["feature_extractor", "--help"])
        supported = "mask_path" in (out or "")
        if not supported:
            ctx.warn("当前 COLMAP 不支持 ImageReader.mask_path，已自动回退为纯黑背景图（效果等价于忽略背景）")
            return None
        if not any(mask_dir.glob("*.png")):
            ctx.warn("未找到掩码文件，跳过 mask 模式")
            return None
        return mask_dir

    def _choose_matcher(self, ctx: StageContext, images_dir: Path) -> str:
        want = str(ctx.cfg("matcher", "auto")).lower()
        if want != "auto":
            return want
        names = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
        # 抽帧产物是 00001.jpg 这种连续编号 -> 序列拍摄，用 sequential
        if names and all(re.match(r"^\d{4,6}\.", n) for n in names):
            ctx.info("检测到连续编号帧，自动选择 sequential 匹配")
            return "sequential"
        ctx.info("未检测到连续编号，自动选择 exhaustive 匹配（更慢但更稳）")
        return "exhaustive"

    def _find_model(self, sparse: Path) -> Path | None:
        """在 `sparse/` 下选出**注册图像最多**的子模型。

        ⚠️ 这里曾有一个**影响很大的 bug**：原来第一行是
            if (sparse / "0").exists(): return sparse / "0"
        也就是只要 `sparse/0` 存在就无条件返回它 —— 下面那段"取注册最多"的
        评分逻辑**永远不会执行**。

        而 COLMAP 的 `mapper` 行为是：**先把第一个重建写进 `sparse/0`**，
        随后发现的其它初始种子对会写进 `sparse/1`、`sparse/2`…
        并且**最后会把最优重建另存为一个新编号**。
        所以 `sparse/0` 经常是**最小的那个**。

        实测后果（IMG2 牌楼，242 帧）：COLMAP 日志明确写着
        `Registering image #169 (num_reg_frames=241)` 与
        `Keeping successful reconstruction`，即**实际注册了 241 帧**；
        但 `sparse/0` 里只有 2 张图，我们就把这个 2 张图的模型当成结果，
        报出"注册率 0.8%"并让后续训练用垃圾位姿。
        （`Mapper.multiple_models` 关闭并不能避免多编号输出。）

        现在改为：**扫描所有子目录，按 `images.bin` 体积取最大者**
        （`images.bin` 大小与注册图像数近似成正比，且不必解析二进制）。
        """
        if not sparse.exists():
            return None
        subs = [d for d in sparse.iterdir() if d.is_dir()]
        if not subs:
            return None

        def score(d: Path) -> int:
            f = d / "images.bin"
            return f.stat().st_size if f.exists() else 0

        best = max(subs, key=score)
        if score(best) == 0:
            return None
        return best

    def _quality_report(self, ctx: StageContext, images_dir: Path, text_dir: Path) -> dict:
        """统计注册率与平均重投影误差。

        ⚠️ 踩坑（曾长期报出 meaningless 的数值）：重投影误差**不在 images.txt 里**。

        `images.txt` 每行格式是：
            IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        第 8 个字段（索引 7）是 **TY（相机平移分量）**，不是误差！
        早期代码读的正是 `parts[7]`，于是"平均重投影误差"实际是相机 TY 的平均值
        （一次实测得到 99.987px，看起来像标定崩了，其实重建误差只有 0.8px），
        还会误触发 "> 2.0px 标定可能不稳定" 的告警。

        真正的逐点重投影误差在 `points3D.txt` 的第 8 列：
            POINT3D_ID X Y Z R G B ERROR TRACK[]
        所以这里改成从 points3D.txt 统计，同时按点数加权。
        """
        total_images = sum(1 for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
        images_txt = text_dir / "images.txt"
        points_txt = text_dir / "points3D.txt"

        registered = 0
        if images_txt.exists():
            for line in images_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # 只统计"位姿行"（第二行是 POINTS2D，字段数远超 10 且第 10 列不是 .jpg/.png）
                parts = line.split()
                if len(parts) >= 10 and parts[9].lower().endswith(tuple(IMAGE_EXT)):
                    registered += 1

        # 逐点重投影误差（points3D.txt 第 8 列）
        errors: list[float] = []
        points = 0
        if points_txt.exists():
            for line in points_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                points += 1
                try:
                    errors.append(float(parts[7]))
                except ValueError:
                    pass

        mean_err = round(sum(errors) / len(errors), 4) if errors else 0.0
        return {
            "total_images": total_images,
            "registered": registered,
            "registered_ratio": round(registered / max(total_images, 1), 4),
            "mean_reproj_error": mean_err,          # 逐点平均，单位 px
            "mean_reproj_error_max": round(max(errors), 4) if errors else 0.0,
            "points3D": points,
            "model": MODEL_DIR,
        }
