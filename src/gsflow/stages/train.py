"""阶段 4：3DGS 模型训练。

输入：03_colmap/sparse/0 + 02_images_clean + 02_masks
产物：
  04_train/data/            去畸变后的训练数据（images/ + sparse/0/ + masks/）
  04_train/point_cloud/iteration_30000/point_cloud.ply   ★ 最终模型
  04_train/metrics.json     训练指标（高斯数量、PSNR 曲线、耗时）

显存守护策略（8GB 卡的现实约束）：
  1. 自动对比显存与图像规模，必要时下调 train.resolution；
  2. cap_max 限制高斯上限；
  3. OOM 时自动降一档分辨率重试一次。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..errors import StageError
from ..gsdata.colmap_io import prepare_undistorted
from ..pipeline.base import Artifact, Stage, StageResult
from ..pipeline.context import StageContext
from ..tools.locate import find_tool
from ..trainers import pick as pick_trainer

RESOLUTION_LADDER = [1600, 1400, 1200, 1000, 800, 600]


class TrainStage(Stage):
    name = "train"
    title = "④ 3DGS 训练"
    description = "用可插拔训练后端（默认 gsplat）把清理后的多视角图像拟合成 3D 高斯模型"
    requires = ("colmap",)
    inputs = (Artifact("03_colmap/sparse/0", "COLMAP 稀疏模型"),)
    outputs = (Artifact("04_train/metrics.json", "训练指标"),)

    def run(self, ctx: StageContext) -> StageResult:
        iterations = int(ctx.cfg("iterations", 30000))
        backend_name = str(ctx.cfg("backend", "gsplat"))
        out_dir = ctx.stage_dir("04_train")
        data_dir = out_dir / "data"

        # 1) 准备去畸变数据（幂等：已存在且完整则复用）
        info_file = out_dir / "undistort_info.json"
        from ..trainers.gsplat_trainer import find_model_dir

        data_ready = (find_model_dir(data_dir) / "cameras.bin").exists() and (data_dir / "images").exists()
        if not data_ready:
            colmap = find_tool("colmap", str(ctx.cfg("tools.colmap", "") or ""))
            assert colmap
            model = ctx.job_dir / "03_colmap" / "sparse" / "0"
            if not model.exists():
                subs = list((ctx.job_dir / "03_colmap" / "sparse").glob("*"))
                subs = [s for s in subs if s.is_dir()]
                if not subs:
                    raise StageError(self.name, "找不到 COLMAP 稀疏模型", "先跑 colmap 阶段")
                model = subs[0]
            ctx.info("准备去畸变训练数据（COLMAP image_undistorter）...")
            info = prepare_undistorted(
                ctx, colmap.path,
                self._pick_images_dir(ctx),
                ctx.job_dir / "02_masks",
                model,
                data_dir,
            )
            ctx.info(f"去畸变完成：{info['image_count']} 张图，掩码目录={info.get('masks_dir') or '无'}")
        else:
            ctx.info("复用已有的去畸变数据")

        # 2) 把掩码同步到去畸变图像名下，供掩码内损失使用
        self._sync_masks(ctx, data_dir, info_file)

        # 3) 训练（带 OOM 降级重试）
        backend = pick_trainer(backend_name, ctx)
        ok, why = backend.available(ctx)
        if not ok:
            raise StageError(self.name, f"训练后端 {backend_name} 不可用：{why}",
                             "运行 gsflow doctor 查看待办；或改用 train.backend=official")
        ctx.info(f"训练后端 = {backend_name}，迭代 {iterations}，分辨率上限 {ctx.cfg('resolution', 1600)}")

        resolution = int(ctx.cfg("resolution", 1600) or 0)
        ladder = [r for r in RESOLUTION_LADDER if resolution == 0 or r <= resolution] or [resolution]
        attempts = 0
        metrics: dict = {}
        while True:
            ctx.params.setdefault("train", {})
            if isinstance(ctx.params.get("train"), dict):
                ctx.params["train"]["resolution"] = resolution
            try:
                metrics = backend.run(ctx, data_dir, out_dir, iterations)
                break
            except StageError as exc:
                msg = str(exc).lower()
                oom = "out of memory" in msg or "cuda error" in msg and "memory" in msg
                attempts += 1
                if oom and attempts < len(ladder):
                    resolution = ladder[attempts]
                    ctx.warn(f"显存不足，自动降到 {resolution}px 重试（第 {attempts + 1} 次）")
                    continue
                raise

        # 4) 归一化最终的 ply 路径到 outputs 约定位置
        final_ply = Path(str(metrics.get("ply", "")))
        target_ply = out_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        if final_ply and final_ply.exists() and final_ply != target_ply:
            target_ply.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_ply, target_ply)
        if not target_ply.exists():
            raise StageError(self.name, f"训练结束但找不到模型文件：{target_ply}",
                             "查看 logs/ 下训练日志；官方后端可能把结果写到 04_train/point_cloud/ 之外")
        metrics["ply"] = str(target_ply)
        metrics["resolution_used"] = resolution
        (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return StageResult(
            artifacts=[str(target_ply), str(out_dir / "metrics.json")],
            metrics=metrics,
            message=f"{metrics.get('gaussians', '?')} 高斯，{metrics.get('elapsed_min', '?')} 分钟"
                    + (f"，PSNR {metrics['psnr']:.2f} dB" if metrics.get("psnr") else ""),
        )

    # ------------------------------------------------------------------ 辅助
    def _pick_images_dir(self, ctx: StageContext) -> Path:
        """与 colmap 阶段一致的取图逻辑：优先清理图，缺失则回退原始图。"""
        clean = ctx.job_dir / "02_images_clean"
        if clean.exists() and any(p.is_file() for p in clean.iterdir()):
            return clean
        raw = ctx.job_dir / "01_images_raw"
        if raw.exists() and any(p.is_file() for p in raw.iterdir()):
            ctx.warn("未找到清理后的图像，训练将使用原始图像（未做背景清理）")
            return raw
        raise StageError(self.name, "找不到训练用图像", "先跑 frames / mask 阶段")

    def _sync_masks(self, ctx: StageContext, data_dir: Path, info_file: Path) -> None:
        """把 02_masks 的掩码按 undistort_info 的 name_map 映射到 images/ 同名 png。"""
        from PIL import Image

        if not info_file.exists():
            return
        try:
            info = json.loads(info_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        mask_src = ctx.job_dir / "02_masks"
        if not mask_src.exists():
            return
        dst = data_dir / "masks"
        dst.mkdir(exist_ok=True)
        name_map: dict[str, str] = info.get("name_map", {})
        written = 0
        for orig, und in name_map.items():
            src = None
            for cand in (mask_src / f"{orig}.png", mask_src / (Path(orig).stem + ".png")):
                if cand.exists():
                    src = cand
                    break
            if src is None:
                continue
            img_p = data_dir / "images" / und
            if not img_p.exists():
                cands = list((data_dir / "images").glob(Path(und).stem + ".*"))
                if not cands:
                    continue
                img_p = cands[0]
            try:
                with Image.open(src) as m, Image.open(img_p) as im:
                    m = m.convert("L")
                    if m.size != im.size:
                        m = m.resize(im.size, Image.NEAREST)
                    # 两种命名都写，兼容我们自己的读取与将来的 COLMAP mask_path
                    m.save(dst / (Path(und).name + ".png"))
                    m.save(dst / (Path(und).stem + ".png"))
                    written += 1
            except Exception:
                continue
        if written:
            ctx.info(f"已同步 {written} 个掩码到训练数据（掩码内损失）")
