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

        # 1) 准备去畸变数据
        #
        # ⚠️ 这里有两个曾经真实踩到的缺陷：
        #   a) 原来硬编码 `03_colmap/sparse/0` —— 与 colmap 阶段同一个坑：
        #      COLMAP 会把最优重建写到别的编号（sparse/2），无条件取 0 会拿到垃圾位姿。
        #      现在统一走 `pick_sparse_model()`（与 colmap 阶段**共用同一份逻辑**）。
        #   b) 原来的 `data_ready` 只检查"目录/文件是否存在"，**不检查数量是否与当前
        #      COLMAP 结果一致**。实测后果：COLMAP 修好后重跑 train，去畸变数据仍是上一次
        #      失败留下的 **2 张图**，于是"复用已有的去畸变数据"，训练只用 2 张图、
        #      报出 42.8 dB 这种**过拟合 2 张图的假高分**（eval_views=0）。
        #      现在把"注册图像数"也纳入就绪判据，数量不符就重新去畸变。
        from ..stages.colmap import pick_sparse_model
        from ..trainers.gsplat_trainer import find_model_dir

        out_dir = ctx.stage_dir("04_train")
        data_dir = out_dir / "data"
        info_file = out_dir / "undistort_info.json"

        colmap_dir = ctx.job_dir / "03_colmap"
        model = pick_sparse_model(colmap_dir / "sparse")
        if model is None:
            raise StageError(self.name, "找不到 COLMAP 稀疏模型", "先跑 colmap 阶段")

        # 选中的模型里有多少张注册图像（用文本模型数，最直接）
        registered = 0
        try:
            from ..gsdata.colmap_io import read_model

            registered = len(read_model(model).images)
        except Exception:  # noqa: BLE001
            registered = 0

        have_cam = (find_model_dir(data_dir) / "cameras.bin").exists()
        have_imgs = (data_dir / "images").exists()
        n_undist = 0
        if have_imgs:
            n_undist = sum(1 for p in (data_dir / "images").iterdir() if p.is_file())
        # ⚠️ `registered == 0`（没有 COLMAP 模型可比对）时**不能**无条件认为数据可用。
        # 原写法 `registered == 0 or n_undist >= registered` 会在没有模型时直接放行，
        # 于是**任何残留的旧 04_train/data 都会被静默复用** —— 哪怕它只有 31 张图，
        # 而当前数据集有 202 张。后果是跑出一个完全不可比的"实验"（本人在做
        # `densify_until` 对照时就踩了：31 张 vs 202 张，结论全废）。
        # 现在的规则：没有模型可比对时，记录数量并**明确警告**；有模型时数量不足则重生成。
        if registered == 0:
            if n_undist > 0:
                ctx.warn(f"复用已有的去畸变数据（{n_undist} 张），但当前没有 COLMAP 模型可比对 —— "
                         f"若这个数量与预期数据集不符（例如应为 200+ 张），说明复用了残留数据，"
                         f"请换新 job-id 或删除 04_train/data 后重跑。")
            data_ready = have_cam and have_imgs and n_undist > 0
        else:
            data_ready = have_cam and have_imgs and n_undist > 0 and n_undist >= registered

        if not data_ready:
            if have_imgs and n_undist:
                ctx.warn(f"去畸变数据与当前 COLMAP 结果不匹配（已有 {n_undist} 张，"
                         f"当前模型注册 {registered} 张），重新生成")
            colmap = find_tool("colmap", str(ctx.cfg("tools.colmap", "") or ""))
            assert colmap
            ctx.info(f"准备去畸变训练数据（COLMAP image_undistorter，模型 {model.name}）...")
            info = prepare_undistorted(
                ctx, colmap.path,
                self._pick_images_dir(ctx),
                ctx.job_dir / "02_masks",
                model,
                data_dir,
            )
            ctx.info(f"去畸变完成：{info['image_count']} 张图，掩码目录={info.get('masks_dir') or '无'}")
        else:
            ctx.info(f"复用已有的去畸变数据（{n_undist} 张）")

        # 2) 把掩码同步到去畸变图像名下，供掩码内损失使用
        self._sync_masks(ctx, data_dir, info_file)

        # 2.5) **自动裁剪**：把"裁掉空白"从手动步骤变成默认行为
        #
        # 为什么必须做：`img3-crop` 的 28.78 dB 依赖我**手动**量出并传入
        # `frames.crop = 46,646,490,316`；**新鲜跑一遍只有 20.74 dB**
        # （物体仅占画面 8.5%，容量浪费在空白上）。好结果不能依赖人工干预。
        #
        # 位置选在这里是因为：`data/images` 已去畸变、`data/masks` 已同步，
        # 两者**严格对齐**，可以直接求掩码并集包围盒。裁剪量记在 `data/crop.json`，
        # 由 `load_colmap_data` 加载时平移内参主点（焦距不变）—— 不重写 COLMAP 二进制。
        if bool(ctx.cfg("auto_crop", True)):
            try:
                from ..gsdata.auto_crop import auto_crop_training_data

                cinfo = auto_crop_training_data(
                    data_dir,
                    threshold=float(ctx.cfg("auto_crop_threshold", 0.75) or 0.75),
                    log=ctx.info)
                if cinfo.get("applied"):
                    ctx.info(f"自动裁剪已生效：{cinfo['orig_w']}x{cinfo['orig_h']} → "
                             f"{cinfo['w']}x{cinfo['h']}"
                             f"（前景 {cinfo['fg_frac_before']*100:.1f}%）")
            except Exception as exc:  # noqa: BLE001
                ctx.warn(f"自动裁剪失败（{type(exc).__name__}: {exc}），按原尺寸训练")

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
