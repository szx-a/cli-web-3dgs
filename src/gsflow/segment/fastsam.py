"""FastSAM 后端（ultralytics）——默认后端：无授权门槛、速度可接受。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..errors import GsflowError
from ..imageops import largest_component
from .base import Segmenter, SegOptions


class FastSamSegmenter(Segmenter):
    name = "fastsam"
    advice = (
        "pip install ultralytics；权重用 `scripts\\download_weights.py fastsam` 下载到 models\\ 下，"
        "本机的 curl 走 Schannel 取不到证书，ultralytics 自带的下载会失败。"
    )

    def _resolve_weights(self) -> str:
        """优先用工作区里已有的权重，避免触发 ultralytics 的 curl 下载。"""
        from .. import paths

        configured = str(self.opt.extra.get("weights") or "").strip()
        if configured and Path(configured).exists():
            return configured
        wanted = configured or "FastSAM-s.pt"
        name = Path(wanted).name
        for base in (paths.MODELS_DIR, paths.TOOLS_DIR, paths.PROJECT_ROOT, Path.cwd()):
            for cand in (base / name, base / "weights" / name):
                if cand.exists():
                    return str(cand)
        return wanted

    def _load(self):
        from ultralytics import FastSAM  # 延迟导入

        model = self._resolve_weights()
        try:
            return FastSAM(model)
        except Exception as exc:
            raise GsflowError(
                f"FastSAM 权重加载失败（{model}）：{exc}。"
                "本机 curl 走 Schannel 无法下载，请先运行 "
                "`env\\venv\\Scripts\\python.exe scripts\\download_weights.py fastsam`，"
                "或改用 rembg / classical 后端。"
            ) from exc

    def _segment_one(self, rgb: np.ndarray) -> np.ndarray:
        model = self.load()
        device = self.opt.extra.get("device", "0")
        imgsz = int(self.opt.imgsz)
        conf = float(self.opt.conf)

        def call(prompt: str | None):
            kwargs = dict(device=device, retina_masks=True, imgsz=imgsz, conf=conf, verbose=False)
            if prompt:
                kwargs["texts"] = prompt          # ultralytics 的正确参数名是 texts
            return model(rgb, **kwargs)

        res = None
        if self.opt.prompt:
            # 文本提示需要 CLIP（本机常因 GitHub/Schannel 拉不下来而失败），
            # 失败就退回自动分割，不要让整张图失败。
            try:
                results = call(self.opt.prompt)
                res = results[0] if results else None
            except Exception:
                res = None

        def has_masks(r) -> bool:
            m = getattr(r, "masks", None)
            return m is not None and getattr(m, "data", None) is not None and len(m.data) > 0

        if res is None or not has_masks(res):
            results = call(None)
            res = results[0] if results else None
        if res is None or not has_masks(res):
            raise GsflowError("FastSAM 未检出任何前景，试试降低 conf 或更换后端")

        masks = res.masks
        data = masks.data.cpu().numpy()
        if data.ndim == 2:
            data = data[None]
        h, w = rgb.shape[:2]
        areas = [float(m.sum()) for m in data]
        order = np.argsort(areas)[::-1]
        if self.opt.keep_largest:
            merged = data[order[0]] > 0.5
        else:
            merged = np.zeros((h, w), dtype=bool)
            for idx in order:
                merged |= data[idx] > 0.5
        if merged.shape != (h, w):
            from PIL import Image

            merged = np.asarray(
                Image.fromarray((merged * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
            ) > 127
        return largest_component(merged) if self.opt.keep_largest else merged
