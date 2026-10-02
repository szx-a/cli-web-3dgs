"""SAM 2.1 后端（transformers 实现）。

用法：给点提示（归一化坐标）或走网格点策略——在图上撒一批点，
取“置信度高、面积适中、彼此最一致”的那个掩码作为前景。

为什么默认用 SAM2 而不是 SAM3：SAM2.1 权重公开无需授权，
点/框提示对“抠出主体做 3DGS”这件事已经够用；SAM3 作为增强后端。
"""

from __future__ import annotations

import numpy as np

from ..errors import GsflowError
from ..imageops import largest_component
from .base import Segmenter

DEFAULT_CKPT = "facebook/sam2.1-hiera-large"


class Sam2Segmenter(Segmenter):
    name = "sam2"
    requires_credentials = ("hf_token",)
    advice = (
        "pip install transformers torch；"
        "若 huggingface.co 不通，请在设置页把 hf_endpoint 设为 https://hf-mirror.com 或配置代理；"
        "大模型可用 sam2.1-hiera-small 降低显存占用。"
    )

    def _load(self):
        import torch
        from transformers import Sam2Model, Sam2Processor

        ckpt = self.opt.extra.get("checkpoint") or DEFAULT_CKPT
        try:
            processor = Sam2Processor.from_pretrained(ckpt)
            model = Sam2Model.from_pretrained(ckpt)
        except Exception as exc:
            raise GsflowError(
                f"SAM2 权重加载失败（{ckpt}）：{exc}。"
                "检查网络/镜像设置与 HF Token；也可改用 fastsam / rembg 后端。"
            ) from exc
        device = self.opt.extra.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        return {"model": model, "processor": processor, "device": device, "torch": torch}

    def _grid_points(self, w: int, h: int, n: int = 4) -> list[tuple[float, float]]:
        if self.opt.points:
            return [(float(x), float(y)) for x, y in self.opt.points]
        pts: list[tuple[float, float]] = []
        for i in range(n):
            for j in range(n):
                pts.append(((j + 1) / (n + 1), (i + 1) / (n + 1)))
        return pts

    def _segment_one(self, rgb: np.ndarray) -> np.ndarray:
        bundle = self.load()
        model, processor, device, torch = bundle["model"], bundle["processor"], bundle["device"], bundle["torch"]
        h, w = rgb.shape[:2]

        prompts: list[list[list[float]]] = []
        labels: list[list[int]] = []
        for x, y in self._grid_points(w, h):
            prompts.append([[x * w, y * h]])
            labels.append([1])
        # 单点 + 中心点组合，覆盖“点在大物体上/小物体上”两种情况
        if not self.opt.points:
            prompts.append([[w * 0.5, h * 0.5]])
            labels.append([1])

        all_masks: list[np.ndarray] = []
        all_scores: list[float] = []
        batch = max(1, int(self.opt.extra.get("point_batch", 8)))
        for start in range(0, len(prompts), batch):
            in_pts = prompts[start:start + batch]
            in_lbl = labels[start:start + batch]
            inputs = processor(
                images=rgb, input_points=[in_pts], input_labels=[in_lbl], return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                out = model(**inputs)

            masks = processor.post_process_masks(
                out.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0]
            scores = out.iou_scores.cpu().numpy().reshape(-1)
            masks = masks.numpy().reshape(-1, h, w)
            for m, s in zip(masks, scores):
                all_masks.append(m > 0.0)
                all_scores.append(float(s))

        if not all_masks:
            raise GsflowError("SAM2 未返回任何掩码")

        # 选掩码：优先与“中点提示”的掩码 IoU 最高者（一致性投票）
        ref_idx = int(np.argmax(all_scores))
        ref = all_masks[ref_idx]
        best_idx, best_iou = ref_idx, -1.0
        for i, m in enumerate(all_masks):
            inter = float(np.logical_and(m, ref).sum())
            union = float(np.logical_or(m, ref).sum()) or 1.0
            iou = inter / union * 0.7 + all_scores[i] * 0.3
            if iou > best_iou:
                best_iou, best_idx = iou, i
        mask = all_masks[best_idx]
        return largest_component(mask) if self.opt.keep_largest else mask
