"""SAM 3 后端（概念/文本提示分割）。

重要现实约束（务必读）：
  1. SAM 3 权重在 Hugging Face 上是 gated 模型，需要先申请授权并配置 HF Token；
     未获授权时 transformers 会下载失败，这里会抛出带修复建议的错误。
  2. transformers 对 SAM3 的支持在不同版本间接口有差异，因此实现采用
     “能力探测 + 明确报错”，而不是硬编码单一 API：
       - 若存在 Sam3Processor/Sam3Model 则优先使用；
       - 否则回退到 pipeline("mask-generation") 风格的通用接口。
  3. 本机 huggingface.co 当前不可达，请在设置页把 hf_endpoint 指向镜像。
"""

from __future__ import annotations

import numpy as np

from ..credentials import require
from ..errors import GsflowError
from ..imageops import largest_component
from .base import Segmenter

DEFAULT_CKPT = "facebook/sam3"


class Sam3Segmenter(Segmenter):
    name = "sam3"
    requires_credentials = ("hf_token",)
    advice = (
        "1) 在 Hugging Face 上申请 facebook/sam3 访问权限（gated）；"
        "2) 在设置页填 hf_token；3) 若直连不通，把 hf_endpoint 设为 https://hf-mirror.com。"
        " 在此之前可先用 fastsam / rembg / sam2 后端。"
    )

    def _load(self):
        import os

        import torch

        token = require("hf_token", self.advice)
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        from ..credentials import load_credentials

        endpoint = load_credentials().get("hf_endpoint")
        if endpoint:
            os.environ.setdefault("HF_ENDPOINT", str(endpoint))

        ckpt = self.opt.extra.get("checkpoint") or DEFAULT_CKPT
        try:
            import transformers
        except ImportError as exc:
            raise GsflowError("未安装 transformers", "pip install transformers") from exc

        bundle: dict = {"device": self.opt.extra.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"), "api": ""}
        if hasattr(transformers, "Sam3Processor") and hasattr(transformers, "Sam3Model"):
            try:
                bundle["processor"] = transformers.Sam3Processor.from_pretrained(ckpt, token=token)
                bundle["model"] = transformers.Sam3Model.from_pretrained(ckpt, token=token).to(bundle["device"]).eval()
                bundle["api"] = "sam3_native"
                return bundle
            except Exception as exc:
                raise GsflowError(
                    f"SAM3 权重加载失败（{ckpt}）：{exc}",
                    "确认已获得 gated 授权、Token 有效、镜像可达；或改用 sam2/fastsam 后端。",
                ) from exc

        # 回退：通用 mask-generation 管线（部分 transformers 版本提供）
        try:
            from transformers import pipeline

            bundle["pipe"] = pipeline(
                "mask-generation", model=ckpt, device=0 if "cuda" in bundle["device"] else -1, token=token
            )
            bundle["api"] = "mask_generation"
            return bundle
        except Exception as exc:
            raise GsflowError(
                f"当前 transformers 版本不支持 SAM3 接口，且 mask-generation 回退失败：{exc}",
                self.advice,
            ) from exc

    def _segment_one(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        bundle = self.load()
        h, w = rgb.shape[:2]
        prompt = self.opt.prompt or "object"

        if bundle.get("api") == "sam3_native":
            processor, model = bundle["processor"], bundle["model"]
            inputs = processor(images=rgb, text=prompt, return_tensors="pt").to(bundle["device"])
            with torch.no_grad():
                outputs = model(**inputs)
            masks = processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu() if "original_sizes" in inputs else None,
            )
            if isinstance(masks, (list, tuple)):
                masks = masks[0]
            arr = np.asarray(masks)
            arr = arr.reshape(-1, h, w) if arr.ndim > 2 else arr[None]
            scores = getattr(outputs, "iou_scores", None)
            scores = scores.cpu().numpy().reshape(-1) if scores is not None else np.ones(len(arr))
            idx = int(np.argmax(scores))
            mask = arr[idx] > 0.0
        else:
            pipe = bundle["pipe"]
            res = pipe(rgb, points_per_batch=32)
            masks = res.get("masks") or []
            scores = res.get("scores") or [1.0] * len(masks)
            if not masks:
                raise GsflowError("SAM3 mask-generation 未返回掩码")
            idx = int(np.argmax(np.asarray(scores).reshape(-1)))
            mask = np.asarray(masks[idx]).astype(bool)

        if mask.shape != (h, w):
            from PIL import Image

            mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) > 127
        return largest_component(mask) if self.opt.keep_largest else mask
