"""rembg 后端（RMBG / U2Net 系）——一键抠图，人像与商品最省事。"""

from __future__ import annotations

import numpy as np

from ..errors import GsflowError
from .base import Segmenter


class RembgSegmenter(Segmenter):
    name = "rembg"
    advice = "pip install rembg onnxruntime（首次运行会下载模型权重）"

    def _load(self):
        from rembg import new_session

        # rembg 默认把 ONNX 权重放在 Path.home()/".rembg"，沙箱下那里不可写；
        # paths._patch_rembg_home() 已把它改到工作区 models\rembg 下。
        from .. import paths as _paths

        _paths._patch_rembg_home()

        # 注意 `or "u2net"`：配置里可能给的是**空字符串**而不是缺键，
        # 而 dict.get(key, default) 只对"键不存在"生效、对 `""` 不生效，
        # 于是空串被原样传给 rembg -> "No session class found for model ''"。
        model = (self.opt.extra.get("model") or "u2net").strip()
        try:
            return new_session(model)
        except Exception as exc:
            raise GsflowError(
                f"rembg 会话创建失败（model={model}）：{exc}。"
                f"权重目录已指向 {_paths.REMBG_HOME}；若该目录不可写或权重缺失，"
                "可改用 classical / fastsam 后端。"
            ) from exc

    def _segment_one(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image
        from rembg import remove

        session = self.load()
        out = remove(Image.fromarray(rgb), session=session)
        alpha = np.asarray(out.split()[-1], dtype=np.uint8)
        return alpha > 16
