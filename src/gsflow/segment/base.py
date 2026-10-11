"""分割后端抽象层。

为什么要有这一层：SAM 3 在 Hugging Face 上是 gated 模型（需申请授权），
且本机 huggingface.co 直连不通；用户还可能在离线机器上跑。
所以流程核心只依赖 Segmenter 接口，后端可换：

    fastsam  无门槛、快、适合常见物体（默认）
    rembg    一键抠图，人像/商品最省事
    sam2     transformers 实现，网格点提示取最大前景
    sam3     概念/文本提示，需 HF 授权 + token（增强项）

新增后端只要实现 `Segmenter`，并在 BACKENDS 注册。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import GsflowError


@dataclass
class SegOptions:
    """后端无关的分割参数。"""

    prompt: str = ""                 # 文本/概念提示（SAM3、FastSAM 文本模式）
    points: list[tuple[int, int]] = field(default_factory=list)  # 归一化 0..1 点提示
    min_area_ratio: float = 0.002
    keep_largest: bool = True
    imgsz: int = 1024                # 推理分辨率
    conf: float = 0.25
    batch_size: int = 4
    extra: dict[str, Any] = field(default_factory=dict)


class Segmenter:
    """所有分割后端的基类。子类实现 _load() 与 _segment_one()。"""

    name = "base"
    requires_credentials: tuple[str, ...] = ()
    advice = ""

    def __init__(self, options: SegOptions | None = None):
        self.opt = options or SegOptions()
        self._model = None
        self._loaded = False

    # --- 懒加载：第一次调用才载权重，避免“只用 frames 阶段”也吃显存 ---
    def load(self):
        if not self._loaded:
            self._model = self._load()
            self._loaded = True
        return self._model

    def _load(self):  # pragma: no cover - 抽象
        raise NotImplementedError

    def _segment_one(self, rgb: np.ndarray) -> np.ndarray:  # pragma: no cover - 抽象
        raise NotImplementedError

    # --- 对外统一入口 ---
    def segment(self, image_paths: list[Path], on_progress=None) -> dict[str, np.ndarray]:
        """逐张分割，并在**基类统一**做后处理（连通域/小面积过滤）。

        ⚠️ 为什么后处理必须放在基类：原来只有 fastsam / sam2 / sam3 各自调了
        `largest_component`，而 **rembg 完全没有** —— 于是同一个配置
        （`mask.keep_largest=true`）在不同后端下**行为不一致**。
        实测后果：用 rembg 跑一个牌楼素材时，掩码把物体拆成主楼 + 左上 + 右下三块，
        碎块全被带进 COLMAP，干扰位姿估计（重投影误差从 0.955px 恶化到 1.212px）。
        放到基类后，"要不要只留最大连通块"对所有后端语义一致；
        子类里已有的同名调用变成幂等操作（对最大连通块再取一次最大，结果不变）。
        """
        from ..imageops import drop_small_components, largest_component, load_rgb

        self.load()
        out: dict[str, np.ndarray] = {}
        total = len(image_paths)
        for i, p in enumerate(image_paths, 1):
            try:
                rgb = load_rgb(p)
                mask = self._segment_one(rgb)
            except GsflowError:
                raise
            except Exception as exc:
                raise GsflowError(f"[{self.name}] 分割失败 {p.name}: {type(exc).__name__}: {exc}") from exc
            m = np.asarray(mask, dtype=bool)
            if m.size:
                # 顺序：先去零散小块，再取最大连通块。
                # `min_area_ratio` 此前是个**空参数**（从未实现），这里按本意补上：
                # 删的是"小连通块"，不是整帧 —— 见 imageops.drop_small_components 的说明。
                if self.opt.min_area_ratio > 0:
                    m = drop_small_components(m, self.opt.min_area_ratio)
                if self.opt.keep_largest and m.any():
                    m = largest_component(m)
            out[p.name] = m
            if on_progress:
                on_progress(i, total, p.name)
        return out


BACKENDS: dict[str, str] = {
    "classical": "gsflow.segment.classical:ClassicalSegmenter",
    "fastsam": "gsflow.segment.fastsam:FastSamSegmenter",
    "rembg": "gsflow.segment.rembg_backend:RembgSegmenter",
    "sam2": "gsflow.segment.sam2_backend:Sam2Segmenter",
    "sam3": "gsflow.segment.sam3_backend:Sam3Segmenter",
}


def create(backend: str, options: SegOptions | None = None) -> Segmenter:
    from importlib import import_module

    key = (backend or "fastsam").strip().lower()
    if key not in BACKENDS:
        raise GsflowError(f"未知分割后端 '{backend}'，可选：{', '.join(BACKENDS)}")
    mod_name, cls_name = BACKENDS[key].split(":")
    try:
        mod = import_module(mod_name)
    except ImportError as exc:
        raise GsflowError(
            f"分割后端 {key} 的依赖未安装（{exc.name}）。"
            f"请运行：pip install -e .[segment]（fastsam/rembg）或 pip install transformers（sam2/sam3）"
        ) from exc
    cls = getattr(mod, cls_name)
    return cls(options)
