"""零依赖后端：基于背景模型 + Otsu 的经典前景分割。

定位与用途：
  * **不需要下载任何权重**，离线/无 GPU/模型下载受限时的兜底方案；
  * 也能直接用来验证 mask 阶段的完整契约（掩码 → 形态学 → 黑背景 → manifest）；
  * 适用场景：静置物体（转台/桌面小物件）、背景相对均匀、主体与背景有明显色差。

算法：
  1. 从图像四周的边框带估计背景色（中位数，抗噪声与被摄物偶发出界）；
  2. 计算每个像素到背景色的距离图（对光照变化不敏感：同时看 RGB 与环境归一化后的色度）；
  3. 用 Otsu 自动阈值二值化距离图；
  4. 形态学开运算去噪 + 保留最大连通域（由上层 keep_largest 控制）。
"""

from __future__ import annotations

import numpy as np

from ..imageops import dilate, erode
from .base import Segmenter


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu 阈值：最大化类间方差。values 为 0..1 的浮点数组。"""
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (edges[:-1] + edges[1:]) / 2.0)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    idx = int(np.argmax(sigma_b))
    return float((edges[idx] + edges[idx + 1]) / 2.0)


class ClassicalSegmenter(Segmenter):
    name = "classical"
    advice = "无需安装任何依赖；适合静置物体 + 相对均匀背景。效果不佳时改用 rembg/fastsam。"

    def _load(self):
        return {"ready": True}

    def _segment_one(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        img = rgb.astype(np.float32)

        # 1) 边框带估计背景色（取外侧 6% 宽度）
        band = max(3, int(min(h, w) * 0.06))
        border = np.concatenate(
            [
                img[:band, :, :].reshape(-1, 3),
                img[-band:, :, :].reshape(-1, 3),
                img[:, :band, :].reshape(-1, 3),
                img[:, -band:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(border, axis=0)

        # 2) 距离图：RGB 欧氏距离 + 色度（归一化颜色）距离，兼顾亮暗与色偏
        diff_rgb = np.linalg.norm(img - bg[None, None, :], axis=2) / 441.7  # 归一化到 0..1

        def chroma(x: np.ndarray) -> np.ndarray:
            s = x.sum(axis=2, keepdims=True) + 1e-6
            return x / s

        diff_chroma = np.linalg.norm(chroma(img) - chroma(bg[None, None, :]), axis=2) / 1.732

        dist = 0.65 * diff_rgb + 0.35 * diff_chroma

        # 3) Otsu 自动阈值
        thr = _otsu_threshold(dist.reshape(-1))
        mask = dist > max(thr, 0.06)

        # 4) 开运算去噪（先腐蚀后膨胀），去掉细碎噪点
        mask = dilate(erode(mask, 1), 1)
        # 没找到前景时返回空掩码（而不是抛异常）：交给 mask 阶段的
        # mask.on_failure 策略决定是“原图直通”还是“丢弃该帧”。
        return mask
