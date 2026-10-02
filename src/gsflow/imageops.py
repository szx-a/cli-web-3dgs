"""纯 numpy/PIL 的掩码与图像算子。

刻意不依赖 OpenCV：本流程在最小依赖下也要能跑（COLMAP/SAM 已经够重了）。
"""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image, ImageFilter


# ------------------------------------------------------------------ 形态学
def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    m = mask.astype(np.uint8)
    for _ in range(int(radius)):
        m = _shift_or(m)
    return m.astype(bool)


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    m = mask.astype(np.uint8)
    for _ in range(int(radius)):
        m = _shift_and(m)
    return m.astype(bool)


def _shift_or(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


def _shift_and(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    out[1:, :] &= m[:-1, :]
    out[:-1, :] &= m[1:, :]
    out[:, 1:] &= m[:, :-1]
    out[:, :-1] &= m[:, 1:]
    return out


def feather(mask: np.ndarray, radius: int) -> np.ndarray:
    """边缘羽化，返回 0..1 的浮点 alpha。"""
    if radius <= 0:
        return mask.astype(np.float32)
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(img, dtype=np.float32) / 255.0


# ------------------------------------------------------- 连通域（BFS 免依赖）
def largest_component(mask: np.ndarray) -> np.ndarray:
    """只保留最大连通前景（4 邻域，迭代式避免递归爆栈）。"""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    best: list[tuple[int, int]] = []
    best_size = 0
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        q = deque([(sy, sx)])
        visited[sy, sx] = True
        comp: list[tuple[int, int]] = []
        while q:
            y, x = q.popleft()
            comp.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
        if len(comp) > best_size:
            best_size = len(comp)
            best = comp
    out = np.zeros((h, w), dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out


def drop_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """去掉面积小于 min_area 的连通域（保留全部达标域）。"""
    if min_area <= 0:
        return mask
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    out = np.zeros((h, w), dtype=bool)
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        q = deque([(sy, sx)])
        visited[sy, sx] = True
        comp: list[tuple[int, int]] = []
        while q:
            y, x = q.popleft()
            comp.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
        if len(comp) >= min_area:
            for y, x in comp:
                out[y, x] = True
    return out


# ------------------------------------------------------------------ 图像
def apply_black_background(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """背景填黑：保持原图像素，只把背景置 0（COLMAP 最稳的输入）。"""
    a = alpha[..., None].astype(np.float32)
    return np.clip(rgb.astype(np.float32) * a, 0, 255).astype(np.uint8)


def apply_alpha(rgb: np.ndarray, alpha: np.ndarray, background: int = 0) -> np.ndarray:
    """通用合成：alpha 合成到纯色背景。"""
    a = alpha[..., None].astype(np.float32)
    bg = np.full_like(rgb, background, dtype=np.float32)
    return np.clip(rgb.astype(np.float32) * a + bg * (1 - a), 0, 255).astype(np.uint8)


def save_mask_png(mask: np.ndarray, path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def mask_paths_for(mask_dir, image_name: str):
    """返回某个图像名对应的全部候选掩码路径。

    为什么会有两种命名（实测踩坑）：
      * 阶段的中间产物我们用 `{stem}.png`（00033.jpg → 00033.png）便于自己读写；
      * **COLMAP 的规则是 `{文件全名}.png`（00033.jpg → 00033.jpg.png）**，
        名字不对它会报 `MASK_ERROR: Failed to read the mask file` 并拒绝该帧。
    所以写入时两种都产出，读取时两种都尝试。
    """
    from pathlib import Path

    d = Path(mask_dir)
    stem = Path(image_name).stem
    return [d / f"{image_name}.png", d / f"{stem}.png"]


def find_mask_path(mask_dir, image_name: str):
    for p in mask_paths_for(mask_dir, image_name):
        if p.exists():
            return p
    return None


def save_overlay(rgb: np.ndarray, alpha: np.ndarray, path) -> None:
    """给 GUI 看的效果图：前景正常，背景压暗。"""
    a = alpha[..., None].astype(np.float32)
    dim = (rgb.astype(np.float32) * 0.25)
    out = np.clip(rgb.astype(np.float32) * a + dim * (1 - a), 0, 255).astype(np.uint8)
    Image.fromarray(out).save(path, quality=88)


def load_rgb(path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def mask_stats(mask: np.ndarray) -> dict:
    total = int(mask.size)
    fg = int(mask.sum())
    return {"foreground_px": fg, "ratio": round(fg / max(total, 1), 5)}
