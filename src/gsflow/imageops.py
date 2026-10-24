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


# ------------------------------------------------------- 连通域
def drop_small_components(mask: np.ndarray, min_area_ratio: float) -> np.ndarray:
    """丢掉面积占比小于 `min_area_ratio` 的**零散小连通块**（不是丢弃整张掩码）。

    为什么需要：这是 `mask.min_area_ratio` 的本意（"小于此面积占比的零散区域丢弃"），
    但这个参数此前**从未被实现** —— 它只在配置里、在调用链里传递、写进 manifest，
    对结果没有任何影响（是个空参数）。实测分割后端确实会留下细碎噪声块。

    ⚠️ 语义边界（很容易写错）：这里删的是"块"，**不是整帧**。
    早先我一度写成"整张前景占比不足就全清空"，那会把"物体离得远/视角偏"的
    正常帧误杀 —— 那些帧的前景占比天然就低。所以判据必须是**每个连通块**
    相对于**全图面积**的比例。
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any() or min_area_ratio <= 0:
        return m
    try:
        from scipy import ndimage

        labels, n = ndimage.label(m, structure=ndimage.generate_binary_structure(2, 1))
        if n <= 1:
            return m
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        thr = float(min_area_ratio) * m.size
        keep_ids = np.nonzero(counts >= max(1.0, thr))[0]
        if keep_ids.size == 0:
            return m          # 全部都很小：宁可原样返回，也不要清空
        return np.isin(labels, keep_ids)
    except Exception:  # noqa: BLE001
        return m


def largest_component(mask: np.ndarray, min_keep_ratio: float = 0.01) -> np.ndarray:
    """只保留最大连通前景（4 邻域）。

    ⚠️ **尺寸守卫**（`min_keep_ratio`）：如果最大的连通块本身还不到全图的
    `min_keep_ratio`，说明这一帧的物体很小（离得远/视角偏），此时多块很可能
    **都是真实内容**而不是噪声 —— 强行只留一块会把物体切掉一截。
    这时宁可**原样返回**。
    实测依据：碗的数据集里 4.1% 的帧有 2~3 块（最大值 3），这些帧的对象更小；
    没有守卫的话 `keep_largest` 会把这些帧阉割。

    ⚠️ 性能：原实现是**纯 Python BFS**（逐像素 deque），在高分辨率图上不可用 ——
    IMG2 是 1920×1920、前景占 37%，单张要跑 ~136 万次 Python 循环。
    改用 scipy 稀疏连通域（C 实现），保留纯 Python 版做兜底（scipy 缺失时）。
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m
    try:
        from scipy import ndimage

        # 4 邻域结构（与原来的 (1,0)/(-1,0)/(0,1)/(0,-1) 一致）
        struct = ndimage.generate_binary_structure(2, 1)
        labels, n = ndimage.label(m, structure=struct)
        if n <= 1:
            return m
        counts = np.bincount(labels.ravel())
        counts[0] = 0                     # 0 是背景
        biggest = int(counts.max())
        if min_keep_ratio > 0 and biggest < float(min_keep_ratio) * m.size:
            return m                      # 物体太小 -> 多块都可能为真，不动它
        return labels == int(counts.argmax())
    except Exception:  # noqa: BLE001
        return _largest_component_py(m)


def _largest_component_py(mask: np.ndarray) -> np.ndarray:
    """纯 Python BFS 兜底实现（4 邻域，迭代式避免递归爆栈）。"""
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
    """把掩码写成 8 位 PNG（**容错多种 dtype**）。

    坑（2026-10-24）：旧写法 mask.astype(np.uint8) * 255 对 **0~1 浮点掩码**会先截断成 0、
    再乘 255 还是 0 ⇒ 写出**全黑掩码**。实测：img3-v2 的 812 张掩码前景占比全是 0.000，
    而日志里只写“平均前景占比 0.0%”，静默一路到 COLMAP。

    现在按 dtype 分支：
      * bool            → True/False 乘 255；
      * float（max<=1） → ×255 后取整；
      * float（max>1）  → 当作已是 0~255 灰度；
      * 整型（uint8/int）→ 直接用（已是 0~255 时）。
    """
    arr = np.asarray(mask)
    if arr.dtype == np.bool_:
        out = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.floating):
        mx = float(np.nanmax(arr)) if arr.size else 0.0
        out = np.clip(arr, 0.0, 1.0 if mx <= 1.0 else 255.0)
        out = (out * 255.0) if mx <= 1.0 else out
        out = out.astype(np.uint8)
    else:
        out = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(out, mode="L").save(path)


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
    """掩码统计。**容错 0/1、0~255、bool 三种表示**。

    坑（2026-10-24）：旧写法直接 mask.sum()/size，若传入的是 **0~255** 的 uint8，
    会算出 **10965%** 这种前景占比（实际接近 43%），日志完全不可信。
    """
    arr = np.asarray(mask)
    total = int(arr.size)
    if arr.dtype == np.bool_:
        fg = int(arr.sum())
    elif np.issubdtype(arr.dtype, np.floating) and float(arr.max() if arr.size else 0) <= 1.0:
        fg = int((arr > 0.5).sum())
    elif np.issubdtype(arr.dtype, np.integer) and (arr.size and int(arr.max()) > 1):
        fg = int((arr > 127).sum())
    else:
        fg = int((arr > 0.5).sum())
    return {"foreground_px": fg, "ratio": round(fg / max(total, 1), 5)}
