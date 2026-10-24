"""回归测试：掩码的 **dtype 陷阱**（三个静默失败，2026-10-24）。

## 三个坑（都表现为"日志看着正常、下游全错"）

1. `save_mask_png` 对 **0~1 浮点掩码** 先 `astype(uint8)` 截断成 0 再 ×255 ⇒ 写出**全黑掩码**；
2. 掩码腐蚀代码把 **bool**（True=1）当 0~255 判 `>127` ⇒ 全部判成 False ⇒ 掩码全黑；
3. `mask_stats` 把 **0~255 uint8** 当 0~1 统计 ⇒ 日志出现 `平均前景占比 10965%`。

后果实测：`img3-v2` 的 812 张掩码前景全是 0.000，一路静默到 COLMAP 才暴露。

这组测试不碰 GPU、不碰文件系统之外的依赖，纯 dtype 行为。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow.imageops import mask_stats, save_mask_png  # noqa: E402


def _roundtrip(tmp: Path, arr: np.ndarray) -> np.ndarray:
    from PIL import Image

    p = tmp / f"m_{arr.dtype}{arr.shape[0]}.png"
    save_mask_png(arr, p)
    with Image.open(p) as im:
        return np.asarray(im.convert("L"))


def test_save_mask_handles_bool(workdir: Path) -> None:
    a = np.zeros((6, 6), bool)
    a[1:5, 1:5] = True
    out = _roundtrip(workdir, a)
    assert out.max() == 255 and out.min() == 0
    assert abs((out > 127).mean() - a.mean()) < 1e-6


def test_save_mask_handles_float01(workdir: Path) -> None:
    """0~1 浮点必须写成 0/255，而不是被截断成全 0（坑 1）。"""
    a = np.zeros((6, 6), np.float32)
    a[2:4, 2:4] = 1.0
    out = _roundtrip(workdir, a)
    # 注意：a.mean() 是 float32（0.11111111），而比例是 float64 ⇒ 必须用容差比较，
    # 直接 == 会因精度不同而失败（这条断言自己就踩过一次）。
    assert abs((out > 127).mean() - float(a.mean())) < 1e-6, (
        f"浮点掩码被写坏了：前景占 {(out > 127).mean()}，期望 {a.mean()}")


def test_save_mask_handles_uint8(workdir: Path) -> None:
    a = (np.ones((4, 4), np.uint8) * 255)
    out = _roundtrip(workdir, a)
    assert out.min() == 255


def test_mask_stats_all_dtypes_agree() -> None:
    b = np.zeros((10, 10), bool)
    b[:6, :] = True
    r_bool = mask_stats(b)["ratio"]
    r_u8 = mask_stats((b * 255).astype(np.uint8))["ratio"]
    r_f = mask_stats(b.astype(np.float32))["ratio"]
    assert abs(r_bool - 0.6) < 1e-6
    assert abs(r_u8 - r_bool) < 1e-6, f"0~255 统计错了：{r_u8}"
    assert abs(r_f - r_bool) < 1e-6


def test_erode_dtype_branch_does_not_zero_bool() -> None:
    """复刻 mask 阶段里的腐蚀分流：bool 掩码腐蚀后必须仍有前景（坑 2）。"""
    from scipy import ndimage as nd

    base = np.zeros((20, 20), bool)
    base[5:15, 5:15] = True

    def erode_like_stage(mask, k=1):
        arr = np.asarray(mask)
        if arr.dtype == np.bool_:
            mb = arr
        elif np.issubdtype(arr.dtype, np.floating):
            mb = arr > (0.5 if float(arr.max()) <= 1.0 else 127.0)
        else:
            mb = arr > 127
        if mb.any():
            mb = nd.binary_erosion(mb, iterations=k)
        return mb

    for arr in (base, base.astype(np.float32), (base * 255).astype(np.uint8)):
        out = erode_like_stage(arr)
        assert out.any(), f"腐蚀把 {arr.dtype} 掩码清零了"
        assert out.sum() < base.sum(), "腐蚀没有变小，逻辑可疑"
