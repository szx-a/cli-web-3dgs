"""回归测试：消除"重置 vs 密度窗口"的冲突（`effective_densify_until`）。

## 为什么需要这个测试

官方 3DGS 的 `reset_every=3000` 与 `refine_stop_iter=15000` 是**一套联动机制**：
重置把不透明度打回低位并裁掉一批高斯，后续 densify 把容量补回来。

我们的 `densify_until=4000`。4000 之后开重置就变成"只删不补"：

    基线（无重置）            0.9478 / 152,578
    重置 3000（窗口 4000）    **0.9429 / 97,594（−36%）**
    重置 3000（自动延到 9000） 0.9456 / 207,384

用户要求"有冲突就改"，所以现在两个参数会被 `effective_densify_until` 联动。
这个函数是纯函数，必须钉住它的边界行为 —— 否则哪天改了数字也无人察觉。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow.trainers.gsplat_trainer import effective_densify_until as f  # noqa: E402


def test_no_reset_keeps_window() -> None:
    """没有重置时不得改动窗口（默认行为必须保持原样）。"""
    assert f(30000, 4000, 0) == 4000
    assert f(30000, 4000, 0, True) == 4000


def test_reset_extends_window_to_three_intervals() -> None:
    """重置开启且窗口偏短 ⇒ 延长到 reset×3（并受 0.6×iterations 上限约束）。"""
    assert f(30000, 4000, 3000) == 9000
    # 上限：10000 步时 0.6×10000=6000 < 9000
    assert f(10000, 4000, 3000) == 6000


def test_long_window_not_shortened() -> None:
    """窗口本来就够长时，绝不缩短（缩短会减少容量）。"""
    assert f(30000, 15000, 3000) == 15000
    assert f(30000, 20000, 3000) == 20000


def test_auto_extend_can_be_disabled() -> None:
    """显式关闭联动 ⇒ 原样返回（供复现历史行为）。"""
    assert f(30000, 4000, 3000, False) == 4000


def test_reset_larger_than_cap_never_shrinks_window() -> None:
    """极大的重置间隔也不能让窗口**缩短**（减容量与初衷相反）。

    第一版公式写成 `min(cap, max(until, need))`，于是 `until=20000 > cap=18000`
    时窗口反而被缩到 18000 —— 正是这个测试抓出来的。
    """
    assert f(1000, 4000, 100000) == 4000      # 不因 cap=600 而缩短
    assert f(30000, 20000, 3000) == 20000     # 不因 cap=18000 而缩短
