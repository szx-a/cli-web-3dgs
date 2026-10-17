"""不透明度治理方案的**单一映射源**：方案名 -> 具体配置键值。

为什么单独一个模块（而不是 CLI 与 GUI 各写一份）：
内部有 5 组开关（`opacity_reset_interval` / `opacity_decay*` /
`utilization_prune_*` / `recovery_prune_*` + 各自系数）。让用户直接填既难懂又容易漏，
而且本会话已经出现过"配置传进去了但功能从不执行"的静默失效
（嵌套函数作用域、累加器每步被清零、`_stage` 未设）。

CLI 的 `--opacity-cleanup` 与 GUI 的下拉框都调用**这里**，因此两端必然一致；
`tests/test_opacity_cleanup.py` 断言两者产出的参数集合相同。

数值参数 `param` 的语义按方案不同：
  * decay       -> 衰减系数（如 0.999）
  * utilization -> 剪枝分位（如 0.2）
  * recovery    -> 剪枝分位（如 0.2）
  * reset/none  -> 忽略
"""
from __future__ import annotations

from typing import Any

CHOICES = ("none", "reset", "decay", "utilization", "recovery")

# 各方案的默认数值（param 未显式给出时使用）
_DEFAULT_PARAM = {
    "decay": 0.999,
    "utilization": 0.2,
    "recovery": 0.2,
}

# ⚠️ 剪枝方案**必须限制总次数**（实测教训）：
# 第一版按 `interval=300/400` 周期执行 ⇒ 8000 步内剪了约 20 次，
# 每轮剪 20% ⇒ `0.8^20 ≈ 0.012`，高斯从 139,719 掉到 **7,070**，PSNR 崩到 21.04。
# 这不是"方案无效"，而是把用户方案（"**每次重置后**剪一次"）的触发条件用错了 ——
# 本项目不用重置，于是它变成了"每 400 步都剪"。
# ⇒ 现在两个剪枝方案都只保留**少数几次**，且用**更小的分位**。
_PRUNE_MAX_TIMES = 2


def opacity_cleanup_params(choice: str, param: float | None = None) -> dict[str, Any]:
    """把"方案选择"翻译成一组 `train.*` 配置键值。

    ⚠️ 只返回**该方案需要**的键；未选中的方案保持默认（即关闭）。
    这样"选了 A 就不会意外开着 B"（不做隐式联动）。
    """
    c = (choice or "none").strip().lower()
    if c not in CHOICES:
        raise ValueError(f"未知的不透明度治理方案：{choice!r}（可选 {', '.join(CHOICES)}）")
    if c == "none":
        return {"train.opacity_cleanup": "none"}
    if c == "reset":
        # 官方方案（实测最差，仅用于对照）
        return {
            "train.opacity_cleanup": "reset",
            "train.opacity_reset_interval": "3000",
            "train.pause_refine_after_reset": "200",
        }
    if c == "decay":
        v = float(param if param is not None else _DEFAULT_PARAM["decay"])
        return {
            "train.opacity_cleanup": "decay",
            "train.opacity_decay": str(v),
            "train.opacity_decay_interval": "50",
        }
    if c == "utilization":
        # 分位默认 0.05（不是 0.2）：只剪"真的几乎没被任何 tile 采样"的那一小撮，
        # 且整个训练只做 `_PRUNE_MAX_TIMES` 次。
        v = float(param if param is not None else 0.05)
        return {
            "train.opacity_cleanup": "utilization",
            "train.utilization_prune_interval": "4000",
            "train.utilization_prune_until": "4100",
            "train.utilization_prune_quantile": str(v),
        }
    # recovery：用户方案 A 的本项目版本 —— 用衰减打压，按"恢复速度"剪枝
    v = float(param if param is not None else 0.05)
    return {
        "train.opacity_cleanup": "recovery",
        "train.recovery_prune_interval": "4000",
        "train.recovery_prune_until": "4100",
        "train.recovery_prune_quantile": str(v),
        "train.recovery_window": "300",
        "train.opacity_decay": "0.999",
        "train.opacity_decay_interval": "50",
    }
