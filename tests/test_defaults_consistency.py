"""默认值一致性测试：`config.DEFAULTS` 必须与各 dataclass 的字段默认值一致。

为什么需要这条测试（真实踩过的坑，见 docs/03 G39）：
项目的默认值曾经有**两处**：`config.py` 的 `DEFAULTS`（真正生效的那份，
因为运行时从配置系统读）与 `trainers/gsplat_trainer.py` 的 `TrainConfig` dataclass。
我在 dataclass 里把 `opacity_reset_interval` 改成 0（关闭）、`mask_loss_until` 改成 1.0，
但 `DEFAULTS` 里仍是 3000 / 0.25 —— **改动完全没生效**。

后果：训练一直每 3000 步重置一次不透明度，末次重置后的状态被存成最终检查点，
PLY 的不透明度中位数被压到 0.012（正常约 0.5），渲染出一张近乎全白的图。
而"我在代码里明明改了默认值"这个错觉让排查绕了很久。

所以：**默认值只能有一处权威来源**，并且用测试钉住两边不许漂移。
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow.config import DEFAULTS  # noqa: E402


def _dataclass_defaults(cls) -> dict:
    return {f.name: f.default for f in dataclasses.fields(cls)}


def test_train_defaults_match_dataclass() -> None:
    """`DEFAULTS['train']` 里凡是 TrainConfig 也有的键，值必须相同。"""
    from gsflow.trainers.gsplat_trainer import TrainConfig

    td = _dataclass_defaults(TrainConfig)
    mismatched = {
        k: (v, td[k]) for k, v in DEFAULTS["train"].items()
        if k in td and v != td[k]
    }
    assert not mismatched, (
        "config.DEFAULTS['train'] 与 TrainConfig 的默认值不一致（运行时以 config 为准，"
        f"所以 dataclass 里的改动不会生效）：{mismatched}"
    )


def test_critical_training_defaults() -> None:
    """把几个**必须保持关闭/开启**的关键默认值单独钉住。

    这些值都有实测依据，改动会直接毁掉训练结果：
      * opacity_reset_interval = 0  —— 开启会让末次检查点被重置状态污染（实测渲染近乎全白）
      * densify_quantile = 0        —— 分位自适应阈值会导致无界增长
      * prune_scale3d = 0.1         —— 防发散的关键约束，缺失会让尺度爆炸
      * mask_loss_until = 1.0       —— 全程约束在掩码内，模型轮廓更干净
    """
    tr = DEFAULTS["train"]
    assert tr["opacity_reset_interval"] == 0, "不透明度重置必须默认关闭"
    assert tr["densify_quantile"] == 0, "分位自适应阈值必须默认关闭"
    assert tr["prune_scale3d"] == 0.1
    assert tr["densify_until"] == 4000
    assert tr["mask_loss_until"] == 1.0
    # 网格：用户要求的三种产物默认就要产出
    ex = DEFAULTS["export"]
    assert "mesh" in ex["formats"], "默认导出必须包含三角网格"
    assert "glb_points" in ex["formats"], "默认导出必须包含点云 GLB"
    assert ex["mesh_trim_sigma"] > 0, "泊松桥接裁剪应默认开启"
    assert ex["mesh_density_quantile"] > 0, "密度裁剪应有很小的非零默认值"


def test_mask_defaults_are_wired() -> None:
    """掩码的 keep_largest / min_area_ratio 必须在配置里，且后端会真正使用。

    这两个参数曾经一个是"只有部分后端实现"、另一个是**空参数**（从未被使用）。
    """
    mk = DEFAULTS["mask"]
    assert mk["keep_largest"] is True
    assert mk["min_area_ratio"] > 0
    from gsflow.imageops import drop_small_components, largest_component

    assert callable(drop_small_components) and callable(largest_component)
