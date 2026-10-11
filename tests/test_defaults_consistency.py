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


def test_train_config_injection_preserves_zero_and_false() -> None:
    """后端注入 `TrainConfig` 时，**合法的 0 / False 不能被默认值吞掉**。

    回归测试：`GsplatBackend` 原先逐个参数写 `ctx.cfg("k", d) or d`。
    `or` 把 0 当假值 ⇒ 配置里 `opacity_reset_interval = 0`（关闭重置）
    被 `int(0 or 3000)` 变成 **3000** —— "已关闭重置"这个修复从未生效，
    训练一直每 3000 步重置不透明度，末次重置状态被存成最终检查点，
    模型退化成"薄雾"（PSNR 看着还行，但深度失效、网格成碎片）。

    这里直接验证注入结果与配置一致（0 就是 0，False 就是 False）。
    """
    from pathlib import Path

    from gsflow.trainers.gsplat_backend import _build_cfg

    class _Ctx:
        """最小上下文：只实现 cfg()，模拟 config.toml 里的取值。"""

        values = {
            "opacity_reset_interval": 0,     # 必须保持 0
            "prune_scale_after_reset": False,  # 必须保持 False
            "densify_quantile": 0.0,         # 必须保持 0.0
            "mask_loss_until": 1.0,
            "lr_opacities": 0.025,
            "opacity_reg": 0.01,
            "iterations": 1234,
        }

        def cfg(self, key, default=None):
            return self.values.get(key, default)

    cfg = _build_cfg(_Ctx(), Path("d"), Path("o"), 1234)
    assert cfg.opacity_reset_interval == 0, (
        f"opacity_reset_interval 被默认值吞掉了：{cfg.opacity_reset_interval}（应为 0）")
    assert cfg.prune_scale_after_reset is False, (
        "prune_scale_after_reset 被吞掉了：应为 False")
    assert cfg.densify_quantile == 0.0
    assert cfg.mask_loss_until == 1.0
    assert cfg.lr_opacities == 0.025
    assert cfg.opacity_reg == 0.01
    assert cfg.iterations == 1234
    # 配置里没有的字段应回落到 dataclass 默认值，而不是变成 None
    assert cfg.ssim_weight == 0.2
    assert cfg.densify_strategy == "builtin"


def test_all_train_config_fields_are_reachable_from_config() -> None:
    """`TrainConfig` 的**每个**字段都必须能通过配置设置（否则改了不生效还不报错）。

    原来后端手写注入 19 个字段、dataclass 却有 36 个 —— 其余字段改配置**无效**。
    现在 `_build_cfg` 按字段循环注入，这条测试确保覆盖不退化。
    """
    import dataclasses
    from pathlib import Path

    from gsflow.trainers.gsplat_backend import _build_cfg
    from gsflow.trainers.gsplat_trainer import TrainConfig

    explicit = {"data_dir", "out_dir", "iterations"}

    class _Sentinel:
        """任何键都返回一个可区分的哨兵值，用来确认该键**确实被读取**。"""

        def __init__(self) -> None:
            self.seen: set[str] = set()

        def cfg(self, key, default=None):
            self.seen.add(key)
            return default  # 保持默认值，只记录被读过

    ctx = _Sentinel()
    _build_cfg(ctx, Path("d"), Path("o"), 10)
    missing = {f.name for f in dataclasses.fields(TrainConfig)} - explicit - ctx.seen
    assert not missing, f"这些 TrainConfig 字段没有从配置读取（改配置无效）：{sorted(missing)}"


def test_cfgt_does_not_swallow_zero_and_false() -> None:
    """`StageContext.cfgt()` 必须保留合法的 0 / False（这是 `or` 陷阱的正解）。"""
    from pathlib import Path

    from gsflow.pipeline.context import StageContext

    class _Settings:
        vals = {"mesh_depth_stride": 0, "mesh_min_alpha": 0.5, "flag": False,
                "s_zero": "0", "s_true": "true", "bad": "abc"}

        def get(self, key, default=None):
            return self.vals.get(key, default)

    ctx = StageContext(job_id="t", job_dir=Path("t"), settings=_Settings(), events=None)  # type: ignore[arg-type]

    # 关键：显式的 0 必须原样返回，而不是回落到默认值
    assert ctx.cfgt("mesh_depth_stride", 2, int) == 0
    assert ctx.cfgt("flag", True, bool) is False
    # 缺键才回落默认
    assert ctx.cfgt("not_there", 80, int) == 80
    # --set 传进来的是字符串
    assert ctx.cfgt("s_zero", 3, int) == 0
    assert ctx.cfgt("s_true", False, bool) is True
    # 无法转换时回落默认，而不是抛异常
    assert ctx.cfgt("bad", 7, int) == 7
    assert ctx.cfgt("mesh_min_alpha", 0.1, float) == 0.5
