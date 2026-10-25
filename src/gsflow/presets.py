"""训练预设的**单一映射源**：预设名 -> 一组点路径覆盖。

## 为什么单独一个模块（与 `opacity_cleanup.py` 同一模式）

预设会被 **CLI（`--preset`）** 与 **GUI（作业表单里的下拉框）** 同时使用。
如果两边各写一份键值，必然漂移；而 8 组键里错一个键名不会报错，只会"静默不生效"
（本项目已多次踩到这类静默失效，见 03/G65）。

因此：
  * 键名在 `resolve()` 里**逐条对照 `DEFAULTS` 校验**，写错立刻抛异常；
  * 值类型按 `DEFAULTS` 里同键的类型转换（CLI 传字符串也能正确落到 bool/int/float）；
  * `config.load()` 在合并完 DEFAULTS 之后、应用显式 `--set` 之前展开预设
    ⇒ **显式覆盖永远赢过预设**。

## 目前提供的预设

| 名称 | 定位 |
|---|---|
| `balanced` | 与当前默认一致（无需设置任何键，等价于不给预设）|
| `quality` | 质量优先：迭代加倍 + 官方密度控制 + AbsGS + 每步 LPIPS + 2DGS 面元正则 + 抗锯齿 |

`quality` 的代价：**约 60~75 分钟/作业**（RTX 5060 8GB、~200 视图、1600 长边）。
"""

from __future__ import annotations

from typing import Any

# ⚠️ 键名必须是 `config.DEFAULTS` 里存在的点路径；`resolve()` 会校验。
PRESETS: dict[str, dict[str, Any]] = {
    # 与默认行为一致（显式给"balanced"时不做任何覆盖）
    "balanced": {},

    # ---- 质量优先（v2，2026-10-25）----
    #
    # v1 实测结论：60 万高斯 + 抗铯齿 ⇒ 感知 LPIPS 好 3.4%，
    # 但 **op>0.9 归零**、op<0.1 占 100%（更“朆朗”），与用户诉求相反。
    # v2 因此收小容量、关抗铯齿、加强推实正则。

    #
    # 依据（均来自本仓库实测，见 docs/03）：
    #   * `densify_strategy="official"` + `absgrad=True`：复用 gsplat 官方
    #     `DefaultStrategy`（屏幕空间梯度、/count 归一化、绝对梯度）；这是社区完整实现，
    #     而不是我们此前"零散补几处"的半移植（G?/训练框架文档有记录）。
    #   * `lambda_lpips=0.1` 且 `lpips_every=1`：**必须每步都算** ——
    #     稀疏施加会让感知梯度与 L1/SSIM 不一致（G152 实测 LPIPS 反而变差）。
    #   * `rasterize_mode="antialiased"`：Mip-Splatting 低通，抑制高频噪声/闪烁。
    #   * `lambda_normal` / `lambda_depth_distortion`：2DGS 面元约束，改善网格。
    #   * `opacity_reset_interval=3000` + `auto_extend_densify_when_reset`：
    #     官方"重置后继续致密化"的联动，避免重置把密度窗口提前关死。
    # ---- 实心优先（v3 实测版，2026-10-25）----
    #
    # 实测结论：“实心度”与“画质”是**真实取舍**。
    #   quality（无 floor）：op<0.1 = 100%，但 LPIPS 0.2300（最好）
    #   solid（带 floor）：op<0.1 = 61.8%，LPIPS 0.2363（=基线 0.2370）
    # 两者都合理，取决于交付目的：要“不透”就选 solid，要“最像”就选 quality。
    "solid": {
        "train.iterations": 60000,
        "train.densify_from": 500,
        "train.densify_until": 10000,
        "train.densify_interval": 100,
        # v4 联合标定：cap 400k 比 300k 更好（PSNR 30.56 vs 29.86、SSIM 相当、
        # 实心度 op中位 0.095 vs 0.052）⇒ 改用 400k
        "train.cap_max": 400000,
        "train.densify_strategy": "builtin",
        "train.rasterize_mode": "classic",
        "train.lambda_lpips": 0.1,
        "train.lpips_every": 1,
        "train.lpips_size": 224,
        "train.lambda_normal": 0.02,
        "train.lambda_depth_distortion": 0.02,
        "train.depth_mode": "expected",
        "train.depth_distortion_mode": "tv",
        "train.sh_degree": 3,
        "train.eval_metrics": True,
        "train.opacity_crisp_reg": 0.05,
        "train.opacity_crisp_warmup": 0.4,
        # 关键：单边下限（对称正则在 op→0 时梯度趋零，实测完全无效）
        "train.opacity_floor": 0.10,
        "train.opacity_floor_reg": 0.5,
        "train.opacity_floor_warmup": 0.3,
    },

    "quality": {
        "train.iterations": 60000,
        "train.densify_from": 500,
        "train.densify_until": 10000,
        "train.cap_max": 300000,
        "train.densify_interval": 100,
        # 实测（2026-10-25）：official + grow_grad2d=2e-4 在本项目数据上致密化失控
        # （step 10000 就涨到 251 万高斯）；已补上 cap_max 但阈值仍需标定。
        # 预设先用**已校准的内置策略**，official 作为实验项（--set train.densify_strategy=official）。
        "train.densify_strategy": "builtin",
        # absgrad 只对 official 策略生效；内置策略下保留开关不起作用，写 False 避免误解
        "train.absgrad": False,
        "train.opacity_reset_interval": 3000,
        "train.auto_extend_densify_when_reset": True,
        "train.rasterize_mode": "classic",
        "train.lambda_lpips": 0.1,
        "train.lpips_every": 1,
        "train.lpips_size": 224,
        "train.lambda_normal": 0.02,
        "train.lambda_depth_distortion": 0.02,
        "train.depth_mode": "expected",
        "train.depth_distortion_mode": "tv",
        "train.sh_degree": 3,
        "train.eval_metrics": True,
        "train.opacity_crisp_reg": 0.05,
        "train.opacity_crisp_warmup": 0.4,
    },
}

ALIASES = {
    "": "balanced",
    "default": "balanced",
    "q": "quality",
    "hq": "quality",
    "solid": "solid",
    "实心": "solid",
    "高质量": "quality",
}


def names() -> list[str]:
    return sorted(PRESETS)


def resolve(name: str, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """把预设名展开成点路径覆盖字典（已做键名校验与类型转换）。

    `defaults`：传入 `config.DEFAULTS` 时会校验键名并按其类型转换；不传则原样返回。
    """
    key = ALIASES.get((name or "").strip().lower(), (name or "").strip().lower())
    if key not in PRESETS:
        raise KeyError(
            f"未知预设 {name!r}；可用：{', '.join(names())}"
        )
    raw = dict(PRESETS[key])
    if defaults is None:
        return raw

    out: dict[str, Any] = {}
    for dotted, value in raw.items():
        if not _has_path(defaults, dotted):
            raise KeyError(f"预设 {key!r} 里的键 {dotted!r} 不存在于 DEFAULTS（键名写错了）")
        ref = _get_path(defaults, dotted)
        out[dotted] = _coerce_like(value, ref)
    return out


def expand(overrides: dict[str, Any] | None, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """在用户显式覆盖**之前**插入预设值；显式键优先。

    `overrides` 里允许两种写法：`{"preset": "quality"}` 或 `{"preset": "quality", ...}`。
    """
    ov = dict(overrides or {})
    name = ov.pop("preset", None)
    if name in (None, "", "none", "None"):
        return ov
    preset = resolve(str(name), defaults)
    merged = dict(preset)
    merged.update(ov)          # 显式覆盖赢
    return merged


# ---------------------------------------------------------------- 内部工具

def _has_path(d: dict, dotted: str) -> bool:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _get_path(d: dict, dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def _coerce_like(value: Any, ref: Any) -> Any:
    """按 DEFAULTS 里的类型转换（bool 要特判：`bool("false")` 是 True）。"""
    if isinstance(ref, bool):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(ref, int) and not isinstance(ref, bool):
        return int(value)
    if isinstance(ref, float):
        return float(value)
    if isinstance(ref, str):
        return str(value)
    return value
