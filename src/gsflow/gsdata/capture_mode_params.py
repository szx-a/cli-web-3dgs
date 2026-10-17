"""拍摄模式（capture mode）的**单一映射源**：模式名 -> 具体配置键值。

用户要求：两种拍法都要支持，且**模式选择必须同步到 GUI 与 CLI**。
与 `trainers/opacity_cleanup.py` 同样的做法：映射只在这里定义一处，
CLI 直接调、GUI 通过 `/api/capture_modes` 取**同一份** —— 两端不可能漂移。

三种可选值：
  * `auto`（默认）—— 按数据自动判定（背景帧间差异）；判定结果与理由会写进日志与 job.json。
  * `turntable` —— 物体在转台上转、**相机固定、光源固定**。
      静态背景是"相机没动"的唯一证据 ⇒ **必须保留背景**参与重建
      （`mask.mode=keep`），导出阶段再按掩码/密度把物体裁出来。
  * `orbit` —— 相机绕物体行走。抠掉背景 + 黑底合成是安全的（现状）。

实测依据（`scripts/detect_turntable_capture.py`）：

| 素材 | 背景帧间差异 | 前景帧间差异 | 比值 | 判定 |
|---|---|---|---|---|
| img3-01 | **0.48** | 31.69 | **0.015** | turntable |
| maskloss-v2 | 19.68 | 39.05 | 0.504 | orbit |
| full361-v2 | 20.89 | 40.34 | 0.518 | orbit |
"""

from __future__ import annotations

from typing import Any

CHOICES = ("auto", "turntable", "orbit")

# 阈值的来历：实测两类素材相差 34 倍（0.015 vs 0.504），取中间值 0.15 足够安全。
AUTO_TURNTABLE_RATIO = 0.15


def capture_mode_params(mode: str, detected: str | None = None) -> dict[str, Any]:
    """把"模式选择"翻译成一组配置键值。

    `detected` 仅在 `mode="auto"` 时使用（自动判定后落地的实际模式）。
    """
    m = (mode or "auto").strip().lower()
    if m not in CHOICES:
        raise ValueError(f"未知的拍摄模式：{mode!r}（可选 {', '.join(CHOICES)}）")

    if m == "auto":
        # auto 本身不预设 mask 相关键；由 `resolve_capture_mode()` 判定后回填。
        # 这里只标记选择，避免"auto 却偷偷改了 mask.mode"。
        return {"capture.mode": "auto"}

    if m == "turntable":
        return {
            "capture.mode": "turntable",
            # 核心：**保留背景**，让静态背景锚定"相机没动"
            "mask.mode": "keep",
            # 转台素材的物体只占画面一小块，重建完要按掩码/密度裁出物体
            "export.mesh_density_quantile": "0.004",
        }
    # orbit：现状（抠背景 + 黑底合成）
    return {
        "capture.mode": "orbit",
        "mask.mode": "black",
    }


def resolve_capture_mode(mode: str, detected: str | None) -> tuple[str, str]:
    """返回 (实际使用的模式, 说明文字)。

    `mode="auto"` 时用 `detected`；否则以用户显式选择为准（并提示与判定不一致）。
    """
    m = (mode or "auto").strip().lower()
    if m == "auto":
        if detected in ("turntable", "orbit"):
            return detected, f"自动判定为 {detected}"
        return "orbit", "无法自动判定（缺帧或掩码），回退 orbit"
    if detected and detected != m:
        return m, f"用户指定 {m}（自动判定为 {detected}，以用户为准）"
    return m, f"用户指定 {m}"
