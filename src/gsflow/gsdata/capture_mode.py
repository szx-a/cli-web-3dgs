"""`capture_mode` 检测与训练策略建议（两种拍法分开处理）。

用户明确要求支持两种模式：
  * **turntable（转台）**：物体在转台上转，相机固定、光源固定。
  * **orbit（相机绕行）**：相机绕物体走一圈。

两者为什么必须分开处理（实测数据支撑）：

| 素材 | 背景帧间差异 | 前景帧间差异 | 比值 | 判定 |
|---|---|---|---|---|
| img3-01 | **0.48** | 31.69 | **0.015** | turntable |
| maskloss-v2 | 19.68 | 39.05 | 0.504 | orbit |
| full361-v2 | 20.89 | 40.34 | 0.518 | orbit |

**turntable 模式下的致命问题**：我们把背景**抠掉了**（`mask` 阶段黑底合成），
但转台拍法里**静态背景是唯一能锚定"相机没动"的证据**。
抠掉之后，重建只能靠物体自身的"旋转"解释像素变化 ⇒
把"物体转 30°"当成"相机走 30°"⇒ 光照/遮挡不自洽 ⇒ 细节必然糊、网格必碎。

**社区对转台的通行做法**：**先重建整个场景（含背景），再提取物体**。
背景把相机锚住，物体各面的几何关系才自洽；提取物体只是后处理（按掩码裁剪/聚类）。

所以本模块给出**两种模式的训练策略**，供 `train` 阶段沿用：

  * `turntable`：
      - **不要**把背景合成成黑（`mask.train_composite=False`），让背景参与监督；
      - 用**完整图像**重建整个场景（相机固定、物体转 → 等价于相机绕物体转，
        但背景提供了正确的"相机确实在动"的约束）；
      - 导出阶段再按掩码把物体裁出来（`export.mesh_crop_*`）。
  * `orbit`：
      - 保持现状（抠背景 + 黑底合成 + 只重建物体）——
        因为相机真在动，物体各面就是不同视角看到的，抠背景不会破坏几何。

本模块只负责**判定 + 给出策略**，不做隐式行为改变（避免"配置了但没生效"）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODES = ("auto", "turntable", "orbit")


@dataclass
class CaptureAnalysis:
    mode: str                 # "turntable" | "orbit"
    bg_diff: float            # 背景区域帧间平均差异
    fg_diff: float            # 前景区域帧间平均差异
    ratio: float              # bg/fg
    n_pairs: int
    reason: str

    def as_dict(self) -> dict:
        return {
            "mode": self.mode, "bg_diff": round(self.bg_diff, 3),
            "fg_diff": round(self.fg_diff, 3), "ratio": round(self.ratio, 4),
            "n_pairs": self.n_pairs, "reason": self.reason,
        }


def detect_from_frames(job_dir: Path, max_pairs: int = 40) -> CaptureAnalysis | None:
    """从抽帧 + 掩码估计拍摄模式。

    判据：**背景区域的帧间差异**。
      * 转台（相机固定）⇒ 背景几乎不动 ⇒ 比值很小（实测 0.015）；
      * 相机绕行 ⇒ 背景有明显视差 ⇒ 比值较大（实测 0.50）。

    阈值 0.15：实测两类素材相差 30 倍以上，取中间值足够安全。
    """
    from PIL import Image

    img_dir = job_dir / "01_images_raw"
    mask_dir = job_dir / "02_masks"
    if not img_dir.exists():
        return None
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if len(files) < 4:
        return None

    def mask_of(p: Path):
        if not mask_dir.exists():
            return None
        for c in (mask_dir / (p.stem + ".png"), mask_dir / (p.name + ".png")):
            if c.exists():
                with Image.open(c) as im:
                    return np.asarray(im.convert("L")) > 127
        return None

    step = max(1, len(files) // max_pairs)
    bgs, fgs = [], []
    for i in range(0, len(files) - step, step):
        a_p, b_p = files[i], files[i + step]
        try:
            with Image.open(a_p) as im1, Image.open(b_p) as im2:
                a = np.asarray(im1.convert("L"), dtype=np.float32)
                b = np.asarray(im2.convert("L"), dtype=np.float32)
        except Exception:  # noqa: BLE001
            continue
        if a.shape != b.shape:
            h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
            a, b = a[:h, :w], b[:h, :w]
        m = mask_of(a_p)
        if m is not None and m.shape != a.shape:
            m = np.asarray(Image.fromarray((m.astype(np.uint8) * 255)).resize(
                (a.shape[1], a.shape[0]), Image.NEAREST)) > 127
        if m is None:
            # 无掩码：两帧都近黑视为背景
            bg = (a < 12) & (b < 12)
            fg = ~bg
        else:
            bg, fg = ~m, m
        d = np.abs(a - b)
        if bg.sum() > 100:
            bgs.append(float(d[bg].mean()))
        if fg.sum() > 100:
            fgs.append(float(d[fg].mean()))

    if not bgs or not fgs:
        return None
    bg_m, fg_m = float(np.mean(bgs)), float(np.mean(fgs))
    ratio = bg_m / max(fg_m, 1e-9)
    if ratio < 0.15:
        mode = "turntable"
        reason = (f"背景几乎静止（差异 {bg_m:.2f}，比值 {ratio:.3f} < 0.15）"
                  f"⇒ 转台拍法：相机固定、物体在转")
    else:
        mode = "orbit"
        reason = (f"背景随视角明显变化（差异 {bg_m:.2f}，比值 {ratio:.3f} ≥ 0.15）"
                  f"⇒ 相机绕物体行走")
    return CaptureAnalysis(mode=mode, bg_diff=bg_m, fg_diff=fg_m, ratio=ratio,
                           n_pairs=len(bgs), reason=reason)


def training_policy(mode: str) -> dict:
    """按拍摄模式给出**训练策略**（不改默认值，只给出建议与理由）。"""
    if mode == "turntable":
        return {
            "train_composite_background": False,
            "why": ("转台拍法里静态背景是'相机没动'的唯一证据。合成黑底会把背景信息抹掉，"
                    "于是重建只能靠'物体旋转'解释像素 ⇒ 把物体转当成相机转 ⇒ "
                    "光照/遮挡不自洽 ⇒ 糊、碎。应**让背景参与监督**，先重建整个场景再提取物体。"),
            "export_hint": "导出用 mesh_crop_quantile / mesh_density_quantile 按掩码裁出物体",
        }
    return {
        "train_composite_background": True,
        "why": ("相机真的在动，物体各面就是不同视角看到的，抠背景不会破坏几何；"
                "合成黑底还能抑制背景里的游离高斯。"),
        "export_hint": "保持现状",
    }
