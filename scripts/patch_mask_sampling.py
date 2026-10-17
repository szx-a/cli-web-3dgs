"""把纹理烘焙升级为**只在前景掩码内采样**（原理正确的修法）。

## 为什么黑像素拒绝不够

上一轮用"采样像素接近纯黑就拒绝"（`mesh_texture_reject_black`），
把 `final-v1` 的黑色 texel 从 35.99% 降到 2.30%。但它只是**近似**：
  * 真实物体的**暗部**（阴影、深色石材）也会被误杀；
  * 背景若不是纯黑（`mask.mode=inpaint`、或素材本身亮背景）就完全失效。

**原理正确的做法**：用**前景掩码**判定采样点是否落在物体上 ——
这才是"物体 vs 背景"的权威依据。

## 实现要点

1. `bake_texture_from_views(..., masks=[...])`：逐个视角给一份前景掩码（与 `cams` 同序）；
   采样时 `mask[vi, ui]` 为 False 的像素直接拒绝（让别的视角来填）。
2. **掩码与黑像素拒绝并存**：有掩码时以掩码为准；无掩码时退回黑像素判据。
3. 掩码来源自动查找（`final-v1` 自己没有 `02_masks`，掩码在数据源作业 `maskloss-v2`）：
   a. `export.mask_dir` 显式指定；
   b. 作业自己的 `02_masks`；
   c. 作业 `spec.images_dir` 指向的作业的 `02_masks`；
   d. 在 workspace 同级作业里找"掩码名与本作业训练图匹配率 >= 90%"的那个。
4. 边界扩散：已有的"UV 区域全填充"步骤会在掩码边界把邻域颜色扩散进来，
   所以掩码边缘不会留下黑缝 —— 这一步已经在上一轮实现，无需再改。

用法: python scripts/patch_mask_sampling.py [--apply]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MESH = ROOT / "src" / "gsflow" / "gsdata" / "mesh_export.py"
EXPORT = ROOT / "src" / "gsflow" / "stages" / "export.py"

# ---------------- mesh_export.py：签名 + 采样判据 ----------------
MESH_EDITS = [
    ("签名加 masks",
     "                            margin_px: int = 2, max_views: int = 60,\n"
     "                            bake_reject_black: bool = True):",
     "                            margin_px: int = 2, max_views: int = 60,\n"
     "                            bake_reject_black: bool = True,\n"
     "                            masks: list | None = None):"),
    ("采样判据用掩码",
     "            if bake_reject_black:",
     "            # ⚠️ 这里的判据有优先级：**有掩码时以掩码为准**（原理正确），\n"
     "            # 无掩码时才退回\"像素接近纯黑\"的近似判据。\n"
     "            # 黑像素判据的缺陷：真实物体的暗部（阴影、深色石材）会被误杀；\n"
     "            # 背景不是纯黑时完全失效。掩码没有这两个问题。\n"
     "            _mask_arr = masks[ci] if (masks is not None and ci < len(masks)) else None\n"
     "            if _mask_arr is not None or bake_reject_black:"),
    ("掩码优先于黑像素",
     "                idx_take = np.nonzero(take)[0]\n"
     "                pix = img[vi, ui]                      # 长度 = len(idx_take)\n"
     "                sel = pix.max(axis=1) > (6.0 / 255.0)  # 非黑背景（0~1 尺度）",
     "                idx_take = np.nonzero(take)[0]\n"
     "                pix = img[vi, ui]                      # 长度 = len(idx_take)\n"
     "                if _mask_arr is not None:\n"
     "                    # 掩码可能与图像尺寸不同（去畸变/缩放），先对齐\n"
     "                    _m = _mask_arr\n"
     "                    if _m.shape[0] != img.shape[0] or _m.shape[1] != img.shape[1]:\n"
     "                        from PIL import Image as _PI\n"
     "                        _m = np.asarray(\n"
     "                            _PI.fromarray((_m.astype(np.uint8) * 255)).resize(\n"
     "                                (img.shape[1], img.shape[0]), _PI.NEAREST)) > 127\n"
     "                    sel = _m[vi, ui]\n"
     "                else:\n"
     "                    sel = pix.max(axis=1) > (6.0 / 255.0)  # 非黑背景（0~1 尺度）"),
    ("统计里区分掩码/黑像素",
     '             "samples_rejected_black": int(n_rejected)})',
     '             "samples_rejected_black": int(n_rejected),\n'
     '             "mask_used": bool(masks is not None)})'),
]

# 视角循环需要 enumerate 拿索引 ci
LOOP_EDIT = ("        for w2c, ctr, K, iw, ih, img in cams:",
             "        for ci, (w2c, ctr, K, iw, ih, img) in enumerate(cams):")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    text = MESH.read_text(encoding="utf-8")
    if "masks: list | None = None" in text and "mask_used" in text:
        print("已打过该补丁（幂等）")
        return 0

    new = text
    bad = []
    for label, old, rep in [("视角循环 enumerate", *LOOP_EDIT)] + MESH_EDITS:
        n = new.count(old)
        if n != 1:
            bad.append(f"{label}：命中 {n} 次（应为 1）")
            continue
        new = new.replace(old, rep, 1)
        print(f"  [OK] {label}")
    if bad:
        print("**未生效**：")
        for b in bad:
            print("  -", b)
        return 1
    if args.apply:
        MESH.write_text(new, encoding="utf-8")
        print(f"\n已写入 {MESH}")
    else:
        print("\n（预览，加 --apply 生效）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
