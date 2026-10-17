"""自动裁剪：把"裁掉空白"从手动步骤变成**默认行为**。

## 为什么必须做（本轮发现的最大缺口）

`img3-crop` 达到 **28.78 dB** 的关键是 `frames.crop = 46,646,490,316` ——
但那个数字是**我手动**用 `measure_foreground_bbox.py` 量出来、手动传进去的。

**新鲜跑一遍 IMG3 只能得到 20.74 dB**（物体只占画面 8.5%，容量浪费在空白上）。
也就是说：**好结果不是默认行为，而是依赖人工干预。** 这不可接受。

## 实现位置与原理

裁剪放在 **train 阶段、去畸变之后**，因为此时：
  * `04_train/data/images/` 已由 COLMAP `image_undistorter` 生成；
  * `04_train/data/masks/` 已同步（`_sync_masks`）；
  * ⇒ **掩码与图像已经严格对齐**，可以直接算联合包围盒。

流程：
  1. 读全部掩码，求**并集包围盒**（保证任何一帧都不被裁掉）；
  2. 若包围盒面积 / 原图面积 < `train_auto_crop_threshold`（默认 0.75），则值得裁；
  3. 裁全部图像 + 裁全部掩码（就地覆盖）；
  4. 写 `data/crop.json` 记录 `{x0,y0,w,h}`；
  5. `load_colmap_data` 读到 `crop.json` 时**平移内参主点** `cx -= x0, cy -= y0`
     —— 这是裁剪对内参的唯一影响（焦距不变）。

⚠️ **幂等保护**：`crop.json` 已存在就直接跳过。否则重复裁剪会把图越裁越小
（而且第二次的坐标相对的是已裁过的图 ⇒ 完全错位）。

## 为什么不改 COLMAP 二进制

`cameras.bin` 的相机模型里存着 width/height/cx/cy。我们**不重写它**，而是把裁剪量
单独记在 `crop.json` 里、在加载时应用。理由：
  * 二进制写回容易把模型写坏，且 COLMAP 还可能有 TEXT 格式分支；
  * 加载时平移是**可逆、可审计**的（`crop.json` 留着就能复现）；
  * 下游所有消费者都经过 `load_colmap_data`，天然一致。
"""
from __future__ import annotations

import json
from pathlib import Path


def compute_mask_bbox(masks_dir: Path) -> tuple[int, int, int, int, int, int] | None:
    """求全部掩码的**并集包围盒**。

    返回 `(x0, y0, x1, y1, W, H)`；无有效掩码返回 None。
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    if not masks_dir.exists():
        return None

    x0s, y0s, x1s, y1s = [], [], [], []
    W = H = None
    for p in sorted(masks_dir.glob("*.png")):
        try:
            with Image.open(p) as im:
                m = np.asarray(im.convert("L")) > 127
        except Exception:  # noqa: BLE001
            continue
        if W is None:
            H, W = m.shape
        if m.shape != (H, W) or not m.any():
            continue
        ys, xs = np.nonzero(m)
        x0s.append(int(xs.min())); x1s.append(int(xs.max()))
        y0s.append(int(ys.min())); y1s.append(int(ys.max()))
    if not x0s:
        return None
    return min(x0s), min(y0s), max(x1s), max(y1s), int(W), int(H)


def auto_crop_training_data(data_dir: Path, threshold: float = 0.75,
                            pad: float = 0.04, log=None) -> dict:
    """若前景过小则裁剪 `images/` 与 `masks/`，并写 `crop.json`。

    返回 `{applied: bool, ...}`；`log` 是可选日志函数。
    """
    def _log(m: str) -> None:
        if log:
            log(m)

    crop_json = data_dir / "crop.json"
    if crop_json.exists():
        try:
            d = json.loads(crop_json.read_text(encoding="utf-8"))
            _log(f"自动裁剪：已应用过（{d.get('w')}x{d.get('h')}，偏移 {d.get('x0')},{d.get('y0')}），跳过")
            return {"applied": False, "already": True, **d}
        except Exception:  # noqa: BLE001
            pass

    bbox = compute_mask_bbox(data_dir / "masks")
    if bbox is None:
        _log("自动裁剪：没有可用掩码，跳过（保持原尺寸）")
        return {"applied": False, "reason": "no_masks"}
    x0, y0, x1, y1, W, H = bbox
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    frac = (bw * bh) / max(W * H, 1)
    _log(f"自动裁剪：前景联合包围盒 {bw}x{bh}（占 {frac*100:.1f}%，"
         f"阈值 {threshold*100:.0f}%）")
    if frac >= threshold:
        _log(f"自动裁剪：前景占比足够（>={threshold*100:.0f}%），无需裁剪")
        return {"applied": False, "reason": "already_tight", "frac": round(frac, 4)}

    # 加边距，且不越界
    px = int(bw * pad) + 2
    py = int(bh * pad) + 2
    cx0 = max(0, x0 - px)
    cy0 = max(0, y0 - py)
    cx1 = min(W - 1, x1 + px)
    cy1 = min(H - 1, y1 + py)
    cw, ch = cx1 - cx0 + 1, cy1 - cy0 + 1
    if cw < 32 or ch < 32:
        _log(f"自动裁剪：裁剪后尺寸过小（{cw}x{ch}），跳过")
        return {"applied": False, "reason": "too_small"}

    try:
        from PIL import Image
    except ImportError:
        _log("自动裁剪：缺 Pillow，跳过")
        return {"applied": False, "reason": "no_pillow"}

    box = (cx0, cy0, cx0 + cw, cy0 + ch)
    n_img = n_msk = 0
    for sub, pat in (("images", "*"), ("masks", "*.png")):
        d = data_dir / sub
        if not d.exists():
            continue
        for p in sorted(d.glob(pat)):
            if not p.is_file() or p.name == "crop.json":
                continue
            try:
                with Image.open(p) as im:
                    im2 = im.crop(box)
                    im2.save(p)
                if sub == "images":
                    n_img += 1
                else:
                    n_msk += 1
            except Exception as exc:  # noqa: BLE001
                _log(f"自动裁剪：{p.name} 失败（{type(exc).__name__}: {exc}）")

    rec = {"x0": cx0, "y0": cy0, "w": cw, "h": ch,
           "orig_w": W, "orig_h": H, "fg_frac_before": round(frac, 4),
           "images_cropped": n_img, "masks_cropped": n_msk}
    crop_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"自动裁剪：完成 —— {W}x{H} → {cw}x{ch}，"
         f"裁了 {n_img} 图 / {n_msk} 掩码；前景占比 "
         f"{frac*100:.1f}% → {(bw*bh)/(cw*ch)*100:.1f}%")
    return {"applied": True, **rec}


def read_crop_shift(data_dir: Path) -> tuple[float, float] | None:
    """读 `crop.json` 的偏移，供 `load_colmap_data` 平移内参主点。"""
    p = data_dir / "crop.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return float(d.get("x0", 0)), float(d.get("y0", 0))
    except Exception:  # noqa: BLE001
        return None
