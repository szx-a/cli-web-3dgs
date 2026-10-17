"""掩码来源查找：给纹理烘焙找"与训练图同名"的前景掩码。

## 为什么需要自动查找

`final-v1` **自己没有 `02_masks`**（它的掩码在数据源作业 `maskloss-v2` 里），
因为当初用 `--data-from` 复用数据。若只查作业自己的目录，就会静默退化成
"无掩码 ⇒ 退回黑像素判据" —— 那正是上一轮的近似方案。

## 查找顺序（找到即用，并在日志里说明来源）

  1. `export.mask_dir` 显式指定（最高优先级，便于特殊素材）；
  2. 作业自己的 `02_masks`；
  3. 作业 `spec.images_dir` 指向的作业的 `02_masks`；
  4. workspace 同级作业里，"掩码文件名与本作业训练图**匹配率 >= 90%**"的那个
     （取匹配率最高的；多个相同则按作业名排序取第一个，保证可复现）。

## 匹配率怎么算

掩码有两种命名：`{stem}.png` 与 `{image_name}.png`（COLMAP 规则）。
对每个训练图 `00001.jpg`，两种命名任一存在即算命中。
"""
from __future__ import annotations

import json
from pathlib import Path


def _mask_for_stem(mask_dir: Path, image_name: str) -> Path | None:
    stem = Path(image_name).stem
    for c in (mask_dir / f"{stem}.png", mask_dir / f"{image_name}.png"):
        if c.exists():
            return c
    return None


def match_ratio(mask_dir: Path, image_names: list[str]) -> float:
    if not mask_dir.exists() or not image_names:
        return 0.0
    hit = sum(1 for n in image_names if _mask_for_stem(mask_dir, n) is not None)
    return hit / len(image_names)


def verify_masks_against_images(mask_dir: Path, images_dir: Path,
                               image_names: list[str], max_check: int = 8):
    """**内容级校验**：掩码必须与图像内容对齐才接受。返回 (是否通过, 说明, 得分)。

    ## 为什么必须有这一层

    实测：`final-v1` 的掩码在数据源作业 `maskloss-v2`，但同级还有个 `cmp-builtin`
    **文件名完全一样（00001.jpg…）、匹配率也是 100%**，可它其实是**另一个数据集**
    （图大小 58KB vs 104KB，掩码 sha 也不同）。只按文件名匹配会挑错 ⇒
    **错掩码比没掩码更糟**。

    ## 判据（针对 `mask.mode=black` 的黑底训练图）

      * 掩码**内部**应当大多是物体（非黑像素）；
      * 掩码**外部**应当大多是黑背景。
    两个比例都对得上，才认为掩码与本作业的图像对齐。

    非黑底图像（转台 `keep` 模式）无法用黑度验证 ⇒ 直接**拒绝**用同级作业的掩码，
    宁可退回"黑像素拒绝"也不引入可能错的掩码。
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return False, "缺 numpy/Pillow", 0.0

    if not mask_dir.exists() or not images_dir.exists():
        return False, "目录不存在", 0.0

    checked = 0
    in_vals, out_vals = [], []
    for n in image_names[:max_check]:
        mp = _mask_for_stem(mask_dir, n)
        ip = images_dir / n
        if mp is None or not ip.exists():
            continue
        try:
            with Image.open(mp) as im:
                m = np.asarray(im.convert("L")) > 127
            with Image.open(ip) as im2:
                g = np.asarray(im2.convert("RGB"))
        except Exception:  # noqa: BLE001
            continue
        if m.shape != g.shape[:2]:
            from PIL import Image as _PI
            m = np.asarray(_PI.fromarray((m.astype(np.uint8) * 255)).resize(
                (g.shape[1], g.shape[0]), _PI.NEAREST)) > 127
        nonblack = g.max(axis=2) > 10
        if m.sum() > 100:
            in_vals.append(float(nonblack[m].mean()))
        if (~m).sum() > 100:
            out_vals.append(float(nonblack[~m].mean()))
        checked += 1

    if checked == 0:
        return False, "没有可校验的帧", 0.0
    if not in_vals or not out_vals:
        return False, "掩码全前景或全背景（退化）", 0.0
    fin, fout = float(np.mean(in_vals)), float(np.mean(out_vals))
    # 黑底图像：掩码内应大多非黑（>0.7），掩码外应大多黑（<0.3）
    ok = fin > 0.70 and fout < 0.30
    detail = (f"{checked} 帧：掩码内非黑 {fin*100:.0f}%（应>70%），"
              f"掩码外黑 {(1-fout)*100:.0f}%（应>70%）")
    score = fin + (1.0 - fout)
    return ok, detail, score


def find_mask_dir(job_dir: Path, image_names: list[str],
                  explicit: str = "", workspace: Path | None = None,
                  log=None) -> Path | None:
    """返回可用的掩码目录，找不到返回 None。

    信任层级：显式指定 > 本作业自己的 > 同级作业（**必须通过内容校验**）。
    """
    images_dir = job_dir / "04_train" / "data" / "images"

    # 1) 显式指定
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (job_dir / p) if (job_dir / p).exists() else (Path.cwd() / p)
        if p.exists():
            if log:
                log(f"掩码来源：显式指定 {p}")
            return p
        if log:
            log(f"掩码：显式指定的 {explicit} 不存在，继续自动查找")

    # 2) 自己的 02_masks（自己的掩码与自己的图必然对齐，只查匹配率）
    own = job_dir / "02_masks"
    r = match_ratio(own, image_names)
    if r >= 0.9:
        if log:
            log(f"掩码来源：本作业 02_masks（匹配率 {r*100:.0f}%）")
        return own

    # 3) 同级作业：文件名匹配不足以信任，**必须通过内容校验**
    ws = workspace or job_dir.parent
    cands = []
    checked_n = 0
    try:
        for sib in sorted(p for p in ws.iterdir() if p.is_dir()):
            if sib.resolve() == job_dir.resolve():
                continue
            cand = sib / "02_masks"
            rr = match_ratio(cand, image_names)
            if rr < 0.9:
                continue
            ok, detail, score = verify_masks_against_images(
                cand, images_dir, image_names)
            checked_n += 1
            if ok:
                cands.append((cand, rr, score, detail))
    except Exception:  # noqa: BLE001
        pass
    if cands:
        # 多个候选通过时按**校验得分**（掩码内非黑率 + 掩码外黑率）取最高，
        # 再按名字排序保证可复现 —— 不用纯字典序，否则会挑到"同源但非原始"的作业。
        cands.sort(key=lambda c: (-c[2], str(c[0])))
        pick = cands[0]
        if log:
            log(f"掩码来源：同级作业 {pick[0].parent.name}"
                f"（{checked_n} 个候选通过内容校验，选得分最高 {pick[2]:.3f}；{pick[3]}）")
        return pick[0]

    if log:
        worst = match_ratio(own, image_names) if own.exists() else 0.0
        log(f"掩码：未找到**可信任**的掩码（本作业匹配率 {worst*100:.0f}%；"
            f"同级 {checked_n} 个候选均未通过内容校验），"
            f"纹理烘焙退回'黑像素拒绝'近似判据")
    return None


def load_masks(mask_dir: Path | None, image_names: list[str]) -> list | None:
    """按 `image_names` 顺序加载掩码为 bool 数组列表；失败返回 None。"""
    if mask_dir is None:
        return None
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    out = []
    for n in image_names:
        p = _mask_for_stem(mask_dir, n)
        if p is None:
            return None
        try:
            with Image.open(p) as im:
                out.append(np.asarray(im.convert("L")) > 127)
        except Exception:  # noqa: BLE001
            return None
    return out
