"""纹理采样的两个判据：**掩码 ∧ 非黑** 取交集，且掩码来源必须经内容校验。

## 为什么要这组测试（三轮踩坑的沉淀）

1. **只用黑像素判据**（`pix.max > 6/255`）：把 `final-v1` 的黑色 texel
   从 35.99% 降到 2.30%，但会**误杀真实暗部**，且背景不黑时完全失效。

2. **只用掩码**：原理正确，但实测 `final-v1` 的黑色 texel **暴涨到 22.03%**。
   原因：掩码在**去畸变之前**生成，与 `04_train/data/images`（已去畸变）
   有几像素错位 ⇒ 被判"在物体内"的像素其实仍是黑背景 ⇒ 复制黑色 ⇒ 扩散铺开。

3. **两者取交集**（掩码负责位置、非黑负责内容）：`final-v1` **1.51%**、
   `img3-crop` **0.00%** —— 比任一单独判据都好。

另外还有一个**必须防的**陷阱：掩码来源不能只按文件名匹配。
实测 workspace 里有 40+ 个同级作业文件名完全一致（`00001.jpg…`）但其实是
**不同数据集**（图 58KB vs 104KB、掩码 sha 不同）。**错掩码比没掩码更糟**，
所以同级掩码必须通过**内容校验**（掩码内应多为非黑、掩码外应多为黑）才能用。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_mask_dir_prefers_own_and_verifies_siblings(tmp_path: Path = None) -> None:
    """本作业的掩码优先；同级掩码必须经内容校验。"""
    from gsflow.gsdata.mask_locate import find_mask_dir

    jd = ROOT / "workspace" / "img3-crop"
    if not (jd / "04_train" / "data" / "images").exists():
        pytest.skip("缺少 img3-crop")
    names = sorted(p.name for p in (jd / "04_train" / "data" / "images").iterdir())[:8]
    d = find_mask_dir(jd, names)
    assert d is not None, "img3-crop 有自己的 02_masks，必须找到"
    # `d` 形如 <workspace>/img3-crop/02_masks ⇒ d.parent 才是作业目录
    assert d.parent.name == "img3-crop", \
        f"应优先用本作业掩码，实际用了 {d}"


def test_sibling_masks_rejected_when_content_mismatched() -> None:
    """**负向测试**：内容不一致的同级掩码必须被拒绝。

    用 `final-v1` 的训练图去校验 `cmp-builtin` 的掩码（另一个数据集，
    文件名却完全一致）—— 必须判定不通过。
    """
    from gsflow.gsdata.mask_locate import verify_masks_against_images

    imgs = ROOT / "workspace" / "final-v1" / "04_train" / "data" / "images"
    bad = ROOT / "workspace" / "cmp-builtin" / "02_masks"
    if not imgs.exists() or not bad.exists():
        pytest.skip("缺少测试素材")
    names = sorted(p.name for p in imgs.iterdir())[:8]
    ok, detail, score = verify_masks_against_images(bad, imgs, names)
    assert not ok, f"错掩码被误接受！{detail}"
    assert score < 1.8, f"错掩码得分应明显偏低，实际 {score:.3f}"


def test_correct_sibling_masks_pass_verification() -> None:
    """正例：同源掩码应通过内容校验（否则判据太严会误杀）。"""
    from gsflow.gsdata.mask_locate import verify_masks_against_images

    imgs = ROOT / "workspace" / "final-v1" / "04_train" / "data" / "images"
    good = ROOT / "workspace" / "maskloss-v2" / "02_masks"
    if not imgs.exists() or not good.exists():
        pytest.skip("缺少测试素材")
    names = sorted(p.name for p in imgs.iterdir())[:8]
    ok, detail, score = verify_masks_against_images(good, imgs, names)
    assert ok, f"同源掩码应通过校验：{detail}"
    assert score > 1.8, f"同源掩码得分应偏高，实际 {score:.3f}"


def test_bake_uses_mask_AND_nonblack() -> None:
    """源码断言：采样判据必须是**掩码 ∧ 非黑**，不能只用其一。"""
    src = (ROOT / "src" / "gsflow" / "gsdata" / "mesh_export.py").read_text(
        encoding="utf-8")
    assert "masks: list | None = None" in src, "签名应接受 masks"
    assert "mask_used" in src, "统计里应记录是否用了掩码"
    # 关键：交集写法
    assert "sel = _m[vi, ui] & (pix.max(axis=1) > (6.0 / 255.0))" in src, \
        "掩码与'非黑'必须取交集（只用掩码会让 final-v1 黑到 22.03%）"


def test_rasterize_mode_defaults_to_classic() -> None:
    """抗锯齿实测**明显有害**（img3-crop：28.79 → 27.72，−1.07 dB），
    默认必须保持 classic，不能被"社区都开"误导。"""
    from gsflow.config import DEFAULTS

    assert DEFAULTS["train"]["rasterize_mode"] == "classic"
    assert DEFAULTS["train"]["absgrad"] is False, \
        "AbsGS 实测 +0.098 dB 且 95% CI [-0.114,+0.310] 不显著，默认关闭"
