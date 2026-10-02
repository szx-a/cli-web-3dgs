"""mask 阶段回归测试：用零依赖的 classical 后端验证完整契约。

这样即使没有 FastSAM / rembg / SAM 权重，也能验证：
  掩码生成 → 形态学 → 黑背景清理图 → overlay → manifest → 失败回退策略。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow import config, imageops  # noqa: E402
from gsflow.events import EventBus  # noqa: E402
from gsflow.pipeline.context import StageContext  # noqa: E402
from gsflow.stages import MaskStage  # noqa: E402


def _sample(size: int = 256, seed: int = 5) -> tuple[Image.Image, np.ndarray]:
    """深色条纹背景 + 亮色圆盘 → 经典后端应能准确分割。"""
    rng = np.random.default_rng(seed)
    bg = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        bg[y, :, :] = 30 + (y % 30) * 2
    bg = np.clip(bg + rng.normal(0, 8, bg.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(bg)
    d = ImageDraw.Draw(img)
    gt = Image.new("L", (size, size), 0)
    gd = ImageDraw.Draw(gt)
    r = size // 4
    box = [size // 2 - r, size // 2 - r, size // 2 + r, size // 2 + r]
    d.ellipse(box, fill=(230, 100, 40), outline=(255, 255, 255), width=3)
    gd.ellipse(box, fill=255)
    return img, np.asarray(gt, dtype=np.uint8) > 127


def _ctx(job_dir: Path, backend: str, **mask_params) -> StageContext:
    settings = config.load()
    settings.set("project.workspace_dir", str(job_dir.parent))
    params = {"backend": backend, **mask_params}
    ctx = StageContext(job_id=job_dir.name, job_dir=job_dir, settings=settings,
                       events=EventBus(), params={"mask": params})
    ctx.bind_stage("mask")
    return ctx


def test_mask_stage_contract_with_classical(workdir: Path) -> None:
    job_dir = workdir / "job"
    raw = job_dir / "01_images_raw"
    raw.mkdir(parents=True)
    gts = {}
    for i in range(3):
        img, gt = _sample(seed=5 + i)
        name = f"{i:05d}.jpg"
        img.save(raw / name, quality=95)
        gts[name] = gt

    result = MaskStage().execute(_ctx(job_dir, "classical", dilate_px=2, feather_px=0))
    assert result.status == "ok"
    assert result.metrics["ok"] == 3

    mask_dir = job_dir / "02_masks"
    clean_dir = job_dir / "02_images_clean"
    manifest = json.loads((mask_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["items"]) == 3
    assert (mask_dir / "_overlay").is_dir()

    for item in manifest["items"]:
        assert (mask_dir / item["mask"]).exists()
        assert (clean_dir / item["clean"]).exists()
        assert item["ratio"] > 0.05, "前景占比过低，分割可能失效"

    # COLMAP 的掩码命名规则是 `图片全名.png`（00033.jpg → 00033.jpg.png），
    # 写错名字 COLMAP 会报 MASK_ERROR 并拒绝该帧 —— 这里做回归保护。
    for name in gts:
        assert (mask_dir / f"{name}.png").exists(), f"缺少 COLMAP 命名的掩码：{name}.png"

    # 质量：IoU 应明显高于随机
    for name, gt in gts.items():
        with Image.open(mask_dir / (Path(name).stem + ".png")) as m:
            pred = np.asarray(m.convert("L"), dtype=np.uint8) > 127
        inter = float(np.logical_and(pred, gt).sum())
        union = float(np.logical_or(pred, gt).sum()) or 1.0
        assert inter / union > 0.75, f"{name} IoU 过低：{inter/union:.3f}"

        # 黑背景：GT 之外的区域应接近全黑
        with Image.open(clean_dir / (Path(name).stem + ".jpg")) as c:
            arr = np.asarray(c.convert("RGB"))
        outside = arr[~gt]
        assert outside.mean() < 20, f"{name} 背景没有被清黑：{outside.mean():.1f}"


def test_mask_stage_falls_back_to_original_when_no_foreground(workdir: Path) -> None:
    """纯色图 → classical 找不到前景 → 默认 keep 策略应原图直通而不是产出全黑图。"""
    job_dir = workdir / "job"
    raw = job_dir / "01_images_raw"
    raw.mkdir(parents=True)
    Image.new("RGB", (128, 128), (64, 64, 64)).save(raw / "00000.jpg")

    result = MaskStage().execute(_ctx(job_dir, "classical"))
    assert result.status == "ok"
    summary = json.loads((job_dir / "02_masks" / "manifest.json").read_text(encoding="utf-8"))["summary"]
    assert summary["fallback_kept_original"] == ["00000.jpg"]
    assert summary["ok"] == 0
    # 回退帧也要有 COLMAP 命名的掩码，否则 COLMAP 会对它报 MASK_ERROR
    assert (job_dir / "02_masks" / "00000.jpg.png").exists()
    # 原图直通：清理图应仍然有内容（不是全黑）
    with Image.open(job_dir / "02_images_clean" / "00000.jpg") as im:
        assert np.asarray(im.convert("L")).mean() > 40


def test_mask_stage_skip_policy_drops_frame(workdir: Path) -> None:
    job_dir = workdir / "job"
    raw = job_dir / "01_images_raw"
    raw.mkdir(parents=True)
    Image.new("RGB", (128, 128), (64, 64, 64)).save(raw / "00000.jpg")

    from gsflow.errors import StageError

    with pytest.raises(StageError):
        MaskStage().execute(_ctx(job_dir, "classical", on_failure="skip"))


def test_mask_paths_for_returns_both_conventions(workdir: Path) -> None:
    """命名约定的单元测试：COLMAP 要 {全名}.png，我们内部用 {stem}.png。"""
    from gsflow.imageops import find_mask_path, mask_paths_for

    d = workdir / "02_masks"
    d.mkdir()
    (d / "00007.jpg.png").write_bytes(b"x")
    paths = [p.name for p in mask_paths_for(d, "00007.jpg")]
    assert paths == ["00007.jpg.png", "00007.png"]
    assert find_mask_path(d, "00007.jpg").name == "00007.jpg.png"
    (d / "00008.png").write_bytes(b"x")
    assert find_mask_path(d, "00008.jpg").name == "00008.png"
    assert find_mask_path(d, "00009.jpg") is None


def test_classical_segmenter_returns_bool_mask(workdir: Path) -> None:
    from gsflow.segment import SegOptions, create

    img, gt = _sample(size=192, seed=11)
    p = workdir / "s.png"
    img.save(p)
    seg = create("classical", SegOptions(keep_largest=True))
    masks = seg.segment([p])
    mask = masks["s.png"]
    assert mask.dtype == bool
    assert mask.shape == gt.shape
    inter = float(np.logical_and(mask, gt).sum())
    union = float(np.logical_or(mask, gt).sum()) or 1.0
    assert inter / union > 0.8
