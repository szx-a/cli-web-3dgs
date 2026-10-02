"""冒烟测试：不依赖 GPU / COLMAP / 网络，验证流程骨架可用。

运行：
    env\\venv\\Scripts\\python.exe -m pytest tests -q -p no:cacheprovider
（workdir fixture 在 conftest.py：本机受限环境下 pytest 自带的
 tmp_path/basetemp 收尾清理会 WinError 5。）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow import config, credentials, imageops  # noqa: E402
from gsflow.pipeline import registry  # noqa: E402
from gsflow.stages import register_all  # noqa: E402

register_all()


def test_stage_registry_order() -> None:
    assert registry.names() == ["frames", "mask", "colmap", "train", "export", "metrics"]
    # colmap 只硬依赖 frames：没有分割后端时也应能用原始图跑通重建
    assert registry.resolve(["train"]) == ["frames", "colmap", "train"]
    assert registry.resolve(["export"]) == ["frames", "colmap", "train", "export"]
    # 显式选择 mask 时才会带上它
    assert registry.resolve(["frames", "mask"]) == ["frames", "mask"]


def test_config_roundtrip(workdir: Path) -> None:
    path = workdir / "config.toml"
    settings = config.load()
    settings.set("colmap.matcher", "exhaustive")
    settings.set("train.iterations", 7000)
    settings.set("frames.blur_threshold", 12.5)
    settings.set("export.formats", ["glb_points", "spz"])
    settings.save(path)

    reloaded = config.load(config_path=path)
    assert reloaded.get("colmap.matcher") == "exhaustive"
    assert reloaded.get("train.iterations") == 7000
    assert reloaded.get("frames.blur_threshold") == 12.5
    assert reloaded.get("export.formats") == ["glb_points", "spz"]


def test_config_rejects_unknown_section() -> None:
    settings = config.load()
    with pytest.raises(config.ConfigError):
        settings.set("nope.key", 1)


def test_credentials_masking(workdir: Path, monkeypatch) -> None:
    monkeypatch.setattr(credentials, "CRED_FILE", workdir / "credentials.json")
    monkeypatch.setattr(credentials, "EXAMPLE_FILE", workdir / "credentials.example.json")
    monkeypatch.setattr(credentials.paths, "PROJECT_ROOT", workdir)
    credentials.save_credentials({"hf_token": "hf_abcdefgh1234", "hf_endpoint": "https://hf-mirror.com"})
    masked = credentials.mask_credentials()
    assert masked["hf_token"].endswith("1234")
    assert "abcdefgh" not in masked["hf_token"]
    assert masked["hf_endpoint"] == "https://hf-mirror.com"


def test_imageops_morphology_and_components() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    mask[15:16, 15:16] = True  # 小块噪点

    bigger = imageops.dilate(mask, 1)
    assert bigger.sum() > mask.sum()
    smaller = imageops.erode(bigger, 1)
    assert smaller.sum() == mask.sum()

    largest = imageops.largest_component(mask)
    assert largest.sum() == 25

    kept = imageops.drop_small_components(mask, min_area=5)
    assert kept.sum() == 25


def test_apply_black_background() -> None:
    rgb = np.full((4, 4, 3), 200, dtype=np.uint8)
    alpha = np.zeros((4, 4), dtype=np.float32)
    alpha[1:3, 1:3] = 1.0
    out = imageops.apply_black_background(rgb, alpha)
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[1, 1].tolist() == [200, 200, 200]


def test_orchestrator_dry_run_on_fake_images(workdir: Path) -> None:
    """用假图片跑 dry-run：应生成作业目录与 job.json，且不真正执行阶段。"""
    from gsflow.events import EventBus
    from gsflow.pipeline.orchestrator import JobSpec, Orchestrator

    src = workdir / "src"
    src.mkdir()
    for i in range(3):
        imageops.save_mask_png(np.ones((8, 8), dtype=bool), src / f"img_{i}.png")

    settings = config.load()
    settings.set("project.workspace_dir", str(workdir / "ws"))
    spec = JobSpec(images_dir=str(src), input_path=str(src), title="dry-run 测试",
                   stages=["frames", "mask"], params={"mask.backend": "rembg"})
    events = EventBus()
    collected: list[str] = []
    events.subscribe(lambda e: collected.append(e.message))

    orch = Orchestrator(spec, settings=settings, events=events, dry_run=True)
    result = orch.run()
    assert result["status"] == "done"
    assert (workdir / "ws" / orch.job_id / "job.json").exists()
    assert any("dry-run" in m for m in collected)


def test_frames_stage_imports_images(workdir: Path) -> None:
    """真实跑 frames 阶段：图片目录 -> 01_images_raw。"""
    from PIL import Image

    from gsflow.events import EventBus
    from gsflow.pipeline.context import StageContext
    from gsflow.stages import FramesStage

    src = workdir / "photos"
    src.mkdir()
    for i in range(3):
        Image.new("RGB", (64, 48), (i * 40, 100, 150)).save(src / f"p{i}.jpg")

    settings = config.load()
    settings.set("project.workspace_dir", str(workdir / "ws"))
    job_dir = workdir / "ws" / "job1"
    job_dir.mkdir(parents=True)
    ctx = StageContext(job_id="job1", job_dir=job_dir, settings=settings, events=EventBus(),
                       spec={"images_dir": str(src)})
    ctx.bind_stage("frames")
    result = FramesStage().execute(ctx)
    assert result.status == "ok"
    assert result.metrics["count"] == 3
    assert len(list((job_dir / "01_images_raw").glob("*.jpg"))) == 3


def test_export_glb_roundtrip(workdir: Path) -> None:
    """手工造一个 3DGS PLY，验证 GLB 导出与 manifest（需要 torch，无则跳过）。"""
    torch = pytest.importorskip("torch")

    from gsflow.events import EventBus
    from gsflow.pipeline.context import StageContext
    from gsflow.stages import ExportStage
    from gsflow.trainers.gsplat_trainer import save_ply

    n = 50
    means = torch.randn(n, 3)
    scales = torch.full((n, 3), -3.0)
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    opac = torch.zeros(n, 1)
    shs = torch.zeros(n, 16, 3)

    job_dir = workdir / "ws" / "job2"
    ply = job_dir / "04_train" / "point_cloud" / "iteration_100" / "point_cloud.ply"
    save_ply(ply, means, scales, quats, opac, shs, 3)

    settings = config.load()
    settings.set("project.workspace_dir", str(workdir / "ws"))
    ctx = StageContext(job_id="job2", job_dir=job_dir, settings=settings, events=EventBus(), spec={})
    ctx.bind_stage("export")
    result = ExportStage().execute(ctx)
    assert result.status == "ok"
    glb = job_dir / "05_export" / "model.glb"
    assert glb.exists() and glb.stat().st_size > 0
    manifest = json.loads((job_dir / "05_export" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["gaussians"] == n

    # GLB 必须通过 glTF 2.0 结构校验（magic/版本/长度/chunk 对齐/min-max/图元模式）
    from gsflow.stages.export import validate_glb

    check = validate_glb(glb)
    assert check["ok"], f"GLB 规范校验失败：{check['problems']}"
    assert check["info"]["mode"] == 0, "导出应为 POINTS 图元"
    assert check["info"]["count"] == n

    # 预览图是几百 KB 的 PNG：单位必须按 KB/MB 正确换算，不能出现 200+ "MB"
    previews = [a for a in manifest["artifacts"] if a["kind"] == "preview"]
    assert previews, "没有生成预览图"
    for art in previews:
        assert 0 < art["size_mb"] < 5, f"预览图大小明显不合理：{art}"
        if "size_kb" in art:
            import os

            real_kb = os.path.getsize(job_dir / "05_export" / art["path"]) / 1024
            assert abs(real_kb - art["size_kb"]) < 1, "size_kb 与实际文件不符"


if __name__ == "__main__":  # 无 pytest 时的降级入口
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
