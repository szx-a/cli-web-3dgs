"""回归测试：交付目录不能被"单项导出对照"悄悄清空。

真实事故：用户在 `final-v1/05_export` 里只看到 3 个模型文件，
而期望是 5 个 —— 因为我用 `run_export_direct.py` 做了几次"只导 mesh"的 depth 对照，
`export.py` 每次重跑会清空整个 05_export，于是点云 glb 与带纹理 glb 被删。

本测试锁住三件事：
  1. `assert_safe_export_target` 在"已有交付物"的目录上**必须抛错**；
  2. 加 `allow_overwrite=True` 时**不抛**；
  3. 快照/还原能完整保留并恢复交付物。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _make_deliverables(d: Path, names=("model.glb", "model_mesh.ply",
                                       "model_mesh.glb", "model_mesh_textured.glb")) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        (d / n).write_bytes(b"x" * (100 + i))


def test_guard_blocks_when_deliverables_exist(workdir: Path) -> None:
    from gsflow.gsdata.export_guard import assert_safe_export_target

    d = workdir / "05_export"
    _make_deliverables(d)
    with pytest.raises(RuntimeError) as ei:
        assert_safe_export_target(d)
    msg = str(ei.value)
    assert "已有" in msg and "清空整个目录" in msg
    # 报错信息必须给出可执行的替代做法
    assert "--job" in msg and "allow-overwrite" in msg


def test_guard_allows_explicit_overwrite(workdir: Path) -> None:
    from gsflow.gsdata.export_guard import assert_safe_export_target

    d = workdir / "05_export"
    _make_deliverables(d)
    assert_safe_export_target(d, allow_overwrite=True)   # 不应抛


def test_guard_allows_empty_or_fresh_dir(workdir: Path) -> None:
    from gsflow.gsdata.export_guard import assert_safe_export_target

    d = workdir / "05_export"
    d.mkdir(parents=True)
    assert_safe_export_target(d)                          # 空目录可以写
    (d / "model.glb").write_bytes(b"y")                   # 只有 1 个，不算交付目录
    assert_safe_export_target(d)


def test_snapshot_and_restore(workdir: Path) -> None:
    from gsflow.gsdata.export_guard import restore_artifacts, snapshot_artifacts

    jobdir = workdir / "jobA"
    d = jobdir / "05_export"
    _make_deliverables(d)
    before = {p.name: p.read_bytes() for p in d.iterdir()}

    snap = snapshot_artifacts(d, root=workdir / "snaps")
    assert snap is not None and snap.exists()

    # 模拟"对照导出把目录清空"
    for p in list(d.iterdir()):
        p.unlink()
    assert not any(d.iterdir())

    n = restore_artifacts(snap, d)
    after = {p.name: p.read_bytes() for p in d.iterdir()}
    assert n == len(before)
    assert after == before, "还原后的文件内容必须与快照完全一致"


def test_export_stage_still_clears_its_own_output() -> None:
    """保护逻辑不能改变 **导出阶段本身** 的清理语义（manifest 必须与磁盘一致）。"""
    src = (ROOT / "src" / "gsflow" / "stages" / "export.py").read_text(encoding="utf-8")
    assert "每次重跑导出都清空旧产物" in src, "导出阶段的清理语义被改动了？"
    assert "old.unlink()" in src
