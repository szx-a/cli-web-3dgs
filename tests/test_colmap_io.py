"""COLMAP 数据层测试：二进制解析 + 布局兼容 + 多模型选择。

用真实 COLMAP 输出（合成数据的作业目录）验证；文件不存在时自动跳过。
这些用例覆盖了三个曾经真实踩到的坑：
  1. sparse/ 布局差异（COLMAP 3.12 输出 sparse/ 而不是 sparse/0/）
  2. 注册图像名匹配
  3. **多模型选择**：mapper 会把不同初始种子对的重建写成 sparse/0、1、2…，
     而 `sparse/0` 常常是**最小的那个** —— 无条件取 0 会丢掉最好的模型
     （实测导致 242 帧只报 2 帧注册，训练用上垃圾位姿）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow.gsdata.colmap_io import (  # noqa: E402
    Camera,
    _rot_to_quat,
    fix_up_axis_ambiguity,
    read_cameras_binary,
    read_model,
    read_points3D_binary,
)

WORKSPACE = ROOT / "workspace"


def _find_job_with_model() -> Path | None:
    if not WORKSPACE.exists():
        return None
    for job in sorted(WORKSPACE.iterdir(), reverse=True):
        for cand in (job / "03_colmap" / "sparse" / "0", job / "03_colmap" / "sparse"):
            if (cand / "cameras.bin").exists() and (cand / "images.bin").exists():
                return cand
    return None


MODEL_DIR = _find_job_with_model()
needs_model = pytest.mark.skipif(MODEL_DIR is None, reason="没有真实 COLMAP 模型可用于测试（先跑合成数据自检）")


def test_camera_intrinsics_variants() -> None:
    f = Camera(1, "SIMPLE_RADIAL", 800, 600, [500.0, 400.0, 300.0, 0.01])
    K = f.intrinsics()
    assert K[0, 0] == 500.0 and K[1, 1] == 500.0
    assert K[0, 2] == 400.0 and K[1, 2] == 300.0

    p = Camera(2, "PINHOLE", 800, 600, [500.0, 510.0, 400.0, 300.0])
    K2 = p.intrinsics()
    assert K2[0, 0] == 500.0 and K2[1, 1] == 510.0

    with pytest.raises(ValueError):
        Camera(3, "UNKNOWN_MODEL", 10, 10, []).intrinsics()


def test_quat_rotation_roundtrip() -> None:
    theta = 0.7
    r = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    w, x, y, z = _rot_to_quat(r)
    w2, x2, y2, z2 = w, x, y, z
    # 由四元数重建旋转矩阵，应与原矩阵一致
    rebuilt = np.array([
        [1 - 2 * y2 * y2 - 2 * z2 * z2, 2 * x2 * y2 - 2 * z2 * w2, 2 * x2 * z2 + 2 * y2 * w2],
        [2 * x2 * y2 + 2 * z2 * w2, 1 - 2 * x2 * x2 - 2 * z2 * z2, 2 * y2 * z2 - 2 * x2 * w2],
        [2 * x2 * z2 - 2 * y2 * w2, 2 * y2 * z2 + 2 * x2 * w2, 1 - 2 * x2 * x2 - 2 * y2 * y2],
    ])
    assert np.allclose(rebuilt, r, atol=1e-6)


def test_fix_up_axis_needs_enough_images(workdir: Path) -> None:
    # 少于 3 张时直接返回 False，不应抛异常
    assert fix_up_axis_ambiguity([]) is False


@needs_model
def test_read_real_model() -> None:
    rec = read_model(MODEL_DIR)
    assert rec.cameras, "相机为空"
    assert rec.images, "图像为空"
    assert rec.points3D.shape[1] == 3
    assert rec.colors3D.shape[1] == 3
    for im in rec.images:
        assert im.camera_id in rec.cameras
        R, t = im.cam_to_world()
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5), "旋转矩阵不是正交阵"
        assert np.isfinite(t).all()


@needs_model
def test_read_cameras_and_points() -> None:
    cams = read_cameras_binary(MODEL_DIR / "cameras.bin")
    assert cams
    for cam in cams.values():
        K = cam.intrinsics()
        assert K[0, 0] > 0 and K[1, 1] > 0
        assert cam.width > 0 and cam.height > 0

    pts, cols, errs = read_points3D_binary(MODEL_DIR / "points3D.bin")
    assert pts.shape[0] == cols.shape[0] == errs.shape[0]


def test_find_model_dir_layouts(workdir: Path) -> None:
    """find_model_dir 必须同时兼容 sparse/0 与 COLMAP 3.12 的 sparse/ 平铺布局。"""
    from gsflow.trainers.gsplat_trainer import find_model_dir

    nested = workdir / "a" / "sparse" / "0"
    nested.mkdir(parents=True)
    (nested / "cameras.bin").write_bytes(b"x")
    (nested / "images.bin").write_bytes(b"x")
    assert find_model_dir(workdir / "a") == nested

    flat = workdir / "b" / "sparse"
    flat.mkdir(parents=True)
    (flat / "cameras.bin").write_bytes(b"x")
    (flat / "images.bin").write_bytes(b"x")
    assert find_model_dir(workdir / "b") == flat

    # 什么都没有时返回一个可读的错误路径，而不是抛异常
    assert find_model_dir(workdir / "c").name in ("0", "sparse")


def test_colmap_stage_picks_largest_sparse_model(workdir: Path) -> None:
    """`ColmapStage._find_model` 必须选**注册图像最多**的子模型，而不是无条件选 `sparse/0`。
    回归测试：这里曾有一个影响很大的 bug —— 原实现第一行是
        if (sparse / "0").exists(): return sparse / "0"
    导致下面那段"取最大"的评分逻辑永远执行不到。而 COLMAP 的 mapper 会先把
    第一个重建写进 `sparse/0`，随后的种子对写进 `sparse/1`、`2`，**最优重建往往
    另存为新编号**。实测：242 帧的作业里 `sparse/0` 只有 2 张图、
    `sparse/2` 有 241 张，结果报告出"注册率 0.8%"并让训练用了垃圾位姿。

    这里用文件体积模拟（`images.bin` 大小近似正比于注册图像数，实现也正是这么评分的）。
    """
    from gsflow.stages.colmap import ColmapStage

    sparse = workdir / "sparse"
    # sparse/0 存在但**很小**；sparse/2 最大 —— 必须选 2
    sizes = {"0": 100, "1": 200, "2": 9000}
    for name, n in sizes.items():
        d = sparse / name
        d.mkdir(parents=True)
        (d / "images.bin").write_bytes(b"x" * n)
        (d / "points3D.bin").write_bytes(b"x" * n)

    picked = ColmapStage()._find_model(sparse)
    assert picked is not None
    assert picked.name == "2", f"应选最大的 sparse/2，实际选了 sparse/{picked.name}"

    # 只有一个模型时照常返回它
    only = workdir / "s1"
    (only / "0").mkdir(parents=True)
    (only / "0" / "images.bin").write_bytes(b"y" * 10)
    assert ColmapStage()._find_model(only).name == "0"

    # 空目录 / 没有任何 images.bin -> None（而不是抛异常）
    empty = workdir / "s2"
    empty.mkdir()
    assert ColmapStage()._find_model(empty) is None
    noimg = workdir / "s3"
    (noimg / "0").mkdir(parents=True)
    (noimg / "0" / "cameras.bin").write_bytes(b"z")
    assert ColmapStage()._find_model(noimg) is None
