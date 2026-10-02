"""单独验证「导出阶段」：不依赖 GPU / 训练，用真实 3DGS 格式的 PLY 走完整导出。

训练还没跑通（等 CUDA），但导出阶段本身可以独立验证：
这里直接按官方 3DGS 的 PLY 字段规范造一个模型文件，
然后走 ExportStage 产出 GLB / 查看器 / 预览图 / manifest，并做格式级校验
（GLB 头、chunk 对齐、accessor min/max、颜色合理性）。

用法：env\\venv\\Scripts\\python.exe scripts\\verify_export_stage.py
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow import paths  # noqa: E402

paths.apply_runtime_env()


def make_gaussians(n: int = 20000, seed: int = 7) -> dict:
    """造一个“有形状”的高斯集合：一个椭球壳 + 地面，便于肉眼判断预览是否正确。"""
    rng = np.random.default_rng(seed)
    n_shell = int(n * 0.7)
    n_floor = n - n_shell

    # 椭球壳
    u = rng.normal(size=(n_shell, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
    shell = u * np.array([1.0, 0.6, 1.0]) * (1.0 + rng.normal(0, 0.02, (n_shell, 1)))

    # 地面网格
    gx = rng.uniform(-1.2, 1.2, n_floor)
    gz = rng.uniform(-1.2, 1.2, n_floor)
    floor = np.stack([gx, np.full(n_floor, -1.0) + rng.normal(0, 0.01, n_floor), gz], axis=1)

    xyz = np.concatenate([shell, floor], axis=0).astype(np.float32)
    # 颜色：壳偏橙，地面偏灰绿
    rgb = np.concatenate([
        np.tile(np.array([230, 120, 60], dtype=np.float32), (n_shell, 1)),
        np.tile(np.array([90, 110, 90], dtype=np.float32), (n_floor, 1)),
    ], axis=0)
    rgb += rng.normal(0, 12, rgb.shape)
    rgb = np.clip(rgb, 0, 255)

    return {
        "xyz": xyz,
        "rgb": rgb,
        "opacity": np.clip(rng.normal(0.85, 0.05, len(xyz)), 0.1, 1.0).astype(np.float32),
        "scales": np.exp(rng.normal(-2.5, 0.3, (len(xyz), 3))).astype(np.float32),
        "quats": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (len(xyz), 1)),
    }


def write_realistic_ply(path: Path, g: dict, sh_degree: int = 3) -> Path:
    """按官方 3DGS 的字段规范写 PLY（f_dc_* + f_rest_* + opacity + scale_* + rot_*）。"""
    from plyfile import PlyData, PlyElement

    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(g["xyz"])
    sh_dim = (sh_degree + 1) ** 2
    # SH DC -> 反解回 f_dc（这样导出阶段解出来才是原色）
    SH_C0 = 0.28209479177387814
    f_dc = (g["rgb"] / 255.0 - 0.5) / SH_C0
    f_rest = np.zeros((n, (sh_dim - 1) * 3), dtype=np.float32)
    # 写 opacity 时用 logit 空间（导出阶段会 sigmoid 还原）
    op = np.log(np.clip(g["opacity"], 1e-4, 1 - 1e-4) / (1 - np.clip(g["opacity"], 1e-4, 1 - 1e-4)))
    # scale 写 log 空间
    sc = np.log(np.clip(g["scales"], 1e-8, None))

    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    cols = [g["xyz"][:, 0], g["xyz"][:, 1], g["xyz"][:, 2],
            np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)]
    for i in range(3):
        attrs.append(f"f_dc_{i}")
        cols.append(f_dc[:, i])
    for i in range(f_rest.shape[1]):
        attrs.append(f"f_rest_{i}")
        cols.append(f_rest[:, i])
    attrs.append("opacity")
    cols.append(op)
    for i in range(3):
        attrs.append(f"scale_{i}")
        cols.append(sc[:, i])
    for i in range(4):
        attrs.append(f"rot_{i}")
        cols.append(g["quats"][:, i])

    arr = np.stack(cols, axis=1).astype(np.float32)
    dtype = [(a, "f4") for a in attrs]
    el = np.empty(n, dtype=dtype)
    el[:] = list(map(tuple, arr))
    PlyData([PlyElement.describe(el, "vertex")]).write(str(path))
    return path


def check_glb(path: Path) -> dict:
    """复用产品代码里的 GLB 规范校验（避免脚本与实现两套逻辑漂移）。"""
    from gsflow.stages.export import validate_glb

    res = validate_glb(path)
    info = dict(res.get("info", {}))
    info["problems"] = res.get("problems", [])
    return info


def main() -> int:
    from gsflow.config import load as load_settings
    from gsflow.events import EventBus
    from gsflow.pipeline.context import StageContext
    from gsflow.stages import ExportStage, register_all  # noqa: F401

    root = Path(paths.ENV_DIR) / "verify-export"
    job_dir = root / "job"
    g = make_gaussians(20000)
    ply = write_realistic_ply(job_dir / "04_train" / "point_cloud" / "iteration_30000" / "point_cloud.ply", g)
    print(f"已生成合成的 3DGS PLY：{ply}  ({ply.stat().st_size/1024/1024:.2f} MB, {len(g['xyz'])} 个高斯)")

    settings = load_settings()
    settings.set("project.workspace_dir", str(root / "workspace"))
    settings.set("export.glb_max_points", 20000)
    ctx = StageContext(job_id="verify", job_dir=job_dir, settings=settings, events=EventBus(),
                       params={"export": {"glb_max_points": 20000,
                                          "formats": ["glb_points", "obj", "viewer_html"]}})
    ctx.bind_stage("export")

    result = ExportStage().execute(ctx)
    print(f"\n阶段状态：{result.status}  {result.message}  ({result.seconds:.2f}s)")

    out = job_dir / "05_export"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    print(f"产物 {len(manifest['artifacts'])} 个：")
    for a in manifest["artifacts"]:
        print(f"  {a['kind']:<9} {a['path']:<22} {a['size_mb']:>8} MB   {a['desc']}")

    problems = []
    if not (out / "model.glb").exists():
        problems.append("缺 model.glb")
    if not (out / "viewer.html").exists():
        problems.append("缺 viewer.html")
    if not list(out.glob("preview_*.png")):
        problems.append("缺预览图")

    if (out / "model.glb").exists():
        info = check_glb(out / "model.glb")
        print(f"\n=== GLB 规范校验 ===")
        print(f"  字节数        : {info['bytes']:,}")
        print(f"  chunk 数      : {info['chunks']}")
        print(f"  图元模式      : {info['mode']} (0=POINTS)")
        print(f"  顶点数        : {info['count']}")
        print(f"  材质数        : {info['materials']}")
        print(f"  extensions    : {info['extensions']}")
        problems.extend(info["problems"])

    print("\n=== 结论 ===")
    if problems:
        for p in problems:
            print(f"  ✗ {p}")
        return 1
    print("  全部通过 ✓（GLB 结构合规、查看器与预览图已生成）")
    print(f"\n产物目录：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
