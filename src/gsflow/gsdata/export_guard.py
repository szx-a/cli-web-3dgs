"""给"单项导出对照"加**交付物保护**：不要在交付目录上做实验。

真实事故（用户发现）：`final-v1/05_export` 里原本有全部 5 个产物，
我用 `run_export_direct.py` 做了几次"只导 mesh"的 depth 对照，
之后目录里只剩 3 个模型文件 —— `model.glb`（点云 glb）与
`model_mesh_textured.glb`（带纹理）**被删了**。

根因在 `stages/export.py`：
    # 每次重跑导出都清空旧产物（避免 manifest 保留旧单位条目）
    for old in out_dir.iterdir():
        if old.is_file(): old.unlink()
这个**设计本身是对的**（manifest 必须与磁盘一致），错的是我的**用法**：
把"对照实验"跑在了交付物的同一个目录里。

本模块提供两条保护：
  1. `assert_safe_export_target()` —— 对照脚本**默认拒绝**写入一个"已有完整产物"的目录，
     除非显式加 `--allow-overwrite`；
  2. `snapshot_artifacts()` / `restore_artifacts()` —— 需要对照时先把产物快照到
     `env/tmp/artifact_snapshots/<job>/`，跑完自动还原。

用法（在对照脚本里）：
    from gsflow.gsdata.export_guard import guard_export_dir
    guard_export_dir(job_dir, allow_overwrite=args.allow_overwrite)
"""
from __future__ import annotations

import shutil
from pathlib import Path

# 交付物的判定：这 5 个里出现 >= 3 个，就认为该目录是"交付目录"
DELIVERABLE_NAMES = (
    "model.glb",                 # 点云 glb
    "model_mesh.ply",            # 实体模型 ply
    "model_mesh.glb",            # 网络(网格) glb
    "model_mesh_textured.glb",   # 带纹理最终 glb
    "viewer.html",
)


def find_deliverables(export_dir: Path) -> list[str]:
    """返回该目录里已存在的交付物文件名。"""
    if not export_dir.exists():
        return []
    return [n for n in DELIVERABLE_NAMES if (export_dir / n).exists()]


def assert_safe_export_target(export_dir: Path, allow_overwrite: bool = False) -> None:
    """若目录里已有交付物且未显式允许覆盖，则**抛错并给出修复建议**。

    为什么抛错而不是警告：覆盖是不可逆的（导出阶段会 unlink 全部文件），
    损失的是"已经花几十分钟训练+导出"的交付物。报警容易被忽略。
    """
    found = find_deliverables(export_dir)
    if len(found) >= 3 and not allow_overwrite:
        raise RuntimeError(
            f"拒绝写入 {export_dir}：该目录已有 {len(found)} 个交付物 {found}。\n"
            f"导出阶段会**清空整个目录**再重建，所以直接在这里做对照实验会删掉交付物。\n"
            f"三种正确做法：\n"
            f"  1) 换一个作业目录做对照（推荐）：--job <新作业>\n"
            f"  2) 把完整 formats 一起导出：--set export.formats=glb_points,mesh,viewer_html\n"
            f"  3) 确实要覆盖：加 --allow-overwrite（并确认交付物可重建）"
        )


def snapshot_artifacts(export_dir: Path, root: Path | None = None) -> Path | None:
    """把 export 目录整体快照到 `env/tmp/artifact_snapshots/<name>/`，返回快照路径。"""
    if not export_dir.exists() or not any(export_dir.iterdir()):
        return None
    root = root or (Path(__file__).resolve().parents[3] / "env" / "tmp" / "artifact_snapshots")
    dest = root / export_dir.parent.name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for f in export_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    return dest


def restore_artifacts(snapshot: Path | None, export_dir: Path) -> int:
    """把快照还原回 export 目录（覆盖同名文件），返回还原的文件数。"""
    if snapshot is None or not snapshot.exists():
        return 0
    export_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in snapshot.iterdir():
        if f.is_file():
            shutil.copy2(f, export_dir / f.name)
            n += 1
    return n
