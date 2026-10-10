"""检查 `_read_ply` 读出的各字段是否合理（量纲/分布），以及与原始 PLY 头是否一致。

动机：三个诊断工具（verify_mesh_quality / tune_mesh / diag_checkpoint_views）
都依赖 `ExportStage._read_ply`。若它的 scale/opacity 反变换有问题，
所有下游结论都会被污染 —— 这种"共同依赖出错"最难发现。

用法：env\\venv\\Scripts\\python.exe scripts\\diag_read_ply.py --ply <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    args = ap.parse_args()

    ply = Path(args.ply)
    print(f"PLY: {ply}")
    print(f"大小: {ply.stat().st_size/1e6:.1f} MB")

    # 1) 用 plyfile 直接读原始字段名
    from plyfile import PlyData

    pd = PlyData.read(str(ply))
    el = pd.elements[0]
    names = [p.name for p in el.properties]
    print(f"\n原始字段（{len(names)} 个）: {names[:20]}")
    print(f"点数: {len(el.data)}")
    for k in ("scale_0", "scale_1", "scale_2", "opacity", "f_dc_0", "rot_0"):
        if k in names:
            v = np.asarray(el.data[k], dtype=np.float64)
            print(f"  {k:10s} min={v.min():>12.5f} p50={np.median(v):>12.5f} max={v.max():>12.5f}")

    # 2) 用项目的 _read_ply 读，比较
    from gsflow.stages.export import ExportStage

    g = ExportStage()._read_ply(ply, max_points=0)
    print(f"\n_read_ply 键: {sorted(g.keys())}")
    for k in ("xyz", "scales", "opacity", "rgb", "quats"):
        if k in g:
            v = np.asarray(g[k], dtype=np.float64)
            if v.ndim == 1:
                print(f"  {k:8s} shape={v.shape} min={v.min():>10.4f} "
                      f"p50={np.median(v):>10.4f} max={v.max():>10.4f}")
            else:
                print(f"  {k:8s} shape={v.shape} min={v.min():>10.4f} "
                      f"p50={np.median(v):>10.4f} max={v.max():>10.4f}")

    # 3) 关键判断：scales 是"真实尺度"还是"log 尺度"
    sc = np.asarray(g["scales"], dtype=np.float64)
    print("\n--- 判定 ---")
    print(f"scales 若为真实尺度：中位数 {np.median(sc):.5f}"
          f"（场景尺度约 3.5，高斯半径应在 0.01~0.2 量级）")
    print(f"scales 若为 log 尺度：exp(中位数) = {np.exp(np.median(sc)):.5f}")
    md = float(np.median(sc))
    if md < 0:
        print("  => 中位数为负，_read_ply 返回的是 **log 尺度**（未取 exp）")
    elif md > 1.0:
        print("  => 中位数 > 1，可能是 log 尺度但用了不合理默认值，或尺度确实过大")
    else:
        print("  => 中位数在合理范围，像是**真实尺度**")
    op = np.asarray(g["opacity"], dtype=np.float64).reshape(-1)
    print(f"opacity 中位数 {np.median(op):.4f}（若为 logit，sigmoid 后 = "
          f"{1/(1+np.exp(-np.median(op))):.4f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
