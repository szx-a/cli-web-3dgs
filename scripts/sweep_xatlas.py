"""探测 xatlas 的可调参数，并测量不同设置下的 UV 质量。

## 问题

img3-crop 的纹理图集有 **1,264 个连通块**（226 个 >1000px），
接缝像素占 9.2%，UV 占用率仅 45%，最大块只占 5.9% 的非空像素。
块内**内容本身是清晰的**（梯度均值 9.78，不糊），所以问题在**UV 展开的碎片化**，
不是烘焙分辨率。

`unwrap_uv()` 现在调用 `xatlas.parametrize(verts, tris)` 用**默认参数**。
xatlas 有 `ChartOptions`（控制切分）与 `PackOptions`（控制排布），
对本项目这种"面数多、表面有噪声"的网格，默认的锐角切分会把表面切成上千块。

本脚本：
  1. 打印 xatlas 的可调字段（不同版本字段名不同，先探测再改）；
  2. 用几组参数组合跑同一网格，输出**(岛数, 占用率, 最大块占比, 接缝占比)**，
     据此选最优组合，而不是凭猜测改参数。

用法: scripts\\gpu_python.cmd scripts\\sweep_xatlas.py --job img3-crop
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def islands_of(uv: np.ndarray, tris: np.ndarray, tex: int = 256) -> dict:
    """把 UV 光栅化成占用图，统计连通块。"""
    from PIL import Image

    img = np.zeros((tex, tex), dtype=np.uint8)
    # 逐三角形填充（粗 rasterize：只填顶点与重心，够统计用）
    for t in tris[:: max(1, len(tris) // 40000)]:
        try:
            p = uv[t] * tex
        except Exception:  # noqa: BLE001
            continue
        for x, y in p:
            xi = int(np.clip(x, 0, tex - 1))
            yi = int(np.clip(y, 0, tex - 1))
            img[yi, xi] = 255
    occ = float((img > 0).mean())
    try:
        from scipy import ndimage
        lab, n = ndimage.label(img > 0)
        sizes = np.bincount(lab.ravel())[1:]
        n_big = int((sizes > 20).sum())
        top = float(np.sort(sizes)[::-1][:10].sum() / max(sizes.sum(), 1))
        return {"occupancy": occ, "n_islands": int(n), "n_big": n_big, "top10": top}
    except ImportError:
        return {"occupancy": occ, "n_islands": -1, "n_big": -1, "top10": 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--ply", default="", help="网格（默认 05_export/model_mesh.ply）")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import open3d as o3d
    import xatlas

    job = ROOT / "workspace" / args.job
    ply = Path(args.ply) if args.ply else (job / "05_export" / "model_mesh.ply")
    mesh = o3d.io.read_triangle_mesh(str(ply))
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    tris = np.asarray(mesh.triangles, dtype=np.uint32)
    print(f"网格 {ply.name}: {len(verts)} 顶点 / {len(tris)} 面")

    # 探测可用参数
    co_fields, po_fields = [], []
    try:
        co = xatlas.ChartOptions()
        co_fields = [k for k in dir(co) if not k.startswith("_")]
        print(f"\nChartOptions 可用字段：{co_fields}")
    except Exception as e:  # noqa: BLE001
        print(f"ChartOptions 不可用：{e}")
    try:
        po = xatlas.PackOptions()
        po_fields = [k for k in dir(po) if not k.startswith("_")]
        print(f"PackOptions 可用字段：{po_fields}")
    except Exception as e:  # noqa: BLE001
        print(f"PackOptions 不可用：{e}")

    # 参数组合
    combos = [("default", {})]
    if "max_cost" in co_fields:
        combos.append(("max_cost=2", {"chart_max_cost": 2.0}))
        combos.append(("max_cost=1e9", {"chart_max_cost": 1e9}))
    if "normal_deviation_weight" in co_fields:
        combos.append(("low_normal_dev", {"chart_normal_deviation_weight": 0.1}))
    if "fix_winding" in co_fields:
        combos.append(("fix_winding", {"chart_fix_winding": True}))
    if "resolution" in po_fields:
        combos.append(("pack_res=2048", {"pack_resolution": 2048}))
    # 去重
    seen = set()
    uniq = []
    for name, kw in combos:
        key = tuple(sorted(kw.items()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((name, kw))

    print(f"\n{'参数组合':<24}{'岛数':>8}{'占用率':>9}{'大块数':>8}{'前10占比':>10}")
    print("-" * 61)
    results = []
    for name, kw in uniq:
        try:
            chart = None
            packing = None
            if any(k.startswith("chart_") for k in kw):
                chart = xatlas.ChartOptions()
                for k, v in kw.items():
                    if k.startswith("chart_"):
                        field = k[len("chart_"):]
                        if hasattr(chart, field):
                            setattr(chart, field, v)
            if any(k.startswith("pack_") for k in kw):
                packing = xatlas.PackOptions()
                for k, v in kw.items():
                    if k.startswith("pack_"):
                        field = k[len("pack_"):]
                        if hasattr(packing, field):
                            setattr(packing, field, v)
            kwargs = {}
            if chart is not None:
                kwargs["chart_options"] = chart
            if packing is not None:
                kwargs["pack_options"] = packing
            vmapping, new_tris, uvs = xatlas.parametrize(
                verts, tris, **kwargs) if kwargs else xatlas.parametrize(verts, tris)
            uv = np.asarray(uvs, dtype=np.float64)
            st = islands_of(uv, np.asarray(new_tris, dtype=np.int64))
            results.append((name, st))
            print(f"{name:<24}{st['n_islands']:>8}{st['occupancy']*100:>8.1f}%"
                  f"{st['n_big']:>8}{st['top10']*100:>9.1f}%")
        except Exception as e:  # noqa: BLE001
            print(f"{name:<24}  失败：{type(e).__name__}: {e}")

    if len(results) >= 2:
        # 选"岛数少 + 前10占比高"的组合
        best = min(results, key=lambda r: (r[1]["n_islands"], -r[1]["top10"]))
        print(f"\n最优（岛数最少、排布最紧凑）：**{best[0]}**"
              f"（{best[1]['n_islands']} 岛，前10占比 {best[1]['top10']*100:.1f}%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
