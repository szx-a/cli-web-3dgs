"""可视化：UV 覆盖 vs 纹理黑色区域 —— 精确区分"正常 padding"与"烘焙黑洞"。

## 为什么需要

img3-crop 的纹理图看起来有大量黑线，但两种可能：
  A. **正常 padding**：UV 岛之间的空隙（xatlas 会给每个 chart 留边），
     模型采样不会用到，无害；
  B. **烘焙黑洞**：UV 岛**内部**没填上颜色 ⇒ 模型表面出现黑斑，是真 bug。

单看纹理图无法区分（两种都是黑的）。本脚本生成一张**三色叠加图**：
  * 灰色 = UV 覆盖**且**纹理有颜色 ⇒ 正常；
  * **红色 = UV 覆盖但纹理是黑的 ⇒ 黑洞（bug）**；
  * 黑色 = 非 UV 区域（padding，无害）。

并输出"黑洞面积 / UV 覆盖面积"的精确比例。

用法: scripts\\gpu_python.cmd scripts\\visualize_texture_holes.py --job img3-crop
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import open3d as o3d
    from PIL import Image

    from gsflow.gsdata.mesh_export import unwrap_uv

    job = ROOT / "workspace" / args.job
    exp = job / "05_export"
    ply = exp / "model_mesh.ply"
    tex_png = exp / "model_mesh_texture.png"
    for p in (ply, tex_png):
        if not p.exists():
            print(f"找不到 {p}")
            return 1

    with Image.open(tex_png) as im:
        tex = np.asarray(im.convert("RGB"))
    T = tex.shape[0]
    dark = tex.max(axis=2) <= 10
    print(f"纹理 {T}x{T}，黑色 texel 占 {dark.mean()*100:.1f}%")

    # UV 覆盖（用与烘焙相同的展开与光栅化，保证一一对应）
    mesh = o3d.io.read_triangle_mesh(str(ply))
    # ⚠️ `unwrap_uv` 返回 `(mesh, new_tris, corner_uv)`，其中 corner_uv 是
    # **逐三角形-角** 的 UV，形状 `(n_tris*3, 2)`（不是 `(n_tris, 3, 2)`）。
    # 我第一版把它 reshape 成 `(-1,3,2)` 再用 `uv[tris[t]]` 做嵌套索引，
    # 于是元素变成 3 个坐标值、解包成 (x,y) 就报 "too many values to unpack"。
    uv_mesh, tris, corner_uv = unwrap_uv(o3d.geometry.TriangleMesh(mesh), tex_size=T)
    tris = np.asarray(tris, dtype=np.int64)
    corner_uv = np.asarray(corner_uv, dtype=np.float64).reshape(-1, 3, 2)  # 现在是 (n_tris,3,2)

    from PIL import ImageDraw
    m = Image.new("L", (T, T), 0)
    dr = ImageDraw.Draw(m)
    px = corner_uv * (T - 1)
    for t in range(len(tris)):
        tri = px[t]                    # 直接取第 t 个三角形的 3 个角
        if len(tri) != 3:
            continue
        dr.polygon([(float(x), float(y)) for x, y in tri], fill=255)
    cov = np.asarray(m) > 127
    print(f"UV 覆盖 texel 占 {cov.mean()*100:.1f}%")

    # 关键区分：**腐蚀** UV 覆盖图，去掉边界 2px，
    # 这样剩下的是 chart"内部"，内部若有黑 = 真空洞
    interior = cov.copy()
    for _ in range(2):
        interior &= np.roll(interior, 1, 0) & np.roll(interior, -1, 0)
        interior &= np.roll(interior, 1, 1) & np.roll(interior, -1, 1)

    holes = interior & dark
    good = interior & ~dark
    print(f"\n=== 拆分 ===")
    print(f"  chart 内部 texel = {int(interior.sum()):,}")
    print(f"    有颜色（正常）  = {int(good.sum()):,}（{good.sum()/max(interior.sum(),1)*100:.1f}%）")
    print(f"    **黑色（黑洞）** = {int(holes.sum()):,}（{holes.sum()/max(interior.sum(),1)*100:.1f}%）")
    print(f"  UV 边界附近 texel = {int((cov & ~interior).sum()):,}（含 padding，无害）")

    # 三色叠加图
    out = np.zeros((T, T, 3), dtype=np.uint8)
    out[cov & ~dark] = (150, 150, 150)          # 灰：UV 有覆盖且有颜色
    out[cov & dark] = (255, 0, 0)               # 红：UV 覆盖但黑（黑洞）
    out[~cov] = (20, 20, 20)                    # 深灰：非 UV
    Image.fromarray(out).save(exp / "texture_holes_overlay.png")
    print(f"\n已生成三色叠加图 {exp / 'texture_holes_overlay.png'}"
          f"（红=黑洞，灰=正常，深灰=padding）")

    hole_ratio = float(holes.sum() / max(interior.sum(), 1))
    rep = {"tex_size": T, "uv_coverage": round(float(cov.mean()), 4),
           "interior_texels": int(interior.sum()),
           "hole_texels": int(holes.sum()),
           "hole_ratio_of_interior": round(hole_ratio, 4)}
    (exp / "texture_holes_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n判读：黑洞占 chart 内部 {hole_ratio*100:.1f}%"
          + ("（<5% 可接受：模型上几乎看不到黑斑）" if hole_ratio < 0.05
             else "（**>5%：模型表面会有可见黑斑，需要修烘焙**）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
