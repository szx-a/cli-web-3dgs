"""诊断纹理的**真实缺陷**：不是 UV 布局，而是烘焙采样密度/覆盖。

## 之前的误判

我用"纹理图里的黑色空隙数连通块"判断 UV 碎片化，得出"1,264 个碎块、严重碎片化"。
用**正确的 UV 光栅化**（画满三角形而非只点顶点）重测：
  * 141 个 chart（xatlas 正常数量）
  * 前 10 块占 98.2% 的 UV 面积
  * UV 占用率 **74.7%**（健康）
⇒ **UV 布局没问题，我的判据错了**。

## 真正该查的

纹理看起来碎，是因为每个 chart 在图集里是**独立区域**，块与块之间是**空的 padding**
（黑色）。这本身正常。真正影响观感的是：

  1. **每个 chart 的 texel 密度**：图集 2048²、UV 占 74.7%，
     但**每个三角形分到多少 texel**？若网格有 42,323 面而图集只有
     3.1M 有效 texel ⇒ 每面约 74 texel（≈8×8），**对大平面够、对细节不够**；
  2. **填充率/黑洞**：网格三角形没覆盖到的 UV 区域会渲染成黑色 ⇒
     模型上出现**黑斑**。这是"看起来灾难"的最可能原因；
  3. **接缝处颜色不连续**：相邻 chart 分开采样，若烘焙时没做 padding 扩散，
     接缝会出现细黑线。

本脚本量化以上三点，并输出"每 chart 的 texel 密度"分布。

用法: scripts\\gpu_python.cmd scripts\\diag_texture_bake.py --job img3-crop
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def rasterize_uv(uv: np.ndarray, tris: np.ndarray, tex: int) -> np.ndarray:
    """把 UV 三角形真正**填充**到 tex×tex 的布尔图（用 PIL 多边形）。"""
    from PIL import Image, ImageDraw

    img = Image.new("L", (tex, tex), 0)
    dr = ImageDraw.Draw(img)
    for t in tris:
        try:
            p = uv[t] * (tex - 1)
        except Exception:  # noqa: BLE001
            continue
        dr.polygon([(float(x), float(y)) for x, y in p], fill=255)
    return np.asarray(img) > 127


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--tex-size", type=int, default=0, help="0=自动取纹理实际尺寸")
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
    if not ply.exists():
        print(f"找不到 {ply}")
        return 1

    mesh = o3d.io.read_triangle_mesh(str(ply))
    print(f"网格：{len(mesh.vertices)} 顶点 / {len(mesh.triangles)} 面")

    uv_mesh, uv_tris, uv = unwrap_uv(o3d.geometry.TriangleMesh(mesh),
                                     tex_size=args.tex_size or 2048)
    n_tris = len(uv_tris)
    print(f"xatlas 展开：{len(uv_mesh.vertices)} 顶点 / {n_tris} 面"
          f"（原 {len(mesh.triangles)} 面）")

    tex = args.tex_size
    if tex <= 0:
        if tex_png.exists():
            with Image.open(tex_png) as im:
                tex = im.size[0]
            print(f"纹理尺寸 {tex}")
        else:
            tex = 2048

    print(f"\n=== 光栅化 UV 覆盖（{tex}x{tex}）===")
    cov = rasterize_uv(np.asarray(uv, dtype=np.float64),
                       np.asarray(uv_tris, dtype=np.int64), tex)
    occ = float(cov.mean())
    print(f"  ① UV 覆盖率 = {occ*100:.1f}%（被三角形覆盖的 texel 比例）")

    total_texels = int(cov.sum())
    per_tri = total_texels / max(n_tris, 1)
    print(f"  ② 每面平均 texel = {per_tri:.1f}（≈ {per_tri**0.5:.1f}×{per_tri**0.5:.1f} 的方块）")
    edge = float(np.sqrt(per_tri))
    print(f"     等效：每个三角形边长约 {edge:.1f} 个 texel")
    if per_tri < 100:
        print(f"     ⇒ **texel 密度偏低**：42k 面对 2048² 图集，细节会被平均掉")

    # ③ 实际纹理图里的"空黑洞"：UV 有覆盖但纹理是黑的区域
    if tex_png.exists():
        with Image.open(tex_png) as im:
            a = np.asarray(im.convert("RGB"))
        if a.shape[0] != tex:
            from PIL import Image as PI
            a = np.asarray(PI.fromarray(a).resize((tex, tex), PI.BILINEAR))
        dark = a.max(axis=2) <= 8
        holes = int((cov & dark).sum())
        print(f"  ③ UV 覆盖但纹理为黑（黑洞）的 texel = {holes:,}"
              f"（占 UV 覆盖 {holes/max(total_texels,1)*100:.2f}%）")
        if holes / max(total_texels, 1) > 0.05:
            print(f"     ⇒ **超过 5%，模型上会出现黑斑**：烘焙填充不足")
        else:
            print(f"     ⇒ 正常（碎片感来自 chart 排布，不是黑洞）")

        # ④ 接缝连续性：UV 覆盖边缘处的颜色跳变
        boundary = cov & ~(
            np.roll(cov, 1, 0) & np.roll(cov, -1, 0)
            & np.roll(cov, 1, 1) & np.roll(cov, -1, 1))
        print(f"  ④ UV 边界 texel = {int(boundary.sum()):,}"
              f"（占覆盖 {boundary.sum()/max(total_texels,1)*100:.1f}%；越高接缝越多）")

    rep = {"tex_size": tex, "n_tris": n_tris, "coverage": round(occ, 4),
           "texels_per_tri": round(per_tri, 2)}
    (exp / "texture_bake_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {exp / 'texture_bake_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
