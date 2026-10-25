"""用 Open3D 离屏渲染**带贴图的网格**，给出一张"实际外观"预览。

## 为什么需要

`export` 阶段预览用的是 Filament 离屏渲染，在本机无窗口环境下会返回空白图（已记录）。
所以"贴图到底长什么样"一直只能靠用户自己打开 glb 看。这里改用 Open3D 的离屏渲染器试一把：
它在同一台机器上**能跑**（泊松重建/网格 IO 都在用），若能渲出图，就补上这个可视化缺口。

用法:
    python scripts/render_textured_mesh.py --job img4-v1 --views 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="img4-v1")
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="env/tmp/textured_mesh.png")
    args = ap.parse_args()

    import open3d as o3d

    jd = ROOT / "workspace" / args.job / "05_export"
    obj = jd / "model_mesh.obj"
    tex = jd / "model_mesh_texture.png"
    if not obj.exists():
        print("缺少 model_mesh.obj")
        return 1

    mesh = o3d.io.read_triangle_mesh(str(obj), enable_post_processing=True)
    if tex.exists():
        try:
            mesh.textures = [o3d.io.read_image(str(tex))]
            # OBJ \u91cc\u5e26\u4e86 vt\uff0cOpen3D \u4f1a\u81ea\u52a8\u5efa\u7acb\u8d34\u56fe\u5750\u6807
        except Exception as exc:  # noqa: BLE001
            print(f"  贴图加载失败：{type(exc).__name__}: {exc}")
    if len(mesh.triangles) == 0:
        print("网格为空")
        return 1

    aabb = mesh.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    extent = float(np.linalg.norm(aabb.get_extent())) / 2
    print(f"  网格 {len(mesh.triangles)} 面，包围盒 {np.round(aabb.get_extent(),2).tolist()}")

    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(args.size, args.size)
    except Exception as exc:  # noqa: BLE001
        print(f"  离屏渲染器不可用（{type(exc).__name__}: {exc}）—— 本机无窗口环境限制")
        return 1

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit" if tex.exists() else "defaultLit"
    mat.base_color = [0.85, 0.85, 0.85, 1.0]
    renderer.scene.add_geometry("mesh", mesh, mat)
    renderer.scene.set_background([0.08, 0.08, 0.10, 1.0])
    try:
        renderer.scene.scene.set_sun_light([0.3, -0.6, -0.7], [1, 1, 1], 90000)
        renderer.scene.scene.enable_sun_light(True)
    except Exception:  # noqa: BLE001
        pass

    imgs = []
    for k, ang in enumerate(np.linspace(0, 2 * np.pi, args.views, endpoint=False)):
        eye = center + extent * 2.4 * np.array([np.cos(ang), 0.45, np.sin(ang)])
        renderer.setup_camera(50.0, center, eye, [0, 1, 0])
        img = renderer.render_to_image()
        arr = np.asarray(img)
        imgs.append(arr)
        print(f"  视角 {k+1}/{args.views} 渲染完成 {arr.shape[:2]}")

    from PIL import Image

    sheet = np.concatenate(imgs, axis=1)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out)
    print(f"  已写出 {out}（{args.views} 视角并排）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
