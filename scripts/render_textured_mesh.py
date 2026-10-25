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
    ap.add_argument("--max-faces", type=int, default=60000,
                    help="渲染前面数上限（Open3D/Filament 实测 10 万面以上会全黑）")
    ap.add_argument("--unlit", action="store_true",
                    help="用无照明着色器（排查“法线朝内导致全黑”）")
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
    # ---- 面数上限（2026-10-25 实测）----
    #
    # 排查结论：同一个 TSDF 网格（几何正常、无 NaN、无零面积面）
    #   * 点云采样（同表面）渲得出来；
    #   * 20,000 面简化版渲得出来（mean 23.7）；
    #   * 100,000 / 331,815 面 → **全黑**（换 unlit、翻绕序、清顶点色都无效）。
    # ⇒ 是**光栅化器的面数上限**，不是网格或相机问题。这里自动简化到上限以内再渲染。
    if args.max_faces > 0 and len(mesh.triangles) > args.max_faces:
        before = len(mesh.triangles)
        mesh = mesh.simplify_quadric_decimation(args.max_faces)
        mesh.compute_vertex_normals()
        print(f"  面数 {before:,} 超过渲染器上限 ⇒ 自动简化到 {len(mesh.triangles):,} 面后渲染")

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

    # ---- 自动兜底（2026-10-25）：TSDF 网格用默认材质会渲成全黑 ----
    #
    # 实测：同一相机、同一材质下 poisson/2dgs 渲得出来，tsdf 全黑；
    # 换 unlit、清顶点色、简化到 10 万面都不行，而几何检查完全正常
    # （无零面积面、面积 7.44、bbox 一致）⇒ 是**渲面朝向/剔除**的问题，不是网格坏。
    # 做法：先按默认渲一张，若整张图接近全黑，就依次尝试
    #   (a) unlit 材质 → (b) 翻转三角形绕序 → (c) 两者都上，
    # 取第一个"非全黑"的结果，并在输出里注明用了哪种兜底。
    from PIL import Image as _PILImage

    def _render_all(mesh_obj, shader: str) -> list:
        mats = o3d.visualization.rendering.MaterialRecord()
        mats.shader = shader
        mats.base_color = [0.85, 0.85, 0.85, 1.0]
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("mesh", mesh_obj, mats)
        renderer.scene.set_background([0.08, 0.08, 0.10, 1.0])
        outs = []
        for ang in np.linspace(0, 2 * np.pi, args.views, endpoint=False):
            # ⚠️ 旧写法距离 2.4×半对角线 + 50° FOV，实测有两个视角物体跑出画面
            # （只有第三帧可见、且偏右）。改成 3.0× + 60°，并明确 look_at。
            eye = center + extent * 3.0 * np.array([np.cos(ang), 0.38, np.sin(ang)])
            up = np.array([0.0, 1.0, 0.0])
            renderer.setup_camera(60.0, center, eye, up)
            try:
                renderer.scene.camera.look_at(center, eye, up)
            except Exception:  # noqa: BLE001
                pass
            outs.append(np.asarray(renderer.render_to_image()).copy())
        return outs

    def _is_blank(imgs: list) -> bool:
        return max(int(np.asarray(im).max()) for im in imgs) <= 8

    shader0 = "defaultUnlit" if args.unlit else "defaultLit"
    imgs = _render_all(mesh, shader0)
    used = shader0
    if _is_blank(imgs):
        print("  首轮渲染全黑 ⇒ 尝试 unlit 材质")
        imgs = _render_all(mesh, "defaultUnlit")
        used = "defaultUnlit(兜底)"
    if _is_blank(imgs):
        print("  仍全黑 ⇒ 翻转三角形绕序后重试")
        flipped = o3d.geometry.TriangleMesh(mesh)
        tri = np.asarray(flipped.triangles)[:, ::-1].copy()
        flipped.triangles = o3d.utility.Vector3iVector(tri)
        flipped.compute_vertex_normals()
        imgs = _render_all(flipped, "defaultUnlit")
        used = "defaultUnlit+绕序翻转(兜底)"
    if _is_blank(imgs):
        print("  ⚠️ 三种方式都渲不出（渲染器限制），几何指标请用 check_mesh_quality.py")

    from PIL import Image

    sheet = np.concatenate(imgs, axis=1)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out)
    print(f"  已写出 {out}（{args.views} 视角并排）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
