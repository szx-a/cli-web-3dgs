"""把网格渲成多角度 PNG（open3d OffscreenRenderer，显式相机 API）。

## 踩坑记录（两次都出在取景）

1. `setup_camera(fov, eye, center, up)` 在本机 open3d 版本上**取景不对**：
   画面里只有角落一小块几何。改用
   `scene.camera.set_projection(...)` + `scene.camera.look_at(...)` 显式设置。
2. 归一化用包围盒中心/对角线（本例无离群点；有尖刺时应先在调用方清理）。

用法: python scripts/render_mesh_views.py --job img3-final --size 720
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
    ap.add_argument("--job", default="img3-final")
    ap.add_argument("--mesh", default="")
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import open3d as o3d
    from open3d.visualization.rendering import Camera

    jd = ROOT / "workspace" / args.job
    mesh_p = Path(args.mesh) if args.mesh else jd / "05_export" / "model_mesh.ply"
    if not mesh_p.exists():
        print(f"没有网格：{mesh_p}")
        return 1
    out_dir = Path(args.out) if args.out else jd / "05_export" / "mesh_views"
    out_dir.mkdir(parents=True, exist_ok=True)

    m = o3d.io.read_triangle_mesh(str(mesh_p))
    m.compute_vertex_normals()
    v = np.asarray(m.vertices)
    c = (v.max(0) + v.min(0)) / 2
    ext = float(np.linalg.norm(v.max(0) - v.min(0)))
    m.translate(-c)
    m.scale(2.0 / max(ext, 1e-9), center=(0, 0, 0))

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.size, args.size)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [0.82, 0.78, 0.70, 1.0]
    renderer.scene.add_geometry("m", m, mat)
    renderer.scene.set_background([0.05, 0.06, 0.08, 1.0])
    # 关键：显式投影 + look_at（setup_camera 在本机取景异常）
    renderer.scene.camera.set_projection(
        50.0, 1.0, 0.01, 100.0, Camera.FovType.Vertical)

    views = {
        "front": (0.0, 8.0),
        "back": (180.0, 8.0),
        "side": (90.0, 8.0),
        "top": (0.0, 80.0),
        "iso": (35.0, 22.0),
    }
    n = 0
    for name, (yaw, pitch) in views.items():
        cy, sy = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
        cp, sp = np.cos(np.deg2rad(pitch)), np.sin(np.deg2rad(pitch))
        eye = np.array([3.6 * cp * sy, 3.6 * sp, 3.6 * cp * cy])
        renderer.scene.camera.look_at([0.0, 0.0, 0.0], eye, [0.0, 1.0, 0.0])
        img = renderer.render_to_image()
        p = out_dir / f"mesh_{name}.png"
        o3d.io.write_image(str(p), img)
        n += 1
        print(f"  {name:<6} -> {p.name}")
    print(f"\n  已写出 {n} 张到 {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
