"""直接复刻 `bake_texture_from_views` 的采样段，找出"覆盖 0.0%"的真正 bug。

`diag_bake_reject.py` 已证明：41 个视角、820,000 次采样里有 **110,073 次（13.42%）**
能通过"在画面内 + 朝向相机 + 像素非黑"三个条件 —— 所以不是"全灭"。
但实际烘焙报 **覆盖 0.0% / 拒绝 0 次**，说明代码路径里有变量名/作用域错误。

本脚本把烘焙里的那段代码**逐字复刻**到一个最小函数里，喂真实数据，
打印每步的数组形状与 `best` 的状态，定位是哪一行把结果清零。

用法: scripts\\gpu_python.cmd scripts/repro_bake_reject.py --job final-v1
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
    ap.add_argument("--job", required=True)
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import open3d as o3d
    from PIL import Image

    from gsflow.gsdata.mesh_export import unwrap_uv
    from gsflow.trainers.gsplat_trainer import load_colmap_data

    job = ROOT / "workspace" / args.job
    mesh = o3d.io.read_triangle_mesh(str(job / "05_export" / "model_mesh.ply"))
    # 与烘焙一致：先展开 UV
    uv_mesh, uv_tris, corner_uv = unwrap_uv(o3d.geometry.TriangleMesh(mesh), tex_size=2048)
    verts = np.asarray(uv_mesh.vertices, dtype=np.float64)
    tris = np.asarray(uv_tris, dtype=np.int64)
    uv = np.asarray(corner_uv, dtype=np.float64).reshape(-1, 3, 2)
    normal = np.asarray(uv_mesh.vertex_normals, dtype=np.float64)
    print(f"UV 展开后：{len(verts)} 顶点 / {len(tris)} 面")

    d = load_colmap_data(job / "04_train" / "data", downscale=1,
                         white_background=False, flip_camera=False)
    c2ws = np.asarray(d["c2ws"], dtype=np.float64)
    ks = np.asarray(d["ks"], dtype=np.float64)
    names = list(d["names"])
    img_dir = job / "04_train" / "data" / "images"

    cams = []
    for i, n in enumerate(names[:60]):
        p = img_dir / n
        if not p.exists():
            continue
        with Image.open(p) as im:
            img = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        cams.append((np.linalg.inv(c2ws[i]), c2ws[i][:3, 3],
                     ks[i], img.shape[1], img.shape[0], img))
    print(f"相机 {len(cams)} 个")

    tex_size = 2048
    uv_px = uv * (tex_size - 1)
    n_face = len(tris)

    # 只跑第一批，逐行打印
    order = np.arange(n_face)
    fid = order[: args.batch]
    P_list, N_list, px_list, py_list = [], [], [], []
    for f in fid:
        t = uv_px[f]
        x0 = max(int(np.floor(t[:, 0].min())) - 1, 0)
        x1 = min(int(np.ceil(t[:, 0].max())) + 1, tex_size - 1)
        y0 = max(int(np.floor(t[:, 1].min())) - 1, 0)
        y1 = min(int(np.ceil(t[:, 1].max())) + 1, tex_size - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
        a, b, c = t[0], t[1], t[2]
        v0, v1 = b - a, c - a
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            continue
        v2 = pts - a
        w1 = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / den
        w2 = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / den
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -0.002) & (w1 >= -0.002) & (w2 >= -0.002)
        if not inside.any():
            continue
        w0, w1, w2 = w0[inside], w1[inside], w2[inside]
        i0, i1, i2 = tris[f]
        P_list.append(w0[:, None] * verts[i0] + w1[:, None] * verts[i1] + w2[:, None] * verts[i2])
        nrm = (w0[:, None] * normal[i0] + w1[:, None] * normal[i1] + w2[:, None] * normal[i2])
        N_list.append(nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9))
        px_list.append(pts[inside][:, 0].astype(np.int32))
        py_list.append(pts[inside][:, 1].astype(np.int32))

    if not P_list:
        print("  **P_list 为空** —— 第一批没有任何采样点！")
        return 1
    P = np.concatenate(P_list, axis=0)
    N = np.concatenate(N_list, axis=0)
    px = np.concatenate(px_list)
    py = np.concatenate(py_list)
    print(f"采样点 {len(P):,} 个")

    best = np.full(len(P), -1.0)
    col = np.zeros((len(P), 3), dtype=np.float32)
    n_rejected = 0
    for ci, (w2c, ctr, K, iw, ih, img) in enumerate(cams):
        vc = P @ w2c[:3, :3].T + w2c[:3, 3]
        z = vc[:, 2]
        ok = z > 1e-6
        if not ok.any():
            continue
        u = (vc @ K[0, :3]) / np.maximum(z, 1e-9)
        v = (vc @ K[1, :3]) / np.maximum(z, 1e-9)
        ok &= (u >= 0) & (u < iw) & (v >= 0) & (v < ih)
        vd = P - ctr
        vd /= np.maximum(np.linalg.norm(vd, axis=1, keepdims=True), 1e-9)
        facing = -(N * vd).sum(axis=1)
        ok &= facing > 0.15
        if not ok.any():
            continue
        sc = np.where(ok, facing, -1.0)
        take = sc > best
        if not take.any():
            continue
        sy = (img.shape[0] - 1) / max(ih - 1, 1)
        sx = (img.shape[1] - 1) / max(iw - 1, 1)
        ui = np.clip((u[take] * sx).astype(np.int32), 0, img.shape[1] - 1)
        vi = np.clip((v[take] * sy).astype(np.int32), 0, img.shape[0] - 1)

        idx_take = np.nonzero(take)[0]
        pix = img[vi, ui]
        sel = pix.max(axis=1) > 0.024        # >6/255
        if ci == 0:
            print(f"  第 1 个视角：take={int(take.sum())}，非黑={int(sel.sum())}"
                  f"（{sel.mean()*100:.1f}%）")
        if not sel.any():
            continue
        its = idx_take[sel]
        col[its] = pix[sel]
        best[its] = sc[its]
        n_rejected += int((~sel).sum())

    got = best >= 0
    print(f"\n  got (best>=0) = {int(got.sum()):,} / {len(P):,}"
          f"（{got.mean()*100:.1f}%）")
    print(f"  被拒采样 = {n_rejected:,}")
    if got.any():
        print(f"  已填色像素数 = {len(np.unique(np.stack([py[got], px[got]], 1), axis=0)):,}")
        print("  ⇒ 采样逻辑**正常**，问题在别处（如 batch 循环外的代码被改动）")
    else:
        print("  ⇒ 采样逻辑在这里也全灭，需继续查")
    return 0


if __name__ == "__main__":
    sys.exit(main())
