"""诊断 `bake_reject_black` 为何把**全部**采样都拒绝（覆盖 0.0%）。

## 现象

给烘焙加"拒绝黑背景"后，`final-v1` 的贴图覆盖从 100% 掉到 **0.0%**，
且日志显示"因黑背景拒绝采样 **0** 次" —— 说明不是拒绝逻辑本身的问题，
而是**一个采样都没被采纳**（`best` 全为 -1）。

## 两种可能

  A. 采样点投影后**全都落在黑背景**（那么 `pix.max() > 6` 全 False）；
  B. 更早的 `facing > 0.15` 就已经全 False（法线方向反了/朝向不对）。

本脚本在同一份数据上分别统计这几个过滤条件各自杀掉多少采样，
定位是"黑背景"还是"朝向"导致的。**不修改任何文件。**

用法: scripts\\gpu_python.cmd scripts/diag_bake_reject.py --job final-v1
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
    ap.add_argument("--samples", type=int, default=20000)
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import open3d as o3d
    from PIL import Image

    from gsflow.trainers.gsplat_trainer import load_colmap_data

    job = ROOT / "workspace" / args.job
    mesh = o3d.io.read_triangle_mesh(str(job / "05_export" / "model_mesh.ply"))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    vn = np.asarray(mesh.vertex_normals, dtype=np.float64)
    print(f"网格 {len(verts)} 顶点 / {len(tris)} 面")

    # 抽样表面点（面心 + 顶点法线平均）
    rng = np.random.default_rng(0)
    sel = rng.choice(len(tris), size=min(args.samples, len(tris)), replace=False)
    P = verts[tris[sel]].mean(axis=1)
    N = vn[tris[sel]].mean(axis=1)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)

    d = load_colmap_data(job / "04_train" / "data", downscale=1,
                         white_background=False, flip_camera=False)
    c2ws = np.asarray(d["c2ws"], dtype=np.float64)
    ks = np.asarray(d["ks"], dtype=np.float64)
    names = list(d["names"])
    img_dir = job / "04_train" / "data" / "images"

    img0 = None
    for n in names:
        for c in (img_dir / n,):
            if c.exists():
                with Image.open(c) as im:
                    img0 = np.asarray(im.convert("RGB"))
                break
        if img0 is not None:
            break
    if img0 is not None:
        print(f"训练图：{img0.shape}，黑像素 {(img0.max(axis=2) <= 6).mean()*100:.1f}%")

    n_total = 0
    n_inframe = 0
    n_facing = 0
    n_notblack = 0
    best_facing = np.full(len(P), -1.0)
    for i, n in enumerate(names):
        p = img_dir / n
        if not p.exists():
            continue
        with Image.open(p) as im:
            img = np.asarray(im.convert("RGB"), dtype=np.float32)
        ih, iw = img.shape[:2]
        w2c = np.linalg.inv(c2ws[i])
        vc = P @ w2c[:3, :3].T + w2c[:3, 3]
        z = vc[:, 2]
        ok = z > 1e-6
        u = (vc @ ks[i][0, :3]) / np.maximum(z, 1e-9)
        v = (vc @ ks[i][1, :3]) / np.maximum(z, 1e-9)
        inf = ok & (u >= 0) & (u < iw) & (v >= 0) & (v < ih)
        ctr = c2ws[i][:3, 3]
        vd = P - ctr
        vd /= np.maximum(np.linalg.norm(vd, axis=1, keepdims=True), 1e-9)
        facing = -(N * vd).sum(axis=1)
        fac = inf & (facing > 0.15)
        # 采样像素
        ui = np.clip(u.astype(np.int32), 0, iw - 1)
        vi = np.clip(v.astype(np.int32), 0, ih - 1)
        pix = img[vi, ui]
        notblack = fac & (pix.max(axis=1) > 6)

        n_total += len(P)
        n_inframe += int(inf.sum())
        n_facing += int(fac.sum())
        n_notblack += int(notblack.sum())
        upd = fac & (facing > best_facing)
        best_facing[upd] = facing[upd]

        if i >= 40:
            break

    print(f"\n=== 各过滤条件通过量（{i+1} 个视角，{n_total:,} 次采样）===")
    print(f"  落在画面内          {n_inframe:>10,} ({n_inframe/max(n_total,1)*100:6.2f}%)")
    print(f"  且朝向相机 >0.15    {n_facing:>10,} ({n_facing/max(n_total,1)*100:6.2f}%)")
    print(f"  且像素非黑(>6)      {n_notblack:>10,} ({n_notblack/max(n_total,1)*100:6.2f}%)")
    print(f"\n  至少一个视角朝向合格的点：{int((best_facing >= 0).sum()):,}"
          f" / {len(P):,}（{(best_facing >= 0).mean()*100:.1f}%）")

    print("\n=== 判读 ===")
    if n_facing == 0:
        print("  ⇒ **朝向判据全灭**：法线方向与'面→相机'反了。")
        print("     这是 `_orient_normals_outward` 与烘焙 facing 约定不一致导致。")
    elif n_notblack == 0:
        print("  ⇒ **朝向还行，但采样全落在黑背景**：说明采样点投影到背景区域。")
        print("     可能是网格比物体大（泊松把背景也包进来了）或位姿有偏移。")
    elif n_notblack / max(n_facing, 1) < 0.2:
        print(f"  ⇒ 只有 {n_notblack/max(n_facing,1)*100:.1f}% 的非黑采样 —— 采样点大量落在背景。")
    else:
        print("  ⇒ 各条件都能通过一部分，问题不在这里（另有原因）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
