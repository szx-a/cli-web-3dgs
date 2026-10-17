"""判定"纹理黑洞"是不是**背景几何**造成的。

## 推理

`mask.mode=keep` 让背景参与重建 ⇒ 网格里同时有**物体**和**背景**（房间、墙、地面）。
但烘焙只从**相机视角**采样颜色：
  * 物体表面：总有相机正面看到它 ⇒ 能采到颜色；
  * **背景表面**：相机是朝物体看的，背景只在画面边缘/背后出现，
    很多背景面**没有任何相机正面看到** ⇒ 那些 UV texel 必然保持黑色。

实测 img3-crop：**43.7% 的 chart 内部 texel 是黑的**。

## 判据

把网格顶点投影到全部训练视角，统计每个顶点"落在**前景掩码内**的视角比例"：
  * 物体顶点：比例接近 1.0（每个视角都能看到它在掩码内）；
  * 背景顶点：比例接近 0（它只在掩码外可见）。

若存在大量"比例 ≈ 0"的顶点 ⇒ 网格含背景 ⇒ 这就是纹理黑洞的来源，
且**它们本来就不该出现在最终模型里**（用户要的是牌坊，不是房间）。

用法: scripts\\gpu_python.cmd scripts\\classify_mesh_fg_bg.py --job img3-crop
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
    ap.add_argument("--views", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import open3d as o3d
    from PIL import Image

    from gsflow.trainers.gsplat_trainer import load_colmap_data

    job = ROOT / "workspace" / args.job
    ply = job / "05_export" / "model_mesh.ply"
    mesh = o3d.io.read_triangle_mesh(str(ply))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    print(f"网格 {len(verts)} 顶点 / {len(mesh.triangles)} 面")

    d = load_colmap_data(job / "04_train" / "data", downscale=1,
                         white_background=False, flip_camera=False)
    c2ws = np.asarray(d["c2ws"], dtype=np.float64)
    ks = np.asarray(d["ks"], dtype=np.float64)
    names = list(d["names"])
    mask_dir = job / "02_masks"

    masks = {}
    for i, n in enumerate(names):
        stem = Path(n).stem
        for c in (mask_dir / f"{stem}.png", mask_dir / f"{n}.png"):
            if c.exists():
                with Image.open(c) as im:
                    masks[i] = np.asarray(im.convert("L")) > 127
                break
    use = sorted(masks.keys())
    if args.views and len(use) > args.views:
        step = len(use) / args.views
        use = [use[int(k * step)] for k in range(args.views)]
    print(f"使用 {len(use)} 个视角（有掩码 {len(masks)}/{len(names)}）")

    vh = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
    inside = np.zeros(len(verts), dtype=np.float64)
    visible = np.zeros(len(verts), dtype=np.float64)
    for i in use:
        m = masks[i]
        H, W = m.shape
        pc = (np.linalg.inv(c2ws[i]) @ vh.T).T[:, :3]
        z = pc[:, 2]
        ok = z > 1e-6
        uv = (ks[i] @ pc.T).T[:, :2] / np.maximum(z, 1e-9)[:, None]
        u = np.round(uv[:, 0]).astype(np.int64)
        v = np.round(uv[:, 1]).astype(np.int64)
        ink = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        hit_in = np.zeros(len(verts), dtype=bool)
        idx = np.nonzero(ink)[0]
        if idx.size:
            hit_in[idx] = m[v[idx], u[idx]]
        inside += hit_in
        visible += ink

    # ⚠️ 必须先区分"从未进入任何视角"的顶点，否则它们的 ratio 会被算成 0，
    # 从而被误判为背景。这类顶点是**未被覆盖**（在物体背面或视锥外），
    # 与"背景几何"是两回事。
    never_visible = visible == 0
    ratio = np.zeros(len(verts), dtype=np.float64)
    ok_mask = ~never_visible
    ratio[ok_mask] = inside[ok_mask] / visible[ok_mask]

    print(f"\n=== 顶点分类（按'可见时落在前景掩码内的比例'）===")
    print(f"  从未进入任何视角（未被覆盖） = {int(never_visible.sum())} 顶点"
          f"（{never_visible.mean()*100:.1f}%）—— 单独处理，不算背景")
    for lo, hi, label in ((0.0, 0.05, "≈0（纯背景）"),
                          (0.05, 0.5, "0.05~0.5（混合）"),
                          (0.5, 0.95, "0.5~0.95（多为前景）"),
                          (0.95, 1.01, "≈1（纯前景/物体）")):
        n = int((ok_mask & (ratio >= lo) & (ratio < hi)).sum())
        print(f"  {label:<22} {n:>7} 顶点（{n/len(verts)*100:5.1f}%）")

    seen = ok_mask
    bg_frac = float((ratio[seen] < 0.5).mean()) if seen.any() else 0.0
    print(f"\n  在被覆盖的顶点里，背景类（比例 < 0.5）合计 = {bg_frac*100:.1f}%")
    print("\n=== 判读 ===")
    if bg_frac > 0.15:
        print(f"  ⇒ **网格里混有约 {bg_frac*100:.0f}% 的背景几何**（占被覆盖顶点）")
        print("     这些面没有任何相机正面看到 ⇒ 烘焙不出颜色 ⇒ 纹理黑洞。")
        print("     它们本来就不该出现在交付模型里（用户要的是牌坊本身）。")
        print("     修法：导出时按此比例过滤，只保留 ratio >= 0.5 的顶点。")
    else:
        print(f"  ⇒ 背景几何很少（{bg_frac*100:.1f}%），纹理黑洞另有原因")

    rep = {"vertices": len(verts), "never_visible": int(never_visible.sum()),
           "bg_frac_lt_0.5_of_seen": round(bg_frac, 4),
           "ratio_median_seen": round(float(np.median(ratio[seen])), 4) if seen.any() else None}
    (job / "05_export" / "mesh_fgbg_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
