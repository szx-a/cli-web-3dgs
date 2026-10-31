"""量化"贴图与真实视角的一致性"——直接度量纹理杂糅程度。

做不了"和真值比"，因为贴图本身就是从这些视角采的；但可以问一个更有判别力的问题：

    对表面上每个采样点，贴图给的颜色，和**最正对该点的那个视角**看到的颜色，差多少？

如果烘焙发生了"不同部位杂糅"，就会出现一批点：贴图颜色和**任何**视角都对不上
（因为它采到了别处的颜色）。指标：

* `err_median`  / `err_p90`：贴图色与最佳视角色的 L1 差（0~1 尺度）；
* `mismatch_frac`：差值 > 0.12 的采样点占比（"明显不对"的比例）；
* `view_err_median`：与**所有可用视角**的最小差（更能反映"像不像某个视角"）。

用法：
  python scripts/probe_texture_consistency.py --job img5-solid-v3 \
      --texture env/tmp/tex_no_occlusion.png [--texture2 workspace/.../model_mesh_texture.png]

注意：几何/UV 必须一致（同一份 obj），只换贴图，才能对比。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from gsflow.gsdata.colmap_io import read_model  # noqa: E402
from gsflow.gsdata.mesh_export import scale_intrinsics  # noqa: E402


def load_obj_with_uv(path: Path):
    """读取 obj 的顶点 / 三角形 / 每角 UV（返回 arrays）。"""
    verts, uvs, tris, tri_uv = [], [], [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            uvs.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("f "):
            parts = line.split()[1:]
            vi, ti = [], []
            for p in parts:
                seg = p.split("/")
                vi.append(int(seg[0]) - 1)
                ti.append(int(seg[1]) - 1 if len(seg) > 1 and seg[1] else 0)
            if len(vi) == 3:
                tris.append(vi)
                tri_uv.append(ti)
    return (np.asarray(verts, dtype=np.float64), np.asarray(tris, dtype=np.int64),
            np.asarray(uvs, dtype=np.float64), np.asarray(tri_uv, dtype=np.int64))


def sample_texture(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """双线性采样贴图（uv 为 0~1）。"""
    h, w = tex.shape[:2]
    x = np.clip(uv[:, 0] * (w - 1), 0, w - 1)
    y = np.clip(uv[:, 1] * (h - 1), 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = (x - x0)[:, None], (y - y0)[:, None]
    c00, c10 = tex[y0, x0], tex[y0, x1]
    c01, c11 = tex[y1, x0], tex[y1, x1]
    return (c00 * (1 - wx) * (1 - wy) + c10 * wx * (1 - wy)
            + c01 * (1 - wx) * wy + c11 * wx * wy)


def evaluate(job: str, texture: Path, n_samples: int = 40000) -> dict:
    job_dir = ROOT / "workspace" / job
    obj = job_dir / "05_export" / "model_mesh_uv.obj"
    if not obj.exists():
        raise SystemExit(f"没有 {obj}（需要带 UV 的网格：export.mesh_texture=true）")
    verts, tris, uvs, tri_uv = load_obj_with_uv(obj)
    tex = np.asarray(Image.open(texture).convert("RGB"), dtype=np.float32) / 255.0

    data = job_dir / "04_train" / "data"
    sp = data / "sparse"
    rec = read_model(sp / "0" if (sp / "0" / "cameras.bin").exists() else sp)
    imgs = sorted(rec.images, key=lambda im: im.name)

    # 在三角形上均匀采样（面积加权）
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if area.sum() <= 0:
        raise SystemExit("网格面积为 0")
    pick = np.random.default_rng(0).choice(len(tris), size=min(n_samples, len(tris) * 4),
                                           p=area / area.sum())
    r1 = np.sqrt(np.random.default_rng(1).random(len(pick)))
    r2 = np.random.default_rng(2).random(len(pick))
    w0 = 1 - r1
    w1 = r1 * (1 - r2)
    w2 = r1 * r2
    P = (w0[:, None] * a[pick] + w1[:, None] * b[pick] + w2[:, None] * c[pick])

    # 采样点的 UV（同一个重心坐标）
    tuv = uvs[tri_uv[pick]]  # (N,3,2)
    uv_pt = w0[:, None] * tuv[:, 0] + w1[:, None] * tuv[:, 1] + w2[:, None] * tuv[:, 2]
    tex_col = sample_texture(tex, uv_pt)

    # 逐视角投影：记录最佳朝向视角的颜色与所有视角的最小差
    best_facing = np.full(len(P), -2.0)
    best_col = np.zeros_like(tex_col)
    min_err = np.full(len(P), np.inf)

    for im in imgs:
        cam = rec.cameras[im.camera_id]
        W, H = cam.width, cam.height
        R, ctr = im.cam_to_world()
        R = np.asarray(R, float)
        ctr = np.asarray(ctr, float)
        K = np.asarray(cam.intrinsics(), float)
        vc = (P - ctr) @ R  # world -> camera
        z = vc[:, 2]
        ok = z > 1e-6
        if not ok.any():
            continue
        u = (vc @ K[0, :3]) / np.maximum(z, 1e-9)
        v = (vc @ K[1, :3]) / np.maximum(z, 1e-9)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        # 用网格法线近似朝向（顶点法线可由三角形法线代替）
        if not ok.any():
            continue
        gt_path = data / "images" / im.name
        if not gt_path.exists():
            continue
        img = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
        ui = np.clip(u.astype(int), 0, W - 1)
        vi = np.clip(v.astype(int), 0, H - 1)
        col = img[vi[ok], ui[ok]]
        e = np.abs(col - tex_col[ok]).mean(axis=1)
        idx = np.nonzero(ok)[0]
        np.minimum.at(min_err, idx, e)
        # 朝向：用相机到点的方向与"点的观测方向"比较（这里用 z 最小的视角近似正对）
        facing = 1.0 / np.maximum(z[ok], 1e-6)
        better = facing > best_facing[idx]
        best_facing[idx[better]] = facing[better]
        best_col[idx[better]] = col[better]

    valid = np.isfinite(min_err)
    err_best = np.abs(best_col - tex_col).mean(axis=1)[valid]
    return {
        "texture": str(texture.relative_to(ROOT)) if texture.is_relative_to(ROOT) else texture.name,
        "samples": int(valid.sum()),
        "err_median": float(np.median(err_best)),
        "err_p90": float(np.percentile(err_best, 90)),
        "mismatch_frac": float((err_best > 0.12).mean()),
        "view_err_median": float(np.median(min_err[valid])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--texture", required=True, help="第一张贴图")
    ap.add_argument("--texture2", default="", help="第二张贴图（同一几何/UV 下对比）")
    ap.add_argument("--samples", type=int, default=40000)
    args = ap.parse_args()

    rows = []
    for t in [args.texture] + ([args.texture2] if args.texture2 else []):
        r = evaluate(args.job, Path(t), args.samples)
        rows.append(r)
        print(f"  {r['texture']:<40} n={r['samples']:>6}  "
              f"err中位={r['err_median']:.4f}  p90={r['err_p90']:.4f}  "
              f"明显不符={r['mismatch_frac']*100:.1f}%  与任意视角最小差中位={r['view_err_median']:.4f}")
    if len(rows) == 2:
        d = rows[0]["err_median"] - rows[1]["err_median"]
        print(f"\n  中位误差变化：{d:+.4f}（负值=第二张更一致）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
