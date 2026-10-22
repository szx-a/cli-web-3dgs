"""检验 `f_rest` 的**布局假设**：通道主序 vs 基主序。

## 现象

同一份 PLY，用 gsplat 渲染时：

| sh_degree | 掩码内 PSNR | 渲染亮度（真值 0.533）|
|---|---|---|
| 0（只用 DC）| **18.38** | 0.538 ✓ |
| 1 | 16.83 | 0.583 |
| 2 | 14.23 | 0.641 |
| 3 | **12.21** | 0.693 ✗ |

DC 正确、每加一阶都更差、亮度单调上升 ⇒ 高阶系数**被乘以了错误的基函数**。
最常见的两种错法：
1. **通道主序 vs 基主序**（`[R0..R14,G0..G14,B0..B14]` vs `[R0,G0,B0,R1,G1,B1,...]`）；
2. 系数在基内的**顺序**（SH 基按 m 的排列不同）。

本脚本枚举候选布局，用 PSNR 判定哪一种能恢复画质。

用法: python scripts/probe_sh_rest_layout.py --job img3-final --view 0
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
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--width", type=int, default=456)
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env

    apply_runtime_env()
    from PIL import Image
    from plyfile import PlyData

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.gsdata.shio import read_gaussians_full, render_view

    jd = ROOT / "workspace" / args.job
    data = jd / "04_train" / "data"
    ply = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"), key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0))[-1]
    g = read_gaussians_full(ply)

    # 直接读原始 f_rest（未经过 shio 的解释）+ 原始 opacity，用来自己重排
    pd = PlyData.read(str(ply))
    v = pd["vertex"].data
    names = list(v.dtype.names)
    rest_names = sorted((n for n in names if n.startswith("f_rest_")),
                        key=lambda s: int(s.split("_")[-1]))
    f_rest = np.stack([v[n] for n in rest_names], axis=1).astype(np.float32)  # [N,45]
    op_raw = np.asarray(g["opacity"], dtype=np.float32).reshape(-1)
    # 对齐顺序：read_gaussians_full 会按不透明度排序，这里用同一套排序
    order = np.argsort(-np.clip(op_raw, 0, 1), kind="stable")
    f_rest = f_rest[order]
    n = f_rest.shape[0]

    rec = max((read_model(c) for c in sorted(
        {p.parent for p in (data / "sparse").rglob("cameras.bin")})),
        key=lambda r: len(r.images))
    imgs = sorted(rec.images, key=lambda im: im.name)
    im = imgs[args.view]
    cam = rec.cameras[im.camera_id]
    W = min(args.width, cam.width)
    H = int(W * cam.height / cam.width)
    R, c = im.cam_to_world()
    R = np.asarray(R, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    K = np.asarray(cam.intrinsics(), dtype=np.float64).copy()
    K[0, :] *= W / cam.width
    K[1, :] *= H / cam.height
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ c
    for cand in (data / "images" / im.name,
                 data / "images" / (Path(im.name).stem + ".png")):
        if cand.exists():
            gt_p = cand
            break
    with Image.open(gt_p) as gi:
        gt = np.asarray(gi.convert("RGB").resize((W, H), Image.BILINEAR),
                        dtype=np.float64) / 255.0
    for cand in (data / "masks" / (im.name + ".png"),
                 data / "masks" / (Path(im.name).stem + ".png")):
        if cand.exists():
            mp = cand
            break
    with Image.open(mp) as mi:
        m = np.asarray(mi.convert("L").resize((W, H), Image.NEAREST)) > 127

    dc = np.asarray(g["shs"])[:, 0, :]          # [N,3] 已按同一顺序

    def build(rest_chw: np.ndarray) -> np.ndarray:
        """rest_chw: [N,3,15]（通道主序的规范形状）→ shs [N,16,3]"""
        shs = np.zeros((n, 16, 3), dtype=np.float32)
        shs[:, 0, :] = dc
        shs[:, 1:, :] = rest_chw.transpose(0, 2, 1)
        return shs

    cands = {}
    # A：通道主序（本项目读写两端都假设这个）
    cands["A 通道主序"] = build(f_rest.reshape(n, 3, 15))
    # B：基主序（每 3 个一组是 R,G,B）
    cands["B 基主序"] = build(f_rest.reshape(n, 15, 3).transpose(0, 2, 1))
    # C：通道主序 + 基内逆序（排除 m 排列方向问题）
    cands["C 通道+基内逆序"] = build(f_rest.reshape(n, 3, 15)[:, :, ::-1])
    # D：全零 rest（应等于 DC-only，作为对照）
    cands["D rest 全零"] = build(np.zeros((n, 3, 15), dtype=np.float32))

    print(f"=== f_rest 布局假设检验：{args.job} / {im.name} ===")
    print(f"  {'假设':<18}{'PSNR(掩码内)':>14}{'渲染亮度':>10}")
    for name, shs in cands.items():
        gg = dict(g)
        gg["shs"] = shs
        gg["sh_degree"] = 3
        colors, alphas = render_view(gg, w2c, K, W, H)
        col = np.asarray(colors.detach().cpu().numpy()
                         if hasattr(colors, "detach") else colors)[..., :3]
        vv = np.clip(col, 0, 1)
        ps = 10 * np.log10(1 / max(float(((vv - gt) ** 2)[m].mean()), 1e-10))
        print(f"  {name:<18}{ps:>14.2f}{vv[m].mean():>10.3f}")
    print("  真值亮度均值 %.3f" % gt[m].mean())
    print("\n  判读：哪个假设的 PSNR 明显最高，就是 PLY 里 f_rest 的真实布局。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
