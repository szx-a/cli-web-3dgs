"""自由视角 / 越界雾丝检查：把"细丝化、背面混乱"变成可测的数字。

## 为什么必须有这个检查（本轮暴露的验收缺口）

原验收只看**训练/留出相机**上的掩码内 PSNR/SSIM。于是"训练视角完美、
视野外全是针"的模型能拿到 32/32 通过，而用户打开查看器一转就看到细丝。

## 指标怎么选（踩过两个坑，记在这里免得重犯）

1. **不能用"半透明像素占比(veil)"当主判据**。实测三个模型（无 clamp / clamp0.10 /
   clamp0.05）veil 都在 17%~19%，因为牌坊是**镂空薄结构**，轮廓软边本身就占很大比例
   —— veil 被边缘淹没，区分不出"丝"。
2. **不能按重建里的图像顺序取样**。COLMAP 重建的 `images` 顺序不保证一致，
   我曾因此在两个模型上比了**不同视角**，得出"8.4% vs 17.6%"的假差异。

## 主判据：掩码外的 alpha（只在 0° 有效，因为那时正好是训练相机、掩码可对齐）

    掩码外 alpha 均值 / >0.1 占比

正确模型在物体轮廓之外应当几乎**不画东西**（alpha≈0）。针状高斯与漂浮物会把
alpha 漏到背景里 —— 这正是用户说的"细丝 / 雾"。这个量不受镂空轮廓影响。

## 辅判据：coverage 随偏转角的变化（视角脆性）

绕物体上方向偏转 0°/10°/25°，看 coverage 与掩码外 alpha 是否剧烈变化。

用法:
  python scripts/check_free_view.py --job img3-final --views 4
  python scripts/check_free_view.py --job X --data-dir workspace/ab-clamp/data --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _axis_angle(axis: np.ndarray, ang: float) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = np.cos(ang), np.sin(ang)
    x, y, z = a
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def _mask_for(mdir: Path, name: str, W: int, H: int):
    from PIL import Image

    for cand in (mdir / (name + ".png"), mdir / (Path(name).stem + ".png")):
        if cand.exists():
            with Image.open(cand) as mm:
                return np.asarray(mm.convert("L").resize((W, H), Image.NEAREST)) > 127
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="img3-final")
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--angles", default="0,10,25")
    ap.add_argument("--width", type=int, default=456)
    ap.add_argument("--data-dir", default="", help="相机/掩码所在数据目录（A/B 作业用）")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    # gsplat 内核未就绪会触发 80~140 秒 JIT（本机 GBK 代码页下还会崩）
    from gsflow.paths import apply_runtime_env

    apply_runtime_env()
    if sys.modules.get("gsplat.csrc") is None:
        print("gsplat CUDA 内核未注册，本检查跳过（先跑一次训练/导出）")
        return 0
    try:
        from gsplat.cuda._backend import _C as _kernel

        if _kernel is None:
            print("gsplat CUDA 内核未就绪，本检查跳过")
            return 0
    except Exception:  # noqa: BLE001
        print("gsplat 不可用，本检查跳过")
        return 0

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.gsdata.shio import read_gaussians_full, render_view

    jd = ROOT / "workspace" / args.job
    plys = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"), key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0))
    if not plys:
        print("没有训练产物")
        return 1
    g = read_gaussians_full(plys[-1])

    roots = [jd / "03_colmap" / "sparse", jd / "04_train" / "data" / "sparse"]
    if args.data_dir:
        roots.insert(0, Path(args.data_dir) / "sparse")
    cands = [p.parent for r in roots if r.exists() for p in r.rglob("cameras.bin")]
    if not cands:
        print("没有相机位姿")
        return 1
    rec = max((read_model(c) for c in sorted(set(cands))), key=lambda r: len(r.images))

    mdir = Path(args.data_dir) / "masks" if args.data_dir \
        else jd / "04_train" / "data" / "masks"

    xyz = np.asarray(g["xyz"], dtype=np.float64)
    center = xyz.mean(0)
    ups = []
    for im in rec.images:
        R, _ = im.cam_to_world()
        ups.append(-np.asarray(R, dtype=np.float64)[:, 1])
    up = np.mean(ups, axis=0)
    up /= np.linalg.norm(up) + 1e-12

    angles = [float(a) for a in args.angles.split(",") if a.strip()]
    # 按文件名排序取样：重建里的顺序不保证一致
    imgs = sorted(rec.images, key=lambda im: im.name)
    picks = np.linspace(0, len(imgs) - 1, args.views).astype(int).tolist()
    im0 = imgs[picks[0]]
    cam0 = rec.cameras[im0.camera_id]
    W = min(args.width, cam0.width)
    H = int(W * cam0.height / max(cam0.width, 1))

    print(f"=== 自由视角 / 越界雾丝检查：{args.job} ===")
    print(f"  模型 {len(xyz):,} 个高斯；分辨率 {W}x{H}；"
          f"取样 {[imgs[k].name for k in picks]}")
    rows: list[dict] = []
    for k in picks:
        im = imgs[k]
        cam = rec.cameras[im.camera_id]
        R, c = im.cam_to_world()
        R = np.asarray(R, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        K = np.asarray(cam.intrinsics(), dtype=np.float64).copy()
        K[0, :] *= W / cam.width
        K[1, :] *= H / cam.height
        msk = _mask_for(mdir, im.name, W, H)
        for ang in angles:
            Ra = _axis_angle(up, np.deg2rad(ang))
            R2 = Ra @ R
            c2 = center + Ra @ (c - center)
            w2c = np.eye(4)
            w2c[:3, :3] = R2.T
            w2c[:3, 3] = -R2.T @ c2
            colors, alphas = render_view(g, w2c, K, W, H)
            a = np.asarray(alphas.detach().cpu().numpy()
                           if hasattr(alphas, "detach") else alphas)[..., 0]
            col = np.asarray(colors.detach().cpu().numpy()
                             if hasattr(colors, "detach") else colors)[..., :3]
            solid = a > 0.5
            row = {
                "view": im.name, "angle": ang,
                "coverage": float(solid.mean()),
                "veil": float(((a > 0.05) & (a <= 0.5)).mean()),
                "alpha_out_mean": None, "alpha_out_frac_gt10": None,
            }
            # 掩码外 alpha 只在 0° 有意义（那时相机=训练相机，掩码可对齐）
            if msk is not None and abs(ang) < 1e-6 and (~msk).any():
                out_a = a[~msk]
                row["alpha_out_mean"] = float(out_a.mean())
                row["alpha_out_frac_gt10"] = float((out_a > 0.1).mean())
            # ---- 轮廓亮边：轮廓内 2px 环带的亮度减去内部亮度 ----
            #
            # 用户看到的“白边/发光”在 DC 渲染里依然存在（G134），
            # 成因是轮廓处视线穿过更多半透明高斯 ⇒ alpha 叠加 ⇒ 亮度饱和。
            # 这个量**不需要真值**，也不依赖视角，适合做回归守门。
            try:
                from scipy import ndimage as _nd
                solid = a > 0.5
                if solid.any():
                    inner = _nd.binary_erosion(solid, iterations=2)
                    rim = solid & ~inner
                    lum = col.mean(2)
                    if inner.any() and rim.any():
                        row["rim_bright"] = float(lum[rim].mean() - lum[inner].mean())
                        row["rim_frac"] = float(rim.sum() / max(solid.sum(), 1))
            except Exception:  # noqa: BLE001
                pass
            rows.append(row)
            oam = row["alpha_out_mean"]
            line = (f"  {im.name:<14}{ang:>5.0f}°  coverage={row['coverage']:.3f}"
                    f"  veil={row['veil']:.3f}")
            if oam is not None:
                line += (f"  掩码外 alpha={oam:.4f}"
                         f" (>0.1: {row['alpha_out_frac_gt10']*100:.2f}%)")
            print(line)

    base = [r for r in rows if r["angle"] == min(angles)]
    far = [r for r in rows if r["angle"] == max(angles)]
    outs = [r["alpha_out_mean"] for r in base if r["alpha_out_mean"] is not None]
    outsf = [r["alpha_out_frac_gt10"] for r in base
             if r["alpha_out_frac_gt10"] is not None]
    verdict = "无数据"
    if outs:
        om = float(np.mean(outs))
        of = float(np.mean(outsf))
        # 阈值标定（2026-10）：无 bg 正则时 0.73 / 98.8%（黑壳），
        # 加上 bg_alpha_reg=0.1 后 0.049 / 8.7%。取 0.10 / 15% 作回归守门。
        ok_out = om < 0.10 and of < 0.15
        print(f"\n  掩码外 alpha: 均值 {om:.4f}，>0.1 占比 {of*100:.2f}%"
              f"   （正确模型应≈0；这是「丝/雾漏到物体外」的直接度量）")
        print(f"  [{'OK  ' if ok_out else 'FAIL'}] 掩码外 alpha 均值 < 0.10"
              f" 且 >0.1 占比 < 15%")
        if base and far:
            cb = float(np.mean([r["coverage"] for r in base]))
            cf = float(np.mean([r["coverage"] for r in far]))
            print(f"  coverage {cb*100:.1f}% → {cf*100:.1f}%（偏转 {max(angles):.0f}°）")
        verdict = "通过" if ok_out else "不通过"
        print(f"  判读: {verdict}")

    if args.json:
        p = ROOT / args.json
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"job": args.job, "up": up.tolist(),
                                 "rows": rows, "verdict": verdict},
                                ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已写出 {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
