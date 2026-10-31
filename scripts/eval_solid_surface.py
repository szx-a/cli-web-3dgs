"""一站式评测"表面是否实心 + 视角是否脆弱 + 感知质量"。

用于对比 `solid-v3`（旧）与 `solid2`（去掉不透明度下限 + 剪枝 + 抗锯齿）：
1. 不透明度分布（是否从"全卡在 0.10"变成双峰）；
2. 视角鲁棒性：偏 0/5/10/20° 时的清晰度（这是"preview 清楚、交互糊"的直接度量）；
3. LPIPS / PSNR（论文对齐实现，8 个留出视角）。

用法：
  python scripts/eval_solid_surface.py img5-solid-v3 img5-solid2
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\3dgs-and-ai")
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from gsflow.gsdata.colmap_io import read_model  # noqa: E402
from gsflow.gsdata.mesh_export import scale_intrinsics  # noqa: E402
from gsflow.gsdata.shio import read_gaussians_full, render_view  # noqa: E402
from gsflow.paths import apply_runtime_env  # noqa: E402
from gsflow.trainers.perceptual import get_lpips  # noqa: E402

apply_runtime_env()

EVAL_DATA = ROOT / "workspace" / "img5-quality4" / "04_train" / "data"
W = H = 512
OFFS = (0.0, 5.0, 10.0, 20.0)


def latest_ply(jd: Path) -> Path:
    return sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
                  key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))[-1]


def sharp(x: np.ndarray) -> float:
    g = x.mean(axis=2)
    return float(np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean())


def rodrigues(axis: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], float)
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


_CENTER_CACHE: dict[int, np.ndarray] = {}


def _center_of(g) -> np.ndarray:
    key = id(g)
    if key not in _CENTER_CACHE:
        xyz = np.asarray(g["xyz"], np.float64)
        _CENTER_CACHE[key] = (xyz.min(0) + xyz.max(0)) / 2.0
    return _CENTER_CACHE[key]


def render_offset(g, rec, im, d_deg: float) -> np.ndarray:
    """绕物体中心**公转** d_deg（相机位置与朝向一起变），距离保持不变。

    ⚠ 早期版本只把朝向绕自身 up 轴转了 d 度、位置不动 —— 10° 以上物体就转出画面，
    测出来的"清晰度暴跌"是假象（见 docs/03 G176）。
    """
    cam = rec.cameras[im.camera_id]
    R, c = im.cam_to_world()
    R = np.asarray(R, float)
    c = np.asarray(c, float)
    center = _center_of(g)
    up = R[:, 1] / max(np.linalg.norm(R[:, 1]), 1e-9)
    Rrot = rodrigues(up, d_deg)
    c_new = center + Rrot @ (c - center)
    fwd = center - c_new
    fwd /= max(np.linalg.norm(fwd), 1e-12)
    right = np.cross(fwd, up)
    right /= max(np.linalg.norm(right), 1e-12)
    up2 = np.cross(right, fwd)
    w2c = np.eye(4)
    w2c[0, :3], w2c[1, :3], w2c[2, :3] = right, up2, fwd
    w2c[:3, 3] = -w2c[:3, :3] @ c_new
    K = scale_intrinsics(np.asarray(cam.intrinsics(), float), (cam.width, cam.height), (W, H))
    col, _ = render_view(g, w2c, K, W, H)
    return np.clip(np.asarray(col.detach().cpu() if hasattr(col, "detach") else col)[..., :3], 0, 1)


def main() -> int:
    fn = get_lpips("cuda", input_size=224)
    jobs = sys.argv[1:] or ["img5-solid-v3"]
    for job in jobs:
        jd = ROOT / "workspace" / job
        if not jd.exists():
            print(f"  {job}: \u4e0d\u5b58\u5728")
            continue
        g = read_gaussians_full(latest_ply(jd))
        op = np.asarray(g["opacity"], np.float64).reshape(-1)
        if op.max() > 1.5:
            op = 1.0 / (1.0 + np.exp(-op))
        print(f"\n  === {job} ===")
        print(f"  n={len(op):>7}  op p10={np.percentile(op,10):.3f} p50={np.median(op):.3f} "
              f"p90={np.percentile(op,90):.3f}  >0.5={float((op>0.5).mean())*100:4.1f}%  "
              f">0.9={float((op>0.9).mean())*100:4.1f}%")

        # 视角鲁棒性
        sp = jd / "04_train" / "data" / "sparse"
        rec = read_model(sp / "0" if (sp / "0" / "cameras.bin").exists() else sp)
        imgs = sorted(rec.images, key=lambda im: im.name)
        picks = [imgs[i] for i in np.linspace(0, len(imgs) - 1, 3).astype(int).tolist()]
        base = {id(im): render_offset(g, rec, im, 0.0) for im in picks}
        row = []
        for off in OFFS:
            vals = [sharp(render_offset(g, rec, im, off)) for im in picks]
            row.append((off, float(np.mean(vals))))
        ref = row[0][1]
        print("  \u89c6\u89d2\u9c81\u68d2\u6027\uff08\u6e05\u6670\u5ea6\uff09: " + "  ".join(
            f"{o:.0f}\u00b0={v:.5f}({v/ref*100:3.0f}%)" for o, v in row))

        # LPIPS / PSNR（8 个留出视角）
        sp2 = EVAL_DATA / "sparse"
        rec2 = read_model(sp2 / "0" if (sp2 / "0" / "cameras.bin").exists() else sp2)
        imgs2 = sorted(rec2.images, key=lambda im: im.name)
        picks2 = np.linspace(0, len(imgs2) - 1, min(8, len(imgs2))).astype(int).tolist()
        vals, psnrs = [], []
        for k in picks2:
            im = imgs2[k]
            cam = rec2.cameras[im.camera_id]
            R, c = im.cam_to_world()
            R = np.asarray(R, float)
            c = np.asarray(c, float)
            w2c = np.eye(4)
            w2c[:3, :3] = R.T
            w2c[:3, 3] = -R.T @ c
            K = np.asarray(cam.intrinsics(), float)
            col, _ = render_view(g, w2c, K, cam.width, cam.height)
            col_t = torch.as_tensor(np.asarray(
                col.detach().cpu() if hasattr(col, "detach") else col)[..., :3],
                dtype=torch.float32, device="cuda").clamp(0, 1)
            with Image.open(EVAL_DATA / "images" / im.name) as gi:
                gt = torch.as_tensor(np.asarray(gi.convert("RGB"), dtype=np.float32) / 255.0,
                                     device="cuda")
            if gt.shape[:2] != col_t.shape[:2]:
                continue
            with torch.no_grad():
                vals.append(float(fn(col_t, gt)))
            mp = EVAL_DATA / "masks" / (im.name.split(".")[0] + ".png")
            if mp.exists():
                m = np.asarray(Image.open(mp).convert("L")) > 127
                if m.shape == gt.shape[:2] and m.any():
                    ca = col_t.detach().cpu().numpy().astype(np.float64)
                    ga = gt.detach().cpu().numpy().astype(np.float64)
                    mse = float(((ca - ga) ** 2)[m].mean())
                    psnrs.append(10 * np.log10(1.0 / max(mse, 1e-10)))
        print(f"  LPIPS={np.mean(vals):.4f}  PSNR(\u524d\u666f\u63a9\u7801)={np.mean(psnrs):.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
