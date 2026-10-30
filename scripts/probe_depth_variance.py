"""量化沿射线的深度方差（"朦胧"的直接度量）。

做法与训练侧一致：把逐高斯相机深度 z 与 z² 当颜色通道渲染，得到
Σ w z、Σ w z² 与 Σ w，于是 E[z] = Σwz/Σw，Var[z] = E[z²] - E[z]²。

输出（掩码内）：
* `var`      方差中位数（scene 单位²）
* `spread`   标准差的中位数（scene 单位）—— 直观的"表面厚度"
* `p90`      90 分位方差（看尾部有多散）
* `T`        累计不透明度中位数（对照：它不是问题所在）

用法：
  python scripts/probe_depth_variance.py --jobs ab-dv2-base0-r1,ab-dv2-dvarA-r1 --views 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from gsplat import rasterization  # noqa: E402
from PIL import Image  # noqa: E402

from gsflow.gsdata.colmap_io import read_model  # noqa: E402
from gsflow.gsdata.mesh_export import scale_intrinsics  # noqa: E402
from gsflow.gsdata.shio import read_gaussians_full  # noqa: E402
from gsflow.paths import apply_runtime_env  # noqa: E402
from gsflow.trainers.depth_regularizers import (  # noqa: E402
    camera_depths,
    expected_depth_stats,
    moment_colors,
    DepthMoments,
)


def _ply(job_dir: Path) -> Path:
    cands = sorted((job_dir / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
                   key=lambda p: int("".join(c for c in p.parent.name if c.isdigit()) or 0))
    if not cands:
        raise SystemExit(f"{job_dir} 没有 point_cloud.ply")
    return cands[-1]


def probe(job: str, views: int, data_from: str = "") -> dict:
    job_dir = ROOT / "workspace" / job
    if data_from:
        data = ROOT / "workspace" / data_from / "04_train" / "data"
    else:
        data = job_dir / "04_train" / "data"
    g = read_gaussians_full(_ply(job_dir))
    sp = data / "sparse"
    rec = read_model(sp / "0" if (sp / "0" / "cameras.bin").exists() else sp)
    imgs = sorted(rec.images, key=lambda im: im.name)
    picks = np.linspace(0, len(imgs) - 1, max(1, views)).astype(int).tolist()

    means = torch.as_tensor(np.asarray(g["xyz"]), dtype=torch.float32, device="cuda")
    quats = torch.as_tensor(np.asarray(g["quats"]), dtype=torch.float32, device="cuda")
    scales = torch.as_tensor(np.asarray(g["scales"]), dtype=torch.float32, device="cuda")
    opac = torch.as_tensor(np.asarray(g["opacity"]).reshape(-1), dtype=torch.float32, device="cuda")
    shs = torch.as_tensor(np.asarray(g["shs"]), dtype=torch.float32, device="cuda")
    sh_deg = int(g["sh_degree"])

    vars_, Ts = [], []
    for k in picks:
        im = imgs[k]
        cam = rec.cameras[im.camera_id]
        W, H = cam.width, cam.height
        R, c = im.cam_to_world()
        R = np.asarray(R, float)
        c = np.asarray(c, float)
        K = np.asarray(cam.intrinsics(), float)
        w2c = np.eye(4)
        w2c[:3, :3] = R.T
        w2c[:3, 3] = -R.T @ c
        vm = torch.as_tensor(w2c, dtype=torch.float32, device="cuda")
        kk = torch.as_tensor(scale_intrinsics(K, (W, H), (W, H)), dtype=torch.float32, device="cuda")
        with torch.no_grad():
            zc = camera_depths(means, vm)
            cols, al, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opac.clamp(0, 1),
                colors=moment_colors(zc), viewmats=vm[None], Ks=kk[None],
                width=W, height=H, sh_degree=None, near_plane=0.01, far_plane=1e10,
                packed=False, render_mode="RGB",
            )
            m = DepthMoments(weight=al[0, ..., 0], first=cols[0, ..., 0], second=cols[0, ..., 1])
            _, var = expected_depth_stats(m)
        var = var.cpu().numpy()
        tt = m.weight.cpu().numpy()
        mp = data / "masks" / (im.name.split(".")[0] + ".png")
        if mp.exists():
            msk = np.asarray(Image.open(mp).convert("L")) > 127
            if msk.shape == var.shape:
                vars_.append(var[msk])
                Ts.append(tt[msk])
                continue
        vars_.append(var.reshape(-1))
        Ts.append(tt.reshape(-1))

    v = np.concatenate(vars_)
    T = np.concatenate(Ts)
    return {
        "job": job,
        "var_median": float(np.median(v)),
        "spread_median": float(np.sqrt(np.median(v))),
        "var_p90": float(np.percentile(v, 90)),
        "T_median": float(np.median(T)),
        "n_px": int(v.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="逗号分隔的作业 id")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--data-from", default="", help="从该作业取相机（harness 作业不拷贝 data）")
    ap.add_argument("--json", default="", help="把结果写到该 JSON")
    args = ap.parse_args()

    apply_runtime_env()
    if not torch.cuda.is_available():
        print("需要 CUDA")
        return 2

    out = []
    print(f"{'作业':<24}{'方差中位':>10}{'表面厚度':>10}{'p90 方差':>10}{'T 中位':>8}")
    for job in [j.strip() for j in args.jobs.split(",") if j.strip()]:
        r = probe(job, args.views, args.data_from)
        out.append(r)
        print(f"{r['job']:<24}{r['var_median']:>10.5f}{r['spread_median']:>10.4f}"
              f"{r['var_p90']:>10.5f}{r['T_median']:>8.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已写 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
