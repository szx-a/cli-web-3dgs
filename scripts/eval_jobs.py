"""一个命令出一张表：高斯数 / 尺度 / 不透明度 / LPIPS / PSNR / 1×·2×·3× 清晰度。

用于横向比较多个作业（solid-v3、long、dense…），结论直接贴文档。

    python scripts/eval_jobs.py img5-solid-v3 img5-long img5-dense
"""
from __future__ import annotations

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


def latest_ply(jd: Path) -> Path:
    # ⚠ 必须按**修改时间**取最新：实验作业是从旧作业整目录复制的，
    # 里面带着旧的 iteration_60000；按序号排会拿到那个旧检查点
    # （实测 img5-dense 曾因此评出与 solid-v3 完全相同的数字）。
    return max((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
               key=lambda p: p.stat().st_mtime)


def sharp(x: np.ndarray) -> float:
    a = x.mean(axis=2)
    return float(np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean())


def main() -> int:
    fn = get_lpips("cuda", input_size=224)
    rows = []
    for job in sys.argv[1:]:
        jd = ROOT / "workspace" / job
        if not (jd / "04_train" / "point_cloud").exists():
            print(f"  {job}: \u7f3a\u5c11\u68c0\u67e5\u70b9")
            continue
        g = read_gaussians_full(latest_ply(jd))
        xyz = np.asarray(g["xyz"], np.float64)
        sc = np.asarray(g["scales"], np.float64).reshape(-1, 3)
        op = np.asarray(g["opacity"], np.float64).reshape(-1)
        if op.max() > 1.5:
            op = 1.0 / (1.0 + np.exp(-op))
        # ⚠ 用 1%/99% 分位数估中心：MCMC 的噪声会留下远处浮点，
        # min/max 中心会被拉偏 ⇒ 近距离相机看到空处（实测 img5-mcmc 全黑）。
        lo = np.percentile(xyz, 1.0, axis=0)
        hi = np.percentile(xyz, 99.0, axis=0)
        center = (lo + hi) / 2.0

        # ---- \u8fd1\u8ddd\u79bb\u6e05\u6670\u5ea6\uff08\u540c\u4e00\u76f8\u673a\uff09----
        sp = jd / "04_train" / "data" / "sparse"
        rec = read_model(sp / "0" if (sp / "0" / "cameras.bin").exists() else sp)
        im = sorted(rec.images, key=lambda x: x.name)[0]
        cam = rec.cameras[im.camera_id]
        R, c = im.cam_to_world()
        R = np.asarray(R, float)
        c = np.asarray(c, float)
        up = R[:, 1] / np.linalg.norm(R[:, 1])
        K = scale_intrinsics(np.asarray(cam.intrinsics(), float), (cam.width, cam.height), (W, H))
        sharps = []
        for div in (1.0, 2.0, 3.0):
            c_new = center + (c - center) / div
            fwd = center - c_new
            fwd /= np.linalg.norm(fwd)
            right = np.cross(fwd, up)
            right /= np.linalg.norm(right)
            up2 = np.cross(right, fwd)
            w2c = np.eye(4)
            w2c[0, :3], w2c[1, :3], w2c[2, :3] = right, up2, fwd
            w2c[:3, 3] = -w2c[:3, :3] @ c_new
            col, _ = render_view(g, w2c, K, W, H)
            sharps.append(sharp(np.clip(np.asarray(
                col.detach().cpu() if hasattr(col, "detach") else col)[..., :3], 0, 1)))

        # ---- LPIPS / PSNR\uff088 \u4e2a\u7559\u51fa\u89c6\u56fe\uff09----
        sp2 = EVAL_DATA / "sparse"
        rec2 = read_model(sp2 / "0" if (sp2 / "0" / "cameras.bin").exists() else sp2)
        imgs2 = sorted(rec2.images, key=lambda x: x.name)
        picks = np.linspace(0, len(imgs2) - 1, min(8, len(imgs2))).astype(int).tolist()
        vals, psnrs = [], []
        for k in picks:
            im2 = imgs2[k]
            cam2 = rec2.cameras[im2.camera_id]
            R2, c2 = im2.cam_to_world()
            R2 = np.asarray(R2, float)
            c2 = np.asarray(c2, float)
            w2c = np.eye(4)
            w2c[:3, :3] = R2.T
            w2c[:3, 3] = -R2.T @ c2
            K2 = np.asarray(cam2.intrinsics(), float)
            col, _ = render_view(g, w2c, K2, cam2.width, cam2.height)
            col_t = torch.as_tensor(np.asarray(
                col.detach().cpu() if hasattr(col, "detach") else col)[..., :3],
                dtype=torch.float32, device="cuda").clamp(0, 1)
            with Image.open(EVAL_DATA / "images" / im2.name) as gi:
                gt = torch.as_tensor(np.asarray(gi.convert("RGB"), dtype=np.float32) / 255.0,
                                     device="cuda")
            if gt.shape[:2] != col_t.shape[:2]:
                continue
            with torch.no_grad():
                vals.append(float(fn(col_t, gt)))
            mp = EVAL_DATA / "masks" / (im2.name.split(".")[0] + ".png")
            if mp.exists():
                m = np.asarray(Image.open(mp).convert("L")) > 127
                if m.shape == gt.shape[:2] and m.any():
                    ca = col_t.detach().cpu().numpy().astype(np.float64)
                    ga = gt.detach().cpu().numpy().astype(np.float64)
                    psnrs.append(10 * np.log10(1.0 / max(float(((ca - ga) ** 2)[m].mean()), 1e-10)))
        rows.append({
            "job": job, "n": len(xyz), "scale": float(np.median(sc.max(1))),
            "op": float(np.median(op)), "lpips": float(np.mean(vals)),
            "psnr": float(np.mean(psnrs)), "s1": sharps[0], "s2": sharps[1], "s3": sharps[2],
        })

    print("\n  | 作业 | 高斯数 | 尺度中位 | op 中位 | LPIPS | PSNR(m) | 1× | 2× | 3× | 3×/1× |")
    print("  |---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"  | {r['job']} | {r['n']:,} | {r['scale']:.4f} | {r['op']:.3f} | "
              f"{r['lpips']:.4f} | {r['psnr']:.2f} | {r['s1']:.5f} | {r['s2']:.5f} | "
              f"{r['s3']:.5f} | {r['s3']/max(r['s1'],1e-9)*100:.0f}% |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
