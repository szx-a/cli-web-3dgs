"""诊断：找出正确的 COLMAP 位姿约定，让高斯真正落进相机视锥。

现象：训练时 `alpha 和 = 0`、`means.grad = 0` —— 没有任何高斯被渲染出来，
自适应密度控制因此完全不触发（模型永远停在初始稀疏点）。

怀疑：`cam_to_world` 的坐标轴翻转约定不对（COLMAP/OpenCV 与 OpenGL 之间），
导致点云被放到相机背后或被 near/far 裁掉。

做法：对 4 种 y/z 翻转组合分别构造 viewmat，统计 alpha 和与位置梯度，
谁最大谁就是正确约定。

用法：
    scripts\\gsflow_gpu.cmd
    env\\venv\\Scripts\\python.exe scripts\\probe_pose_convention.py --job <job_id>
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow import paths  # noqa: E402

paths.apply_runtime_env()

import torch  # noqa: E402
from gsplat import rasterization  # noqa: E402

from gsflow.gsdata.colmap_io import read_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="20261002-222324-eaf09b", help="工作区里的作业 id")
    ap.add_argument("--resolution", type=int, default=400)
    args = ap.parse_args()

    data_dir = ROOT / "workspace" / args.job / "04_train" / "data"
    model_dir = data_dir / "sparse" / "0"
    if not (model_dir / "cameras.bin").exists():
        alt = data_dir / "sparse"
        model_dir = alt if (alt / "cameras.bin").exists() else model_dir
    print(f"模型目录: {model_dir}")
    rec = read_model(model_dir)
    print(f"已注册图像 {len(rec.images)} 张，稀疏点 {len(rec.points3D)}")

    if not len(rec.images):
        print("没有注册图像，先跑 colmap 阶段")
        return 1

    pts = rec.points3D.astype(np.float32)
    cols = rec.colors3D.astype(np.float32) / 255.0

    # 图像尺寸（用于 K）
    from PIL import Image

    img0 = sorted((data_dir / "images").glob("*"))[0]
    with Image.open(img0) as im:
        W0, H0 = im.size
    scale = args.resolution / max(W0, H0) if args.resolution else 1.0
    W, H = int(W0 * scale), int(H0 * scale)

    print(f"图像: {img0.name} {W0}x{H0} -> 用于测试 {W}x{H}")
    print(f"点云范围: min={pts.min(0)} max={pts.max(0)}")

    means = torch.nn.Parameter(torch.from_numpy(pts).to("cuda"))
    n = means.shape[0]
    dist = torch.clamp(torch.sqrt(((means - means.mean(0)) ** 2).sum(1, keepdim=True)), 1e-3, None)
    scales = torch.nn.Parameter(torch.log(dist * 0.5).repeat(1, 3))
    quats = torch.nn.Parameter(torch.zeros(n, 4, device="cuda"))
    quats.data[:, 0] = 1.0
    opac = torch.nn.Parameter(torch.logit(torch.full((n, 1), 0.1, device="cuda")))
    shs = torch.nn.Parameter(torch.zeros(n, 1, 3, device="cuda"))
    shs.data[:, 0, :] = (torch.from_numpy(cols).to("cuda") - 0.5) / 0.28209479177387814

    print("\n=== 扫描 y/z 翻转组合（4 种）===")
    header = f"{'flip':<22}{'cam_z 均值':>12}{'alpha 和':>12}{'可见像素':>10}{'means 梯度':>14}"
    print(header)
    print("-" * len(header))

    results = []
    for flip_y, flip_z in itertools.product((1.0, -1.0), repeat=2):
        c2w_list, K_list = [], []
        for im in sorted(rec.images, key=lambda x: x.name)[:6]:
            cam = rec.cameras[im.camera_id]
            K = cam.intrinsics().copy()
            K[0, 0] *= scale
            K[1, 1] *= scale
            K[0, 2] *= scale
            K[1, 2] *= scale
            R, t = im.cam_to_world()
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = t
            c2w[0:3, 1] *= flip_y
            c2w[0:3, 2] *= flip_z
            c2w_list.append(c2w)
            K_list.append(K)

        c2ws = torch.from_numpy(np.stack(c2w_list)).float().cuda()
        viewmats = torch.linalg.inv(c2ws)
        Ks = torch.from_numpy(np.stack(K_list)).float().cuda()

        # 相机空间里的点深度（相机 z 轴朝前时为正）
        pts_h = torch.cat([torch.from_numpy(pts).cuda(), torch.ones(len(pts), 1, device="cuda")], dim=1)
        cam_pts = (viewmats[0] @ pts_h.T).T
        cam_z_mean = float(cam_pts[:, 2].mean())
        in_front = int((cam_pts[:, 2] > 0).sum())

        means.grad = None
        colors, alphas, _ = rasterization(
            means=means, quats=quats / (quats.norm(dim=-1, keepdim=True) + 1e-8),
            scales=torch.exp(scales), opacities=torch.sigmoid(opac).squeeze(-1),
            colors=shs, viewmats=viewmats[0][None], Ks=Ks[0][None],
            width=W, height=H, sh_degree=0, packed=False, render_mode="RGB",
        )
        loss = colors[0].abs().mean()
        loss.backward()
        grad = float(means.grad.abs().sum()) if means.grad is not None else -1.0
        alpha_sum = float(alphas.sum())
        visible = int((alphas[0] > 0.01).sum())

        tag = f"y*{flip_y:+.0f} z*{flip_z:+.0f}"
        print(f"{tag:<22}{cam_z_mean:>12.3f}{alpha_sum:>12.1f}{visible:>10}{grad:>14.3e}"
              f"   (相机前方点 {in_front}/{len(pts)})")
        results.append(((flip_y, flip_z), alpha_sum, grad))

    print("\n=== 结论 ===")
    best = max(results, key=lambda r: (r[1], r[2]))
    if best[1] <= 0:
        print("  四种约定都渲染不出任何像素 → 问题不只是符号翻转，")
        print("  更可能是 near/far 裁剪、尺度，或 COLMAP 位姿本身与图像不一致。")
    else:
        print(f"  最佳约定：y*{best[0][0]:+.0f} z*{best[0][1]:+.0f}  "
              f"alpha 和={best[1]:.1f}  梯度={best[2]:.3e}")
        print("  这是应该写进 gsplat_trainer.load_colmap_data 的翻转方式。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
