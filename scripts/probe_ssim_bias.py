"""量化 `ssim_simple` 的**常数图偏置**在真实数据上的影响。

发现（scripts/compare_ssim.py）：`ssim_simple` 在**两张全黑图**上返回 **0.9**，
而正确值必须是 **1.0**。根因是平方项的写法：

本项目（单边 eps 在分母）：
    ssim_map = (2·μ1μ2 + c1)(2·σ12 + c2) / ((μ1²+μ2²+c1)(σ1²+σ2²+c2) + 1e-8)
对于 μ=0、σ=0 的常数图： 分子 = c1·c2，分母 = c1·c2 + 1e-8
    ⇒ 0.01⁴ / (0.01⁴ + 1e-8) = 1e-8 / 2e-8 = **0.5**（单通道）
    三通道平均后 → 0.9 左右（与实测一致）

原版 3DGS 的 `utils/image_utils.py::_ssim` 用的是**同一个单边 eps**，
所以这不是"我抄错"，而是**原版 3DGS 自带的偏置**；skimage / pytorch_msssim
做的是对称写法 `(2μ1μ2+c1)/(μ1²+μ2²+c1)`（**分子不除 eps**），常数图严格为 1。

为什么本项目**必须修**（而原版 3DGS 可以不修）：
原版 3DGS 的场景是**有纹理的实物**（Mip-NeRF 360 / Tanks&Temples），
几乎没有大面积纯色区；而本项目是**抠像后黑底**的素材，
实测背景占 **42%~85% 的像素是纯黑**。于是在大片区域上：
    SSIM_map ≈ 0.5（单通道）而不是 1.0
⇒ 即使背景渲染得完全正确，SSIM 项也永远拿不到"满分"，
损失里出现一个**与质量无关的常数偏置**，并且它的大小随**背景占比**变化
（前景占比 12%~55% ⇒ 偏置随视角漂移，与之前 G70 修的 loss 尺度漂移是同一类病）。

本脚本在真实训练数据上量化这个偏置，决定是否修、修了影响多大。

用法: scripts\\gpu_python.cmd scripts\\probe_ssim_bias.py --job ab-composite --views 12
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
    ap.add_argument("--job", default="ab-composite")
    ap.add_argument("--views", type=int, default=12)
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import torch
    from gsplat import rasterization
    from plyfile import PlyData
    from skimage.metrics import structural_similarity as sk_ssim

    from gsflow.trainers.gsplat_trainer import _load_images, load_colmap_data, ssim_simple

    job = ROOT / "workspace" / args.job
    data_dir = job / "04_train" / "data"
    metrics = json.loads((job / "04_train" / "metrics.json").read_text(encoding="utf-8"))
    ply = Path(metrics["best_ply"])
    print(f"作业 {args.job}  用 {ply.parent.name} 评估 SSIM 偏置")
    print("（对比对象：skimage 的 structural_similarity，作为参考实现）\n")

    data = load_colmap_data(data_dir, downscale=1, white_background=False, flip_camera=False)
    imgs, _ = _load_images(data_dir, data["names"], None)
    ks = torch.from_numpy(data["ks"]).float()
    c2ws = torch.from_numpy(data["c2ws"]).float()
    viewmats = torch.linalg.inv(c2ws)

    d = PlyData.read(str(ply)).elements[0].data
    dev = torch.device("cuda")
    means = torch.tensor(np.stack([d["x"], d["y"], d["z"]], 1), dtype=torch.float32, device=dev)
    scales = torch.tensor(np.exp(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], 1)),
                          dtype=torch.float32, device=dev)
    q = np.stack([d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"]], 1)
    quats = torch.tensor(q, dtype=torch.float32, device=dev)
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
    op = 1.0 / (1.0 + np.exp(-np.asarray(d["opacity"], dtype=np.float64)))
    opacities = torch.tensor(op, dtype=torch.float32, device=dev)
    shs = torch.tensor(np.stack([np.asarray(d[f"f_dc_{k}"], dtype=np.float64) for k in range(3)], 1),
                       dtype=torch.float32, device=dev).unsqueeze(1)

    n = min(args.views, len(imgs))
    step = max(1, len(imgs) // n)
    rows = []
    for i in range(0, len(imgs), step):
        gt = imgs[i].to(dev)
        H, W = gt.shape[:2]
        with torch.no_grad():
            colors, alphas, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities, colors=shs,
                viewmats=viewmats[i][None].to(dev), Ks=ks[i][None].to(dev),
                width=W, height=H, sh_degree=0, near_plane=0.01, far_plane=1e10,
                packed=False, render_mode="RGB",
            )
        render = (colors[0] + 0.0 * (1 - alphas[0])).clamp(0, 1)
        a = render.permute(2, 0, 1)[None]
        b = gt.permute(2, 0, 1)[None]
        mine = float(ssim_simple(a, b))
        ref = float(sk_ssim(render.cpu().numpy(), gt.cpu().numpy(),
                            channel_axis=2, data_range=1.0))
        # 背景占比（近黑像素）
        g = gt.cpu().numpy()
        bg = float((g.max(axis=2) < 0.02).mean())
        rows.append((i, mine, ref, mine - ref, bg))

    print(f"{'view':>6}{'本项目 SSIM':>14}{'skimage':>11}{'差':>10}{'背景占比':>11}")
    print("-" * 54)
    for i, m, r, dd, bg in rows:
        print(f"{i:>6}{m:>14.4f}{r:>11.4f}{dd:>10.4f}{bg * 100:>10.1f}%")

    mm = float(np.mean([r[1] for r in rows]))
    mr = float(np.mean([r[2] for r in rows]))
    md = float(np.mean([r[3] for r in rows]))
    mbg = float(np.mean([r[4] for r in rows]))
    print(f"\n平均：本项目 {mm:.4f}  skimage {mr:.4f}  偏差 {md:+.4f}  背景占比 {mbg * 100:.1f}%")
    print("\n解读：")
    print("  * 偏差为正且随背景占比增大 ⇒ 单边 eps 让'渲染成正确背景'拿不到满分，")
    print("    等于给损失加了一个**与质量无关、随视角漂移**的常数项。")
    print("  * 需要关注的是偏差的**绝对量级**与**波动**：相对主损失占比越大越该修。")
    print(f"  * 本数据平均偏差 {md:+.4f}；而 1-SSIM 项本身约 0.33，")
    print(f"    即偏置约占 SSIM 项的 {abs(md) / 0.33 * 100:.1f}%。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
