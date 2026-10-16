"""用**训练视角的真实相机**渲染已训练模型，并与真值逐像素对比。

为什么必须这样做：`05_export/preview_*.png` 里的"渲染"来自
`_render_previews_splat`，它用的是**环绕视角**（围绕场景转的任意相机），
**不是训练视角**。所以拿它和真值并排看"尺度是否一致"是**无效的**——
我从这个假象里读出过"渲染比真值大 2.3 倍"，并据此怀疑相机标定，浪费了时间。
（这是本项目第三次踩"诊断工具本身有问题"：另外两次是读错 API 键名、
用 HEAD 请求查只支持 GET 的端点。）

本脚本以**训练循环同一套相机与内参**渲染指定视角，输出：
  * PSNR（与训练/评估口径一致：全图 MSE）
  * 渲染与真值中非黑内容的包围盒（判断取景是否一致）
  * 并排对比图，供肉眼确认

用法:
  scripts\\gpu_python.cmd scripts\\render_train_views.py --job maskloss-v2 --views 0,50,100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def bbox(arr: np.ndarray, thresh: float = 0.06) -> dict:
    m = arr.max(axis=2) > thresh
    if not m.any():
        return {"empty": True}
    ys, xs = np.nonzero(m)
    h, w = m.shape
    return {"top": float(ys.min() / h), "bottom": float(ys.max() / h),
            "height_frac": float((ys.max() - ys.min() + 1) / h),
            "coverage": float(m.mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--views", default="0,50,100")
    ap.add_argument("--resolution", type=int, default=0,
                    help="0 = 与训练一致（读 job 的配置）")
    args = ap.parse_args()

    import torch
    from gsplat import rasterization
    from plyfile import PlyData

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    from gsflow.trainers.gsplat_trainer import _load_images, load_colmap_data

    job = ROOT / "workspace" / args.job
    data_dir = job / "04_train" / "data"
    metrics = json.loads((job / "04_train" / "metrics.json").read_text(encoding="utf-8"))
    ply = Path(metrics["best_ply"])
    print(f"作业 {args.job}  检查点 {ply.parent.name}")

    data = load_colmap_data(data_dir, downscale=1, white_background=False, flip_camera=False)
    names = data["names"]
    max_side = args.resolution or 0
    # 与训练一致：图像按 max_side 缩放，K 同步缩放
    imgs, _ = _load_images(data_dir, names, max_side if max_side > 0 else None)
    ks = torch.from_numpy(data["ks"]).float()
    if max_side > 0:
        for i, im in enumerate(imgs):
            ow, oh = data["widths"][i], data["heights"][i]
            ks[i, 0, :] *= im.shape[1] / ow
            ks[i, 1, :] *= im.shape[0] / oh
    c2ws = torch.from_numpy(data["c2ws"]).float()
    viewmats = torch.linalg.inv(c2ws)

    # ⚠️ 必须用**完整 SH** 读取与渲染（含 45 个 f_rest_*）。
    # 旧版本只读 f_dc_* 并以 sh_degree=0 渲染 ⇒ 实测**低估 1.91 dB**，
    # 而且抹掉了视角相关外观（详见 src/gsflow/gsdata/shio.py 的说明）。
    # 读取统一走 shio，避免"同一决定写多遍、然后一起写错"。
    from gsflow.gsdata.shio import read_gaussians_full

    device = torch.device("cuda")
    G = read_gaussians_full(ply)
    print(f"高斯基色读取：SH 阶数 {G['sh_degree']}，高斯 {G['n']} 个")

    out_dir = job / "04_train" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    psnrs = []
    for vi in [int(v) for v in args.views.split(",") if v != ""]:
        gt = imgs[vi].to(device)
        H, W = gt.shape[:2]
        with torch.no_grad():
            colors, alphas, _ = rasterization(
                means=torch.as_tensor(G["xyz"], device=device),
                quats=torch.as_tensor(G["quats"], device=device),
                scales=torch.as_tensor(G["scales"], device=device),
                opacities=torch.as_tensor(G["opacity"], device=device).reshape(-1).clamp(0, 1),
                colors=torch.as_tensor(G["shs"], device=device),
                viewmats=viewmats[vi][None].to(device),
                Ks=ks[vi][None].to(device), width=W, height=H,
                sh_degree=int(G["sh_degree"]), near_plane=0.01, far_plane=1e10,
                packed=False, render_mode="RGB",
            )
        render = colors[0].clamp(0, 1)
        mse = float(torch.mean((render - gt) ** 2).clamp_min(1e-10))
        psnr = 10.0 * np.log10(1.0 / mse)
        psnrs.append(psnr)
        r = render.detach().cpu().numpy()
        g = gt.detach().cpu().numpy()
        br, bg = bbox(r), bbox(g)
        print(f"  view {vi} ({names[vi]}): PSNR {psnr:.2f} dB")
        if not br.get("empty") and not bg.get("empty"):
            print(f"     渲染 高度占比 {br['height_frac']:.3f} 覆盖 {br['coverage']:.3f} | "
                  f"真值 高度占比 {bg['height_frac']:.3f} 覆盖 {bg['coverage']:.3f} "
                  f"| 比 {br['height_frac'] / max(bg['height_frac'], 1e-6):.2f}")
        # 并排图
        both = np.concatenate([r, g], axis=1)
        Image.fromarray((np.clip(both, 0, 1) * 255).astype(np.uint8)).save(
            out_dir / f"trainview_{vi:03d}.png")
    print(f"\n平均 PSNR（训练视角，全图口径）= {sum(psnrs) / max(len(psnrs), 1):.2f} dB")
    print(f"并排图已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
