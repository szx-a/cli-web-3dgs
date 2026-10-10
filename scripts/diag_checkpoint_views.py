"""验证"训练在第 6000 步附近真的退化"这一假设（逐视图、逐步骤）。

背景：PSNR 在 6000 / 12000 等**固定步数**骤降，且
  * 内置密度控制与官方 DefaultStrategy 都降
  * 3 个固定评估视图也降
=> 排除"评估视图混叠"，指向**真实退化**。

本脚本直接检查训练途中保存的检查点：
  1. 逐个可用检查点、逐个留出视图算 PSNR（不受评估调度影响）
  2. 同时报告该检查点的不透明度分布与尺度分布
若某些检查点"个别视图极低、其余正常"，退化就是**局部的**（少数高斯变大/变透明）；
若"所有视图同时低"，则是全局性的。

用法：env\\venv\\Scripts\\python.exe scripts\\diag_checkpoint_views.py --job cmp-builtin
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
    ap.add_argument("--job", default="cmp-builtin")
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--dump", action="store_true", help="存一张渲染|真值并排图用于肉眼确认")
    args = ap.parse_args()

    from gsflow import paths

    paths.apply_runtime_env()
    import torch
    from gsplat import rasterization
    from PIL import Image

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.stages.export import ExportStage
    from gsflow.trainers.gsplat_trainer import find_model_dir

    job = ROOT / "workspace" / args.job
    data_dir = job / "04_train" / "data"
    rec = read_model(find_model_dir(data_dir))
    imgs = list(rec.images)
    n = len(imgs)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    eval_idx = sorted(set(perm[: max(1, n // 10)]))
    print(f"留出评估视图 {len(eval_idx)} 个: {eval_idx}")

    ckpts = sorted((job / "04_train" / "point_cloud").glob("iteration_*"),
                   key=lambda p: int(p.name.split("_")[1]))
    if not ckpts:
        print("没有检查点")
        return 1
    print(f"检查点 {len(ckpts)} 个: {[c.name for c in ckpts]}\n")

    dev = "cuda"
    img_dir = data_dir / "images"
    size = args.size
    gts, ks, vms = {}, {}, {}
    for i in eval_idx:
        im = imgs[i]
        p = img_dir / im.name
        if not p.exists():
            cands = list(img_dir.glob(Path(im.name).stem + ".*"))
            if not cands:
                continue
            p = cands[0]
        # ⚠️ 关键：用与 trainer `_load_images()` **完全一致**的缩放方式
        # （LANCZOS，按长边 max_side 缩放并按比例取整）。
        # 早先我用 `interpolate(mode="area", size=(h*sc, w*sc))`，取整/滤波与 trainer 不同，
        # 再叠加内参缩放的比例误差，渲染就整体偏了 -> PSNR 只有 10 dB，
        # 而 trainer 对同一检查点报 22~24 dB。**诊断脚本与被测对象必须用同一套预处理。**
        with Image.open(p) as x:
            t_img = x.convert("RGB")
        if max(t_img.size) > size:
            sc = size / max(t_img.size)
            t_img = t_img.resize((max(1, int(t_img.width * sc)),
                                  max(1, int(t_img.height * sc))), Image.LANCZOS)
        t = torch.from_numpy(np.asarray(t_img, dtype=np.uint8).copy()).float().to(dev) / 255.0
        gts[i] = t
        cam = rec.cameras[im.camera_id]
        K = cam.intrinsics().astype(np.float32)
        # 按同样比例缩放内参（含主点）
        sx, sy = t.shape[1] / cam.width, t.shape[0] / cam.height
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        ks[i] = torch.from_numpy(K).to(dev)
        r, tt = im.cam_to_world()
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = r
        c2w[:3, 3] = tt
        vms[i] = torch.from_numpy(np.linalg.inv(c2w)).to(dev)

    print(f"{'检查点':>18}{'高斯数':>10}" + "".join(f"{('v'+str(i)):>8}" for i in eval_idx)
          + f"{'均值':>8}{'不透明度p50':>12}{'不透明度p99':>12}{'尺度max':>10}")
    print("-" * (28 + 8 * len(eval_idx) + 42))
    for ck in ckpts:
        ply = ck / "point_cloud.ply"
        if not ply.exists():
            continue
        g = ExportStage()._read_ply(ply, max_points=0)
        n_g = len(g["xyz"])
        xyz = torch.as_tensor(g["xyz"], dtype=torch.float32, device=dev)
        sc = torch.as_tensor(g["scales"], dtype=torch.float32, device=dev)
        qt = torch.as_tensor(g["quats"], dtype=torch.float32, device=dev)
        op = torch.as_tensor(g["opacity"], dtype=torch.float32, device=dev).reshape(-1)
        rgb = torch.as_tensor(g["rgb"], dtype=torch.float32, device=dev) / 255.0
        shs = ((rgb - 0.5) / 0.28209479177387814).reshape(n_g, 1, 3)

        row = []
        with torch.no_grad():
            for i in eval_idx:
                if i not in gts:
                    row.append(float("nan"))
                    continue
                gt = gts[i]
                colors, alphas, _ = rasterization(
                    means=xyz, quats=qt, scales=torch.exp(sc),
                    opacities=op.clamp(0, 1), colors=shs,
                    viewmats=vms[i][None], Ks=ks[i][None],
                    width=gt.shape[1], height=gt.shape[0],
                    sh_degree=0, packed=False, render_mode="RGB")
                mse = torch.mean((colors[0] - gt) ** 2).clamp_min(1e-10)
                row.append(float(10.0 * torch.log10(1.0 / mse)))
                # 存一张"渲染 | 真值"并排图，肉眼确认几何是否对齐
                if args.dump and i == eval_idx[1]:
                    from PIL import Image as _I

                    r = (colors[0].detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    gimg = (gt.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    strip = np.concatenate([r, gimg], axis=1)
                    out = ROOT / "env" / "tmp" / f"diag_{ck.name}_v{i}.png"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    _I.fromarray(strip).save(out)
                    print(f"  [已存 {out.name}]")
        mean = float(np.nanmean(row))
        opv = torch.sigmoid(op)
        print(f"{ck.name:>18}{n_g:>10}"
              + "".join(f"{v:>8.2f}" for v in row)
              + f"{mean:>8.2f}"
              + f"{float(opv.median()):>12.4f}{float(opv.quantile(0.99)):>12.4f}"
              + f"{float(torch.exp(sc).max()):>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
