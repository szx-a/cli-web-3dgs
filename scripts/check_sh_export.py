"""检查导出的 PLY 里是否有**高阶 SH 系数**，以及导出/诊断链路是否把它们丢掉了。

背景（用户指出的"实现逻辑被简化"）：
3DGS 的颜色是视角相关的球谐（SH）。官方默认 `sh_degree=3` ⇒ 每个高斯有
3 个 DC 分量 + 45 个高阶分量（(3+1)²−1 = 15 个基 × 3 通道 = 45）。

如果 PLY **写了** `f_rest_*` 但读取侧只读 `f_dc_*` 并以 `sh_degree=0` 渲染，
那么：
  * 所有**预览图**都不是模型真实的样子（缺视角相关高光/色彩变化）；
  * 我用 `sh_degree=0` 的诊断脚本测的 PSNR/SSIM/LPIPS **低估了模型**，
    而且对"改善高频细节"的优化手段（如感知损失）**天然不敏感**；
  * 网格的**顶点色**也只取了 DC ⇒ 贴图/顶点色偏平。

这个脚本只做事实核查（不训练），输出：
  1. PLY 的字段清单里有没有 f_rest_*；
  2. 若有，量化"DC-only 渲染"与"全 SH 渲染"的差别（PSNR 差多少）。

用法: scripts\\gpu_python.cmd scripts\\check_sh_export.py --job maskloss-v2
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
    ap.add_argument("--job", default="maskloss-v2")
    ap.add_argument("--views", type=int, default=6)
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    import torch
    from plyfile import PlyData

    from gsflow.trainers.gsplat_trainer import _load_images, load_colmap_data

    job = ROOT / "workspace" / args.job
    metrics = json.loads((job / "04_train" / "metrics.json").read_text(encoding="utf-8"))
    ply = Path(metrics["best_ply"])
    print(f"作业 {args.job}  检查点 {ply.parent.name}")
    print(f"metrics.json 记录 sh_degree = {metrics.get('sh_degree', '(未记录)')}")

    v = PlyData.read(str(ply))["vertex"]
    names = list(v.data.dtype.names or ())
    f_dc = [n for n in names if n.startswith("f_dc_")]
    f_rest = [n for n in names if n.startswith("f_rest_")]
    print(f"\nPLY 字段总数 {len(names)}；f_dc_* {len(f_dc)} 个；f_rest_* **{len(f_rest)} 个**")
    if f_rest:
        print(f"  ⇒ PLY **有**高阶 SH（{len(f_rest)} 个分量）")
        n_rest_expected = 45  # sh_degree=3: 15 基 × 3 通道
        if len(f_rest) == n_rest_expected:
            print(f"  ⇒ 数量与 sh_degree=3 吻合（{n_rest_expected}）")
        else:
            print(f"  ⇒ 数量与 sh_degree=3 的 {n_rest_expected} 不符，需确认 sh_degree")
    else:
        print("  ⇒ PLY **没有**高阶 SH（只有 DC）—— 模型导出即退化为视角无关颜色")

    # 量化 DC-only 与全 SH 的差别
    if not f_rest:
        print("\n没有高阶 SH，无法比较；跳过渲染对比。")
        return 0

    data = load_colmap_data(job / "04_train" / "data", downscale=1,
                            white_background=False, flip_camera=False)
    imgs, _ = _load_images(job / "04_train" / "data", data["names"], None)
    ks = torch.from_numpy(data["ks"]).float()
    c2ws = torch.from_numpy(data["c2ws"]).float()
    viewmats = torch.linalg.inv(c2ws)

    dev = torch.device("cuda")
    n = len(v["x"])
    means = torch.tensor(np.stack([np.asarray(v["x"]), np.asarray(v["y"]),
                                   np.asarray(v["z"])], 1), dtype=torch.float32, device=dev)
    scales = torch.tensor(np.exp(np.stack([np.asarray(v["scale_0"]), np.asarray(v["scale_1"]),
                                           np.asarray(v["scale_2"])], 1)),
                          dtype=torch.float32, device=dev)
    quats = torch.tensor(np.stack([np.asarray(v["rot_0"]), np.asarray(v["rot_1"]),
                                   np.asarray(v["rot_2"]), np.asarray(v["rot_3"])], 1),
                         dtype=torch.float32, device=dev)
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
    op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
    opacities = torch.tensor(op, dtype=torch.float32, device=dev)

    # 全 SH：DC 3 个 + rest 45 个，按 3DGS 约定重排成 [N, 16, 3]
    # f_rest 的存储顺序是**通道主序**（先所有基的 R，再 G，再 B），长度 = 15*3
    sh = np.zeros((n, 16, 3), dtype=np.float32)
    for c in range(3):
        sh[:, 0, c] = np.asarray(v[f"f_dc_{c}"], dtype=np.float32)
    nr = len(f_rest) // 3
    for c in range(3):
        for k in range(nr):
            sh[:, k + 1, c] = np.asarray(v[f"f_rest_{c * nr + k}"], dtype=np.float32)
    shs = torch.tensor(sh, dtype=torch.float32, device=dev)
    shs_dc = shs[:, :1, :].contiguous()

    from gsplat import rasterization

    print(f"\n{'view':>6}{'全SH PSNR':>12}{'仅DC PSNR':>12}{'差':>9}")
    print("-" * 40)
    a_all, a_dc = [], []
    step = max(1, len(imgs) // args.views)
    for i in range(0, len(imgs), step):
        gt = imgs[i].to(dev)
        H, W = gt.shape[:2]
        vals = []
        for sh_in, deg in ((shs, 3), (shs_dc, 0)):
            with torch.no_grad():
                colors, alphas, _ = rasterization(
                    means=means, quats=quats, scales=scales, opacities=opacities,
                    colors=sh_in, viewmats=viewmats[i][None].to(dev),
                    Ks=ks[i][None].to(dev), width=W, height=H,
                    sh_degree=deg, near_plane=0.01, far_plane=1e10,
                    packed=False, render_mode="RGB",
                )
            r = colors[0].clamp(0, 1)
            mse = float(torch.mean((r - gt) ** 2).clamp_min(1e-10))
            vals.append(10.0 * np.log10(1.0 / mse))
        a_all.append(vals[0]); a_dc.append(vals[1])
        print(f"{i:>6}{vals[0]:>12.2f}{vals[1]:>12.2f}{vals[0] - vals[1]:>+9.2f}")

    print(f"\n平均：全SH {np.mean(a_all):.2f} dB   仅DC {np.mean(a_dc):.2f} dB   "
          f"差 {np.mean(a_all) - np.mean(a_dc):+.2f} dB")
    print("\n解读：若差值明显（> 0.3 dB），说明导出/诊断链路用 sh_degree=0 是**真实简化**，")
    print("      它会让预览图与网格顶点色都不等于模型真实外观。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
