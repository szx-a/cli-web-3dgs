"""一次性试验：官方 `gsplat.DefaultStrategy` 能否直接接上本项目的 trainer 接口。

背景：本项目自写的密度控制（3D 梯度 + 固定阈值）与官方机制不成套，
诊断证明**浮点污染**是当前网格质量的唯一瓶颈，而零散补公式已被证明无效
（三次尝试分别崩到 13~17 / 23.9 / 9.4 dB）。

所以这一轮改为**直接复用官方 `DefaultStrategy`**（含 `ops.py` 的
duplicate/split/remove/reset_opa），而不是继续抄。

本脚本只回答"接不接得上"这一件事，不做完整训练：
  1. 构造一个小场景（真实 COLMAP 位姿 + 少量图）
  2. 用官方要求的方式建 params / optimizers 字典
  3. 跑若干步：step_pre_backward -> backward -> optimizer.step -> step_post_backward
  4. 打印每步的高斯数与 alpha 和，确认不报错、不发生爆炸

用法：scripts\\gpu_python.cmd scripts\\probe_official_strategy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import numpy as np
    import torch

    from gsflow import paths

    paths.apply_runtime_env()

    from gsplat import DefaultStrategy, rasterization
    from gsplat.strategy.ops import reset_opa

    from gsflow.gsdata.colmap_io import read_model
    from gsflow.trainers.gsplat_trainer import find_model_dir

    job = ROOT / "workspace" / "cmp-maskfull"
    data_dir = job / "04_train" / "data"
    rec = read_model(find_model_dir(data_dir))
    print(f"相机数: {len(rec.images)}")

    dev = "cuda"
    img_dir = data_dir / "images"
    names = [im.name for im in rec.images]
    from PIL import Image

    imgs = []
    ks, vms = [], []
    for im in rec.images[:6]:
        p = img_dir / im.name
        if not p.exists():
            cands = list(img_dir.glob(Path(im.name).stem + ".*"))
            if not cands:
                continue
            p = cands[0]
        with Image.open(p) as x:
            a = np.asarray(x.convert("RGB")).copy()   # copy：numpy 只读数组会让 torch 警告
        t = torch.from_numpy(a).float().to(dev) / 255.0
        t = torch.nn.functional.interpolate(
            t.permute(2, 0, 1)[None], size=(160, 90), mode="area")[0].permute(1, 2, 0)
        imgs.append(t)
        r, tt = im.cam_to_world()
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = r
        c2w[:3, 3] = tt
        vms.append(torch.from_numpy(np.linalg.inv(c2w)).to(dev))
        cam = rec.cameras[im.camera_id]
        K = cam.intrinsics().astype(np.float32)
        K[0] *= 90.0 / cam.width
        K[1] *= 160.0 / cam.height
        K[0, 2] = 45.0
        K[1, 2] = 80.0
        ks.append(torch.from_numpy(K).to(dev))
    print(f"载入 {len(imgs)} 张（160x90）")

    # 官方要求的 params 键：means / scales / quats / opacities（+ 颜色用 shs）
    n = 5000
    g = torch.Generator(device="cpu").manual_seed(0)
    pts = torch.randn(n, 3, generator=g) * 0.3
    params = {
        "means": torch.nn.Parameter(pts.to(dev)),
        "scales": torch.nn.Parameter(torch.full((n, 3), -3.0, device=dev)),
        "quats": torch.nn.Parameter(torch.cat(
            [torch.ones(n, 1, device=dev), torch.zeros(n, 3, device=dev)], dim=1)),
        "opacities": torch.nn.Parameter(torch.zeros(n, device=dev)),
        "shs": torch.nn.Parameter(torch.zeros(n, 1, 3, device=dev)),
    }
    optimizers = {
        k: torch.optim.Adam([{"params": [v], "lr": lr, "name": k}], eps=1e-15)
        for k, v, lr in (("means", params["means"], 1.6e-4),
                         ("scales", params["scales"], 5e-3),
                         ("quats", params["quats"], 1e-3),
                         ("opacities", params["opacities"], 5e-2),
                         ("shs", params["shs"], 2.5e-3))
    }

    strategy = DefaultStrategy(verbose=False)
    try:
        strategy.check_sanity(params, optimizers)
        print("check_sanity: OK")
    except AssertionError as exc:
        print(f"check_sanity 失败: {exc}")
        return 1
    state = strategy.initialize_state(scene_scale=3.0)
    print(f"state keys: {list(state.keys())}")

    for step in range(1, 401):
        i = step % len(imgs)
        gt = imgs[i]
        colors, alphas, info = rasterization(
            means=params["means"], quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=params["shs"], viewmats=vms[i][None], Ks=ks[i][None],
            width=90, height=160, sh_degree=0, packed=False, render_mode="RGB",
        )
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss = torch.nn.functional.l1_loss(colors[0], gt)
        loss.backward()
        for o in optimizers.values():
            o.step()
            o.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if step % 100 == 0:
            print(f"  step {step:4d}  高斯={params['means'].shape[0]:6d}  "
                  f"loss={float(loss):.4f}  alpha和={float(alphas[0].sum()):.0f}")

    print("\n结论：官方 DefaultStrategy 可直接使用（含 ops.py 的 duplicate/split/remove/reset_opa）")
    print(f"最终高斯数 {params['means'].shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
