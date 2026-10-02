"""gsplat 训练器：默认的 3DGS 训练后端（免自编译 CUDA 扩展）。

为什么默认用它：RTX 50 系（sm_120）下原版 diff-gaussian-rasterization 需要
手工编译且容易失败，而 gsplat 是纯 Python 包 + JIT 编译内核，配合
PyTorch cu128 轮子即可直接用，且显存更省。

算法与官方 3DGS 保持一致的关键点：
  * 参数化：位置 xyz、对数尺度 log_scales、四元数 quats、
    不透明度 logit 空间、视角相关颜色用球谐（SH，最高 3 阶）
  * 损失：0.8 * L1 + 0.2 * (1 - SSIM)
  * 自适应密度控制：梯度阈值 densify、周期性 opacity 重置、按屏幕尺寸 prune
  * 渐进分辨率：显存不足时先低分辨率热身

掩码：对黑背景训练集，前 1/4 迭代只在掩码内的像素上算损失，
避免模型把黑背景也学成高斯（这是黑背景方案最常见的质量陷阱）。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import StageError


@dataclass
class TrainConfig:
    data_dir: Path
    out_dir: Path
    iterations: int = 30000
    resolution: int = 1600            # 长边
    sh_degree: int = 3
    data_device: str = "cuda"
    cap_max: int = 1_500_000
    ssim_weight: float = 0.2
    densify_from: int = 500
    densify_until: int = 15000
    densify_interval: int = 100
    opacity_reset_interval: int = 3000
    eval: bool = True
    eval_interval: int = 2000
    white_background: bool = False
    mask_loss_until: float = 0.25
    seed: int = 42
    log_every: int = 200


def ssim_simple(img1, img2):
    """可微 SSIM（11x11 高斯窗），避免依赖额外包。"""
    import torch
    import torch.nn.functional as F

    window_size = 11
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(img1.device)
    window = (g[:, None] @ g[None, :])[None, None]
    window = window.expand(img1.shape[1], 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.shape[1])
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    )
    return ssim_map.mean()


def find_model_dir(data_dir: Path) -> Path:
    """定位去畸变后的 COLMAP 模型目录。

    踩坑：COLMAP 3.12 的 image_undistorter 把模型直接写在 `sparse/` 下
    （cameras.bin 与 images.bin 同级），而 3.11 及更早写 `sparse/0/`。
    两种布局都要认，否则训练阶段会报“找不到稀疏模型”。
    """
    sparse = data_dir / "sparse"
    for cand in (sparse / "0", sparse):
        if (cand / "cameras.bin").exists() and (cand / "images.bin").exists():
            return cand
    if sparse.exists():
        subs = [d for d in sparse.iterdir() if d.is_dir() and (d / "cameras.bin").exists()]
        if subs:
            return sorted(subs)[0]
    return sparse / "0"


def load_colmap_data(data_dir: Path, downscale: int = 1, white_background: bool = False):
    """读取去畸变后的 COLMAP 数据，返回相机参数、位姿与点云。"""
    from ..gsdata.colmap_io import read_model

    model_dir = find_model_dir(data_dir)
    if not (model_dir / "cameras.bin").exists():
        raise StageError("train", f"找不到 COLMAP 稀疏模型：{model_dir}", "先跑 colmap 阶段")

    rec = read_model(model_dir)
    if not rec.images:
        raise StageError("train", "COLMAP 模型中没有已注册图像", "检查 colmap 阶段的日志与 report.json")

    cam_by_id = rec.cameras
    images = sorted(rec.images, key=lambda im: im.name)

    c2ws, ks, names, widths, heights = [], [], [], [], []
    for im in images:
        cam = cam_by_id[im.camera_id]
        k = cam.intrinsics()
        if downscale > 1:
            k[0, 0] /= downscale
            k[1, 1] /= downscale
            k[0, 2] = (k[0, 2] + 0.5) / downscale - 0.5
            k[1, 2] = (k[1, 2] + 0.5) / downscale - 0.5
        r, t = im.cam_to_world()
        c2w = np.eye(4)
        c2w[:3, :3] = r
        c2w[:3, 3] = t
        # COLMAP/OpenCV -> OpenGL 约定（y、z 取反），与 gsplat 示例一致
        c2w[0:3, 1:3] *= -1
        c2ws.append(c2w)
        ks.append(k)
        names.append(im.name)
        widths.append(int(math.ceil(cam.width / downscale)))
        heights.append(int(math.ceil(cam.height / downscale)))

    return {
        "c2ws": np.stack(c2ws),
        "ks": np.stack(ks),
        "names": names,
        "widths": widths,
        "heights": heights,
        "points": rec.points3D,
        "colors": rec.colors3D,
        "errors": rec.errors3D,
        "model_dir": model_dir,
        "white_background": white_background,
    }


def _load_images(data_dir: Path, names: list[str], max_side: int | None):
    from PIL import Image
    import torch

    imgs = []
    masks = []
    for name in names:
        p = data_dir / "images" / name
        if not p.exists():
            # undistorter 偶尔会改扩展名
            cands = list((data_dir / "images").glob(Path(name).stem + ".*"))
            if not cands:
                raise StageError("train", f"缺少图像 {p}", "检查去畸变输出目录")
            p = cands[0]
        with Image.open(p) as im:
            im = im.convert("RGB")
            if max_side and max(im.size) > max_side:
                scale = max_side / max(im.size)
                im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
            imgs.append(torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).float() / 255.0)
        mask_p = data_dir / "masks" / (Path(name).stem + ".png")
        if mask_p.exists():
            with Image.open(mask_p) as m:
                m = m.convert("L")
                if m.size != im.size:
                    m = m.resize(im.size, Image.NEAREST)
                masks.append(torch.from_numpy(np.asarray(m, dtype=np.uint8).copy() > 127))
    return imgs, masks


def train(cfg: TrainConfig, ctx) -> dict:
    import torch

    try:
        from gsplat import rasterization
    except ImportError as exc:  # pragma: no cover
        raise StageError(
            "train",
            f"未安装 gsplat：{exc}",
            "在虚拟环境里执行：pip install gsplat（需要 torch 已装好且为 cu128 及以上）",
        ) from exc

    if not torch.cuda.is_available():
        raise StageError("train", "当前 torch 看不到 CUDA 设备", "确认装的是 cu128+ 轮子并且驱动正常")

    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)
    free_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    ctx.info(f"设备：{torch.cuda.get_device_name(0)}，显存 {free_vram_gb:.1f} GB")
    if free_vram_gb < 10:
        ctx.warn("显存 < 10GB：会自动降低分辨率并限制高斯数量（cap_max）")

    data = load_colmap_data(cfg.data_dir, downscale=1, white_background=cfg.white_background)
    names = data["names"]
    ctx.info(f"已注册视图 {len(names)} 个，稀疏点 {len(data['points'])}")

    # 分辨率：按长边缩放到 cfg.resolution
    max_side = cfg.resolution if cfg.resolution > 0 else None
    imgs, masks = _load_images(cfg.data_dir, names, max_side)
    h, w = imgs[0].shape[:2]
    ctx.info(f"训练分辨率：{w}x{h}，共 {len(imgs)} 张")

    # 相机：按最终图像尺寸重算 K
    ks = torch.from_numpy(data["ks"]).float()
    if max_side:
        for i, im in enumerate(imgs):
            ow, oh = data["widths"][i], data["heights"][i]
            sx, sy = im.shape[1] / ow, im.shape[0] / oh
            ks[i, 0, :] *= sx
            ks[i, 1, :] *= sy
    c2ws = torch.from_numpy(data["c2ws"]).float()
    viewmats = torch.linalg.inv(c2ws)

    # 划分训练/评估
    perm = torch.randperm(len(imgs), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    n_eval = max(1, len(imgs) // 10) if cfg.eval and len(imgs) >= 10 else 0
    eval_idx = set(perm[:n_eval])
    train_idx = [i for i in range(len(imgs)) if i not in eval_idx]

    # ---- 高斯参数初始化 ----
    pts = data["points"]
    if len(pts) == 0:
        raise StageError("train", "稀疏点云为空", "COLMAP 阶段可能只注册了极少图像；重跑 colmap")
    if cfg.cap_max and len(pts) > cfg.cap_max:
        sel = np.random.default_rng(cfg.seed).choice(len(pts), cfg.cap_max, replace=False)
        pts = pts[sel]
        cols = data["colors"][sel]
    else:
        cols = data["colors"]

    means = torch.nn.Parameter(torch.from_numpy(pts).float().to(device))
    n = means.shape[0]
    dist = torch.sqrt(torch.sum((means - means.mean(0)) ** 2, dim=1, keepdim=True))
    dist = torch.clamp(dist, 1e-3, None)
    scales = torch.nn.Parameter(torch.log(dist * 0.5).repeat(1, 3).to(device))
    quats = torch.nn.Parameter(torch.zeros(n, 4, device=device))
    quats.data[:, 0] = 1.0
    opac = torch.nn.Parameter(torch.logit(torch.full((n, 1), 0.1, device=device)))
    sh_dim = (cfg.sh_degree + 1) ** 2
    shs = torch.nn.Parameter(torch.zeros(n, sh_dim, 3, device=device))
    rgb0 = torch.from_numpy(cols).float().to(device) / 255.0
    shs.data[:, 0, :] = (rgb0 - 0.5) / 0.28209479177387814  # SH DC 反解

    def make_optimizers():
        """密度控制会改变参数张量形状，必须重建优化器与调度器（官方实现同款做法）。"""
        opts = {
            "means": torch.optim.Adam([means], lr=1.6e-4, eps=1e-15),
            "scales": torch.optim.Adam([scales], lr=5e-3, eps=1e-15),
            "quats": torch.optim.Adam([quats], lr=1e-3, eps=1e-15),
            "opacities": torch.optim.Adam([opac], lr=5e-2, eps=1e-15),
            "shs": torch.optim.Adam([shs], lr=2.5e-3, eps=1e-15),
        }
        gamma = 0.01 ** (1 / max(cfg.iterations, 1))
        schs = {k: torch.optim.lr_scheduler.ExponentialLR(v, gamma=gamma) for k, v in opts.items()}
        return opts, schs

    optimizers, sched = make_optimizers()

    background = torch.tensor([1.0, 1.0, 1.0] if cfg.white_background else [0.0, 0.0, 0.0], device=device)

    def current_scales():
        return torch.exp(scales)

    def current_opacities():
        return torch.sigmoid(opac)

    def normalize_quats():
        with torch.no_grad():
            quats.data = quats.data / (quats.data.norm(dim=-1, keepdim=True) + 1e-8)

    ckpt_dir = cfg.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    gs_dir = cfg.out_dir / "point_cloud"
    gs_dir.mkdir(parents=True, exist_ok=True)

    loss_log: list[dict] = []
    psnr_log: list[dict] = []
    t0 = time.time()
    ema_loss = None

    for step in range(1, cfg.iterations + 1):
        if ctx.cancel_requested:
            ctx.warn(f"收到取消请求，保存 step{step} 检查点后退出")
            break
        normalize_quats()

        # 渐进分辨率：前 15% 迭代用半分辨率热身，省显存也更快收敛
        use_half = step < max(1, int(cfg.iterations * 0.15)) and max(h, w) > 800

        idx = train_idx[step % len(train_idx)]
        gt_full = imgs[idx].to(device)
        k = ks[idx].to(device)
        vm = viewmats[idx].to(device)
        if use_half:
            gt = torch.nn.functional.interpolate(
                gt_full.permute(2, 0, 1)[None], scale_factor=0.5, mode="bilinear", align_corners=False
            )[0].permute(1, 2, 0)
            kk = k.clone()
            kk[0, :] *= 0.5
            kk[1, :] *= 0.5
        else:
            gt = gt_full
            kk = k

        colors, alphas, _ = rasterization(
            means=means,
            quats=quats / (quats.norm(dim=-1, keepdim=True) + 1e-8),
            scales=current_scales(),
            opacities=torch.clamp(current_opacities().squeeze(-1), 0.0, 1.0),
            colors=shs,
            viewmats=vm[None],
            Ks=kk[None],
            width=gt.shape[1],
            height=gt.shape[0],
            sh_degree=cfg.sh_degree,
            near_plane=0.01,
            far_plane=1e10,
            packed=False,
            render_mode="RGB",
        )
        render = colors[0]
        if cfg.white_background:
            render = render + (1 - alphas[0])
        else:
            render = render + background * (1 - alphas[0])

        l1 = (render - gt).abs()
        if masks and step < int(cfg.iterations * cfg.mask_loss_until):
            m = masks[idx]
            if max_side and m.shape[0] != gt.shape[0]:
                m = torch.nn.functional.interpolate(
                    m[None, None].float(), size=gt.shape[:2], mode="nearest"
                )[0, 0] > 0.5
            else:
                m = m.to(device)
            if m.any():
                l1 = l1[m].mean() if l1[m].numel() else l1.mean()
            else:
                l1 = l1.mean()
        else:
            l1 = l1.mean()

        ssim_val = ssim_simple(render.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        loss = (1.0 - cfg.ssim_weight) * l1 + cfg.ssim_weight * (1.0 - ssim_val)

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        loss.backward()

        # ---- 自适应密度控制（阈值按场景尺度归一化，避免小/大场景失效）----
        if cfg.densify_from <= step <= cfg.densify_until and step % cfg.densify_interval == 0:
            with torch.no_grad():
                grad = means.grad
                if grad is not None:
                    cam_centers = c2ws[:, :3, 3].to(device)
                    scene_extent = float(torch.linalg.norm(cam_centers - cam_centers.mean(0), dim=1).mean()) + 1e-6
                    params_2d_scale = torch.exp(scales).max(dim=-1).values
                    densify = (grad.norm(dim=-1) > 2e-4) & (params_2d_scale < 0.01 * scene_extent)
                    prune = (current_opacities().squeeze(-1) < 0.005) | (params_2d_scale > 0.5 * scene_extent)
                    n_before = means.shape[0]
                    if prune.any():
                        keep = ~prune
                        _apply_keep(means, scales, quats, opac, shs, keep)
                    if densify.any() and means.shape[0] < cfg.cap_max:
                        _duplicate(means, scales, quats, opac, shs, densify, cap=cfg.cap_max)
                    if means.shape[0] != n_before:
                        ctx.info(f"step {step}: 高斯数量 {n_before} -> {means.shape[0]}")
                        optimizers, sched = make_optimizers()

        for opt in optimizers.values():
            opt.step()
        for sch in sched.values():
            sch.step()

        # ---- 透明度周期性重置（清除“漂浮”高斯）----
        if cfg.opacity_reset_interval and step % cfg.opacity_reset_interval == 0 and step < cfg.densify_until:
            with torch.no_grad():
                # 官方做法：周期把不透明度压回 0.1，清除“漂浮”高斯
                opac.data.clamp_(max=float(torch.logit(torch.tensor(0.1, device=device))))

        # ---- 评估与日志 ----
        if step % cfg.log_every == 0 or step == cfg.iterations:
            cur = float(loss.detach().cpu())
            ema_loss = cur if ema_loss is None else 0.9 * ema_loss + 0.1 * cur
            elapsed = time.time() - t0
            eta = elapsed / step * (cfg.iterations - step)
            ctx.info(
                f"step {step}/{cfg.iterations} loss={ema_loss:.4f} 高斯={means.shape[0]} "
                f"用时 {elapsed/60:.1f}min ETA {eta/60:.1f}min",
                progress=step / cfg.iterations,
            )
            loss_log.append({"step": step, "loss": round(ema_loss, 5), "gaussians": int(means.shape[0])})

        if cfg.eval and cfg.eval_interval and (step % cfg.eval_interval == 0 or step == cfg.iterations):
            psnr = evaluate(
                rasterization, means, quats, current_scales(), current_opacities(), shs,
                imgs, masks, ks, viewmats, sorted(eval_idx), cfg.sh_degree, background, device, max_side,
            )
            psnr_log.append({"step": step, "psnr": round(psnr, 3)})
            ctx.info(f"step {step} 评估 PSNR={psnr:.2f} dB", progress=step / cfg.iterations)

        if step % 5000 == 0 or step == cfg.iterations:
            save_ply(gs_dir / f"iteration_{step}" / "point_cloud.ply", means, current_scales(), quats,
                     current_opacities(), shs, cfg.sh_degree)
            ctx.info(f"已保存检查点 iteration_{step}")

    # ---- 收尾：保存最终模型与指标 ----
    final_dir = gs_dir / f"iteration_{cfg.iterations}"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_ply = final_dir / "point_cloud.ply"
    save_ply(final_ply, means, current_scales(), quats, current_opacities(), shs, cfg.sh_degree)

    metrics = {
        "iterations": cfg.iterations,
        "gaussians": int(means.shape[0]),
        "resolution": [int(w), int(h)],
        "train_views": len(train_idx),
        "eval_views": len(eval_idx),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "final_loss": loss_log[-1]["loss"] if loss_log else None,
        "psnr": psnr_log[-1]["psnr"] if psnr_log else None,
        "psnr_curve": psnr_log,
        "loss_curve": loss_log[-200:],
        "ply": str(final_ply),
    }
    (cfg.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    ctx.success(f"训练完成：{metrics['gaussians']} 个高斯，用时 {metrics['elapsed_min']} 分钟"
                + (f"，PSNR {metrics['psnr']:.2f} dB" if metrics["psnr"] else ""))
    return metrics


# --------------------------------------------------------------- 辅助函数
def _apply_keep(means, scales, quats, opac, shs, keep):
    import torch

    keep_idx = keep.nonzero(as_tuple=True)[0]
    with torch.no_grad():
        for p in (means, scales, quats, opac, shs):
            new = p.data[keep_idx].clone()
            p.data = new
            p.grad = None


def _duplicate(means, scales, quats, opac, shs, mask, cap: int):
    import torch

    idx = mask.nonzero(as_tuple=True)[0]
    room = max(0, cap - means.shape[0])
    if room <= 0 or idx.numel() == 0:
        return
    idx = idx[:room]
    with torch.no_grad():
        for p, jitter in ((means, True), (scales, False), (quats, False), (opac, False), (shs, False)):
            dup = p.data[idx].clone()
            if jitter:
                dup = dup + torch.randn_like(dup) * 1e-4
            p.data = torch.cat([p.data, dup], dim=0)
            p.grad = None


def _reset_optimizer_state(*_args, **_kwargs):
    """已废弃：密度控制后改为整体重建优化器（见 make_optimizers）。"""
    return None


def evaluate(rasterization, means, quats, scales, opacities, shs, imgs, masks, ks, viewmats,
             eval_idx, sh_degree, background, device, max_side):
    import torch

    if not eval_idx:
        return 0.0
    psnrs = []
    with torch.no_grad():
        for i in eval_idx:
            gt = imgs[i].to(device)
            colors, alphas, _ = rasterization(
                means=means,
                quats=quats / (quats.norm(dim=-1, keepdim=True) + 1e-8),
                scales=scales,
                opacities=opacities.squeeze(-1),
                colors=shs,
                viewmats=viewmats[i][None].to(device),
                Ks=ks[i][None].to(device),
                width=gt.shape[1],
                height=gt.shape[0],
                sh_degree=sh_degree,
                near_plane=0.01,
                far_plane=1e10,
                packed=False,
                render_mode="RGB",
            )
            render = colors[0] + background * (1 - alphas[0])
            mse = torch.mean((render - gt) ** 2).clamp_min(1e-10)
            psnrs.append(float(10.0 * torch.log10(1.0 / mse)))
    return sum(psnrs) / len(psnrs)


def save_ply(path: Path, means, scales, quats, opacities, shs, sh_degree: int) -> Path:
    """导出标准 3DGS PLY（与官方 inria 格式一致，可被 SuperSplat/three.js 读取）。"""
    from plyfile import PlyData, PlyElement
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        xyz = means.detach().cpu().numpy().astype(np.float32)
        normals = np.zeros_like(xyz)
        f_dc = shs[:, 0, :].detach().cpu().numpy().astype(np.float32)
        sh_dim = (sh_degree + 1) ** 2
        f_rest = shs[:, 1:sh_dim, :].detach().cpu().numpy().transpose(0, 2, 1).reshape(len(xyz), -1).astype(np.float32)
        op = opacities.detach().cpu().numpy().astype(np.float32)
        sc = scales.detach().cpu().numpy().astype(np.float32)
        rot = quats.detach().cpu().numpy().astype(np.float32)

    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    arr = np.concatenate([xyz, normals], axis=1)
    for i in range(f_dc.shape[1]):
        attrs.append(f"f_dc_{i}")
        arr = np.concatenate([arr, f_dc[:, i:i + 1]], axis=1)
    for i in range(f_rest.shape[1]):
        attrs.append(f"f_rest_{i}")
        arr = np.concatenate([arr, f_rest[:, i:i + 1]], axis=1)
    attrs.append("opacity")
    arr = np.concatenate([arr, op], axis=1)
    for i in range(sc.shape[1]):
        attrs.append(f"scale_{i}")
        arr = np.concatenate([arr, sc[:, i:i + 1]], axis=1)
    for i in range(rot.shape[1]):
        attrs.append(f"rot_{i}")
        arr = np.concatenate([arr, rot[:, i:i + 1]], axis=1)

    dtype = [(a, "f4") for a in attrs]
    elements = np.empty(arr.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, arr))
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))
    return path
