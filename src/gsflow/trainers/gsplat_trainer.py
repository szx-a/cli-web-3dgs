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

# torch 必须在模块级导入：密度控制/参数迁移的辅助函数用了 @torch.no_grad() 装饰器
try:
    import torch
except ImportError:  # 允许在没有 torch 的环境里 import 本模块（便于静态检查）
    torch = None  # type: ignore[assignment]


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
    # 加密判据：位置梯度范数超过阈值。
    # 实测校准：某场景 step 500 时 grad_norm 均值约 2e-4、最大约 1.2e-2；
    # 随着优化推进梯度会明显衰减（step 3000 已掉到 1e-5 量级），固定阈值会导致
    # "只在前几步加密过一次"。因此采用**分位数自适应阈值**：
    # 取所有高斯梯度范数的 (1 - densify_quantile) 分位数，并夹在 min/max 之间。
    densify_quantile: float = 0.75
    densify_grad_threshold: float = 1.5e-4   # 自适应阈值的下限
    densify_grad_max: float = 5e-3           # 自适应阈值的上限
    # 加密时"大尺度分裂 / 小尺度克隆"的分界（相对尺度中位数的倍数），官方 0.01*scene_scale
    split_scale_ratio: float = 1.0
    # 裁剪阈值：不透明度低于此值的高斯被删除（官方 0.005）
    prune_opacity: float = 0.005
    opacity_reset_interval: int = 3000
    eval: bool = True
    eval_interval: int = 2000
    white_background: bool = False
    mask_loss_until: float = 0.25
    seed: int = 42
    log_every: int = 200


def _quantile(values, q: float) -> float:
    """取分位数（0..1）。

    只用排序实现：torch.quantile 在元素很多时可能报错/很慢，
    而 torch.kthvalue 需要 1-based 的 k，算错就会拿到最小值
    （踩坑：曾出现"分位 0.00e+00"导致阈值退化成下限）。
    """
    import torch

    v = values.detach().float().reshape(-1)
    n = v.numel()
    if n == 0:
        return 0.0
    q = min(max(float(q), 0.0), 1.0)
    k = min(n - 1, max(0, int(round(q * (n - 1)))))
    return float(torch.sort(v).values[k])


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
        # COLMAP/OpenCV -> OpenGL 约定。
        #
        # 实测踩坑：早期按 gsplat 示例写成 `c2w[0:3, 1:3] *= -1`（y、z 都翻），
        # 结果**所有高斯落在相机背后**（相机空间 z 全为负、alpha 恒为 0、
        # 位置梯度恒为 0 → 自适应密度控制完全不触发，模型永远停在初始稀疏点）。
        # 用 scripts/probe_pose_convention.py 扫过 4 种翻转组合后确认：
        # 本流程的数据只需要翻 y。
        c2w[0:3, 1] *= -1
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

    # 参数用字典管理，和官方 gsplat 一致：密度控制会**替换**参数对象并迁移优化器状态，
    # 所以不能像早期版本那样把参数绑死在闭包里的局部变量上。
    params: dict[str, torch.nn.Parameter] = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opac,
        "shs": shs,
    }
    _LR = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "shs": 2.5e-3}
    _gamma = 0.01 ** (1 / max(cfg.iterations, 1))

    def make_optimizers():
        """为**当前**的参数对象建优化器（只在开头调用一次）。

        密度控制时不重建优化器，而是迁移状态 + 原地重建同 gamma 的调度器
        （见 _update_param_with_optimizer 与 official gsplat 的 ops.py）。
        """
        opts = {k: torch.optim.Adam([v], lr=_LR[k], eps=1e-15) for k, v in params.items()}
        return opts

    def schedule_lrs(step_idx: int) -> None:
        """与 ExponentialLR(gamma) 等价：手动设置每个参数组的学习率。"""
        factor = _gamma ** step_idx
        for k, opt in optimizers.items():
            for group in opt.param_groups:
                group["lr"] = _LR[k] * factor

    optimizers = make_optimizers()
    sched = None  # 用 schedule_lrs 手动等价实现，避免替换参数后调度器失效

    def refresh_params() -> None:
        """密度控制替换参数对象后，把局部变量重新指向新对象。"""
        nonlocal means, scales, quats, opac, shs
        means = params["means"]
        scales = params["scales"]
        quats = params["quats"]
        opac = params["opacities"]
        shs = params["shs"]

    background = torch.tensor([1.0, 1.0, 1.0] if cfg.white_background else [0.0, 0.0, 0.0], device=device)

    def current_scales():
        return torch.exp(scales)

    def current_opacities():
        return torch.sigmoid(opac)

    def normalize_quats():
        with torch.no_grad():
            quats.data = quats.data / (quats.data.norm(dim=-1, keepdim=True) + 1e-8)

    def densify_and_prune(step: int, grad_norm: "torch.Tensor | None") -> tuple[int, int]:
        """自适应密度控制，返回 (加密数, 裁剪数)。

        严格照官方 gsplat 的模式（参考 `gsplat/strategy/ops.py` 与 `default.py`）：
          * 参数用字典管理，扩缩容时**替换 `torch.nn.Parameter` 对象**；
          * 同步迁移优化器里的 Adam 状态（按行拼接/裁剪），绝不重建优化器；
          * 调用时机在 `optimizer.step()` + `zero_grad(set_to_none=True)` 之后。

        两个关键点（都踩过坑）：
          1. 判据不能用 `means.grad` —— 官方顺序里 zero_grad 在密度控制之前，
             那时 .grad 已是 None，会导致一次都不加密。所以由调用方传入
             "backward 之后缓存的位置梯度范数"。
          2. 必须替换参数对象 + 迁移优化器状态。早期"改 p.data + 重建优化器"会报
             "Function ... returned an invalid gradient at index 0 -
              got [N_new, ...] but expected shape compatible with [N_old, ...]"。
        """
        if grad_norm is None:
            return 0, 0
        with torch.no_grad():
            q = _quantile(grad_norm, 1.0 - cfg.densify_quantile)
            thr = min(max(q, cfg.densify_grad_threshold), cfg.densify_grad_max)
            is_grad_high = grad_norm > thr

            # 官方 DefaultStrategy：梯度过阈者，小尺度克隆、大尺度分裂。
            scales_max = torch.exp(params["scales"].data).max(dim=-1).values
            ref = float(scales_max.median()) if scales_max.numel() else 0.0
            is_split = is_grad_high & (scales_max > cfg.split_scale_ratio * ref)
            is_dup = is_grad_high & (~is_split)
            n_before = params["means"].shape[0]
            if cfg.cap_max and n_before >= cfg.cap_max:
                is_dup = torch.zeros_like(is_dup)
                is_split = torch.zeros_like(is_split)

            n_dup, n_split = _grow(params, optimizers, is_dup, is_split)
            refresh_params()

            # 裁剪只用不透明度（官方做法）。早期把"尺度过大"也当裁剪条件，
            # 会立刻删掉刚加密出来的点（427 -> 加密 46 -> 又被裁 78，做了负功）。
            is_prune = torch.sigmoid(params["opacities"].data).squeeze(-1) < cfg.prune_opacity
            n_prune = int(is_prune.sum())
            if n_prune:
                _apply_keep(params, optimizers, ~is_prune)
                refresh_params()

            ctx.info(f"step {step} 密度：阈值={thr:.2e} 克隆 {n_dup} 分裂 {n_split} "
                     f"裁剪 {n_prune} -> {params['means'].shape[0]} 个高斯")
            for p in params.values():
                p.grad = None
        return n_dup + n_split, n_prune

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

        # ---- 反向 ----
        loss.backward()
        # 缓存本步的位置梯度范数：密度控制需要它，但那时 .grad 已被 zero_grad 清掉。
        grad_norm_cache = None
        if means.grad is not None:
            with torch.no_grad():
                grad_norm_cache = means.grad.norm(dim=-1).detach().clone()
        # 探针：确认渲染确实覆盖了像素、且位置参数拿到梯度。
        # 曾经出现 alpha 和 = 0 / means.grad = 0（相机位姿翻转写错，高斯全在相机背后）。
        # 训练"能跑完"不等于"在工作"，这是最省事的健康断言。
        if step == 1:
            ga = means.grad
            gn = float(ga.abs().sum()) if ga is not None else -1.0
            alpha_sum = float(alphas[0].sum())
            ctx.info(f"训练探针（step 1）：alpha 和={alpha_sum:.1f} "
                     f"可见像素={int((alphas[0] > 0.01).sum())} means.grad.abs().sum()={gn:.3e}")
            if ga is None or gn == 0.0:
                ctx.warn(
                    "位置参数梯度为 0：高斯没有落进相机视锥，密度控制不会触发。"
                    "检查 COLMAP 位姿约定（见 scripts/probe_pose_convention.py）。"
                )
            elif alpha_sum <= 0:
                ctx.warn("alpha 全为 0：渲染没有覆盖任何像素，检查深度范围/尺度。")

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        schedule_lrs(step)

        # ---- 结构更新（后向之后进行，官方称之为 post-backward step）----
        # 官方 simple_trainer.py 的顺序就是：
        #   optimizer.step() -> optimizer.zero_grad(set_to_none=True) -> strategy.step_post_backward(...)
        #
        # 但 zero_grad 之后 means.grad 已经是 None，所以判据必须在 backward 之后、
        # step 之前就缓存下来（这一点和官方不同：官方用的是策略自己累积的 grad2d 统计量，
        # 不依赖 .grad）。早期直接读 means.grad -> 恒为 None -> 一次都不加密。
        if cfg.densify_from <= step <= cfg.densify_until and step % cfg.densify_interval == 0:
            densify_and_prune(step, grad_norm_cache)

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
def _update_param_with_optimizer(param_fn, optimizer_fn, params: dict, optimizers: dict, names=None):
    """按官方 gsplat 的做法更新参数与优化器状态。

    这是本项目踩坑最久的一处，必须严格照抄官方模式（gsplat/strategy/ops.py）：
      * **创建新的 `torch.nn.Parameter` 对象**，而不是修改 `p.data`；
      * 把优化器 `param_groups` 指向新对象，并**迁移 Adam 的 state**（exp_avg / exp_avg_sq），
        迁移函数按行拼接（新增行补 0）；
      * 绝不重建优化器（重建会丢动量，还会让 autograd 与旧张量的关联残留，
        导致 "returned an invalid gradient ... expected shape compatible with [N_old]"）。

    参考：`gsplat.strategy.ops._update_param_with_optimizer`。
    """
    import torch

    if names is None:
        names = list(params.keys())

    for name in names:
        old = params[name]
        new = param_fn(name, old)
        params[name] = new
        opt = optimizers.get(name)
        if opt is None:
            continue
        for group in opt.param_groups:
            state = opt.state.pop(old, None)
            if state is not None:
                migrated = {}
                for key, value in state.items():
                    if key == "step" or not torch.is_tensor(value):
                        migrated[key] = value
                    else:
                        migrated[key] = optimizer_fn(key, value)
                opt.state[new] = migrated
            group["params"] = [new]


@torch.no_grad()
def _apply_keep(params: dict, optimizers: dict, keep) -> None:
    """按 keep 掩码裁剪高斯（官方 remove 模式）。"""
    import torch

    sel = torch.where(keep)[0]

    def param_fn(name, p):
        return torch.nn.Parameter(p[sel].clone(), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return v[sel].clone()

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


@torch.no_grad()
def _grow(params: dict, optimizers: dict, duplicate_mask, split_mask) -> tuple[int, int]:
    """加密：对 duplicate_mask 克隆、对 split_mask 分裂（官方 duplicate/split 模式）。"""
    import torch

    n_dup = int(duplicate_mask.sum())
    if n_dup:
        sel_d = torch.where(duplicate_mask)[0]

        def param_fn(name, p):
            merged = torch.cat([p, p[sel_d]], dim=0)
            return torch.nn.Parameter(merged, requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.cat([v, torch.zeros((len(sel_d), *v.shape[1:]), device=v.device)], dim=0)

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers,
                                     names=[k for k in params if k != "shs"])
        # SH 的 DC 分量要一起复制，高阶分量为 0（与官方 split/duplicate 一致）
        shs = params["shs"]
        sh0 = torch.cat([shs[:, :1, :], shs[sel_d][:, :1, :]], dim=0)
        rest = torch.zeros((sh0.shape[0], shs.shape[1] - 1, 3), device=sh0.device, dtype=sh0.dtype)
        new_shs = torch.cat([sh0, rest], dim=1)
        _swap_param("shs", new_shs, params, optimizers,
                    lambda v: torch.cat([v, torch.zeros((len(sel_d), *v.shape[1:]), device=v.device)], dim=0))

    n_split = 0
    if split_mask is not None and bool(split_mask.any()):
        sel_s = torch.where(split_mask)[0]
        n_split = int(sel_s.numel())
        # 分裂：位置抖动 2*scale，尺度除以 1.6（官方做法）
        scales = torch.exp(params["scales"].data[sel_s])
        offsets = torch.randn_like(params["means"].data[sel_s]) * 2.0 * scales

        def param_fn(name, p):
            extra = p[sel_s].clone()
            if name == "means":
                extra = extra + offsets
            elif name == "scales":
                extra = torch.log(torch.exp(extra) / 1.6)
            return torch.nn.Parameter(torch.cat([p, extra], dim=0), requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.cat([v, torch.zeros((len(sel_s), *v.shape[1:]), device=v.device)], dim=0)

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers,
                                     names=[k for k in params if k != "shs"])
        shs = params["shs"]
        sh0 = torch.cat([shs[:, :1, :], shs[sel_s][:, :1, :]], dim=0)
        rest = torch.zeros((sh0.shape[0], shs.shape[1] - 1, 3), device=sh0.device, dtype=sh0.dtype)
        new_shs = torch.cat([sh0, rest], dim=1)
        _swap_param("shs", new_shs, params, optimizers,
                    lambda v: torch.cat([v, torch.zeros((len(sel_s), *v.shape[1:]), device=v.device)], dim=0))
    return n_dup, n_split


def _swap_param(name: str, new_tensor, params: dict, optimizers: dict, optimizer_fn) -> None:
    """单独替换某个参数（shs 需要特殊处理，走这里）。"""
    import torch

    old = params[name]
    new = torch.nn.Parameter(new_tensor, requires_grad=old.requires_grad)
    params[name] = new
    opt = optimizers.get(name)
    if opt is None:
        return
    for group in opt.param_groups:
        state = opt.state.pop(old, None)
        if state is not None:
            migrated = {}
            for key, value in state.items():
                migrated[key] = value if (key == "step" or not torch.is_tensor(value)) else optimizer_fn(value)
            opt.state[new] = migrated
        group["params"] = [new]


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
