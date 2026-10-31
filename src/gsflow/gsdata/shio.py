"""读取 3DGS PLY 的**完整**高斯参数（含高阶 SH），并渲染。

为什么单独建这个模块（一条真实的踩坑）：
训练用 `sh_degree=3`，PLY 里有 3 个 `f_dc_*` + **45 个 `f_rest_*`**。
但导出阶段与 **22 个诊断脚本**都各自手写了一遍"从 PLY 取颜色"，
而且**全部**退化成了 `rgb -> DC -> sh_degree=0` 渲染。

后果（实测，同一模型只换渲染阶数）：
    全 SH 22.10 dB  vs  仅 DC 20.20 dB   ⇒ **低估 1.91 dB**
更要命的是：`sh_degree=0` 抹掉了**视角相关**的那部分外观，
于是对"改善视角相关细节"的优化手段（感知损失、深度失真正则）**天然不敏感**
—— 用一把看不见优化目标的尺子去判断优化有没有用，必然得到假阴性。

修法不是"把 22 个脚本各改一遍"，而是**只留一条读取路径**：
本模块。导出阶段与诊断脚本都从这里取 SH，就不会再分叉。

用法:
    from gsflow.gsdata.shio import read_gaussians_full, render_view
    g = read_gaussians_full(ply_path)          # 含 shs / sh_degree
    img = render_view(g, viewmat, K, W, H)     # 用完整 SH 渲染
"""
from __future__ import annotations

import numpy as np

SH_C0 = 0.28209479177387814


def read_gaussians_full(ply, max_points: int = 0) -> dict:
    """读 3DGS PLY。返回：
        xyz[N,3] float32 | scales[N,3]（**已指数化**，真实尺度）| quats[N,4]（已归一化）
        opacity[N]（已 sigmoid）| rgb[N,3] 0..255（DC 解出，供不支持 SH 的格式用）
        shs[N,(deg+1)^2,3] float32（**完整 SH**）| sh_degree int
        n int
    """
    from plyfile import PlyData

    v = PlyData.read(str(ply))["vertex"]
    names = set(v.data.dtype.names or ())

    def col(n):
        return np.asarray(v[n])

    xyz = np.stack([col("x"), col("y"), col("z")], axis=1).astype(np.float32)
    n = len(xyz)

    if {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
        dc = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=1).astype(np.float32)
        rgb = np.clip(dc * SH_C0 + 0.5, 0.0, 1.0) * 255.0
    elif {"red", "green", "blue"} <= names:
        rgb = np.stack([col("red"), col("green"), col("blue")], axis=1).astype(np.float32)
        dc = (rgb / 255.0 - 0.5) / SH_C0
    else:
        rgb = np.full((n, 3), 128.0, dtype=np.float32)
        dc = np.zeros((n, 3), dtype=np.float32)

    # 尺度：PLY 存的是 log 尺度
    if {"scale_0", "scale_1", "scale_2"} <= names:
        scales = np.exp(np.stack([col("scale_0"), col("scale_1"), col("scale_2")], axis=1)
                        .astype(np.float32))
    else:
        scales = np.full((n, 3), 0.01, dtype=np.float32)

    if {"rot_0", "rot_1", "rot_2", "rot_3"} <= names:
        quats = np.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], axis=1
                         ).astype(np.float32)
        quats = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    else:
        quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))

    op = (1.0 / (1.0 + np.exp(-col("opacity").astype(np.float32)))
          if "opacity" in names else np.ones(n, dtype=np.float32))

    # ---- 完整 SH：f_rest 是**通道主序**（先所有基的 R，再 G，再 B）----
    f_rest = sorted((k for k in names if k.startswith("f_rest_")),
                    key=lambda s: int(s.rsplit("_", 1)[1]))
    per_ch = len(f_rest) // 3
    n_bases = per_ch + 1
    deg = int(round(n_bases ** 0.5)) - 1
    if per_ch and (deg + 1) ** 2 == n_bases:
        shs = np.zeros((n, n_bases, 3), dtype=np.float32)
        shs[:, 0, :] = dc
        for c in range(3):
            for k in range(per_ch):
                shs[:, k + 1, c] = col(f"f_rest_{c * per_ch + k}").astype(np.float32)
    else:
        shs = dc.reshape(n, 1, 3).copy()
        deg = 0

    order = np.argsort(-op)
    if max_points and len(order) > max_points:
        order = order[:max_points]
    return {
        "xyz": xyz[order], "scales": scales[order], "quats": quats[order],
        "opacity": op[order], "rgb": rgb[order], "shs": shs[order],
        "sh_degree": int(deg), "n": int(len(order)),
    }


def render_view(g: dict, viewmat, K, width: int, height: int,
                device: str = "cuda", render_mode: str = "RGB",
                rasterize_mode: str = "classic", keep_min_opacity: float = 0.0):
    """用 gsplat 以**完整 SH** 渲染一帧。返回 (colors[H,W,C], alphas[H,W,1])。

    `viewmat`/`K` 可以是 numpy 或 torch；内部统一转 torch。
    """
    import torch
    from gsplat import rasterization

    def T(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

    opac = T(g["opacity"]).reshape(-1).clamp(0, 1)
    means, quats, scales, colors = T(g["xyz"]), T(g["quats"]), T(g["scales"]), T(g["shs"])
    # 与预览一致：过滤掉几乎不可见的高斯（远处浮点会把画面弄脏）
    if keep_min_opacity and float(keep_min_opacity) > 0:
        keep = opac > float(keep_min_opacity)
        if int(keep.sum()) >= 16:
            means, quats, scales, colors, opac = (means[keep], quats[keep], scales[keep],
                                                  colors[keep], opac[keep])
    out = rasterization(
        means=means, quats=quats, scales=scales,
        opacities=opac,
        colors=colors,
        viewmats=T(viewmat)[None], Ks=T(K)[None],
        width=int(width), height=int(height),
        sh_degree=int(g.get("sh_degree", 0)),
        near_plane=0.01, far_plane=1e10, packed=False, render_mode=render_mode,
        rasterize_mode=rasterize_mode,
    )
    return out[0][0], out[1][0]
