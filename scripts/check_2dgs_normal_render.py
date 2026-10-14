"""验证 gsplat 的**法线渲染**那一趟能否正常工作（2DGS 法线正则依赖它）。

背景（踩坑）：`sh_degree=None` 时 gsplat 要求 `colors` 是 `[N, D]`，
传 `[N, 1, 3]` 会抛 `AssertionError: torch.Size([N,1,3])`。
而调用处当时用 `try/except` 吞掉了异常 -> 法线项静默为 None，
日志里只看得到 depth 项，表现为"加了法线正则但完全没生效"。

本脚本直接验证：
  1. `gaussian_normals` 的输出形状与单位性
  2. `render_mode="D"` 能拿到累计深度，且与几何一致
  3. `render_mode="RGB"` + `colors=[N,3]` + `sh_degree=None` 能渲染法线
  4. `two_dgs_regularizers` 同时返回 depth 与 normal 两项（且 normal 非 None）

用法：scripts\\gpu_python.cmd scripts\\check_2dgs_normal_render.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ⚠️ 必须先应用运行环境再 import torch/gsplat：
# 它负责加载 MSVC/CUDA 环境并**预加载已编译好的 gsplat 内核**。
# 不调用的话会在 torch 的 cpp_extension 里尝试重新编译，
# 而那条路径在本机沙箱下会以 `UnicodeDecodeError: 'cp1' codec ...` 失败
# （已知坑：环境探测子进程输出编码问题）。
from gsflow import paths  # noqa: E402

paths.apply_runtime_env()

import torch  # noqa: E402

from gsflow.trainers.gsplat_trainer import (  # noqa: E402
    gaussian_normals,
    two_dgs_regularizers,
)

dev = "cuda"
torch.manual_seed(0)
N = 500
# 在相机前方 z≈2 处放一团扁平高斯（模拟一个正对相机的平面）
means = torch.zeros(N, 3, device=dev)
means[:, 0] = torch.rand(N, device=dev) * 1.0 - 0.5
means[:, 1] = torch.rand(N, device=dev) * 1.0 - 0.5
means[:, 2] = 2.0
quats = torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(N, 1)   # 无旋转
scales = torch.stack([torch.full((N,), 0.05, device=dev),
                      torch.full((N,), 0.05, device=dev),
                      torch.full((N,), 0.001, device=dev)], dim=-1)  # 最短轴 = z
opac = torch.full((N,), 0.9, device=dev)
shs = torch.full((N, 1, 3), 0.5, device=dev)                     # 中灰
vm = torch.eye(4, device=dev)
K = torch.tensor([[80.0, 0.0, 48.0], [0.0, 80.0, 40.0], [0.0, 0.0, 1.0]], device=dev)
W, H = 96, 80

ok = True

# 1)
n = gaussian_normals(quats, scales)
print(f"gaussian_normals: shape={tuple(n.shape)} 单位性偏差={float((n.norm(dim=-1)-1).abs().max()):.2e}")
if tuple(n.shape) != (N, 3) or float((n.norm(dim=-1) - 1).abs().max()) > 1e-5:
    print("  **失败**")
    ok = False

# 2) 主渲染（拿 alpha），再单独渲染深度
from gsplat import rasterization  # noqa: E402

colors, alphas, meta = rasterization(
    means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
    viewmats=vm[None], Ks=K[None], width=W, height=H, sh_degree=0,
    near_plane=0.01, far_plane=1e10, packed=False, render_mode="RGB")
d, _, _ = rasterization(
    means=means, quats=quats, scales=scales, opacities=opac, colors=shs,
    viewmats=vm[None], Ks=K[None], width=W, height=H, sh_degree=0,
    near_plane=0.01, far_plane=1e10, packed=False, render_mode="D")
dm = d[0][..., 0]
valid = alphas[0][..., 0] > 0.5
print(f"深度渲染: 有效像素 {int(valid.sum())}  深度中位数 "
      f"{float(dm[valid].median()) if bool(valid.any()) else float('nan'):.4f}（应≈2.0）")
if bool(valid.any()) and abs(float(dm[valid].median()) - 2.0) > 0.1:
    print("  **深度与几何不符**")
    ok = False

# 3) 法线渲染：colors 必须是 [N,3]
col_n = (n * 0.5 + 0.5).reshape(-1, 3)
n_rgb, a_n, _ = rasterization(
    means=means, quats=quats, scales=scales, opacities=opac, colors=col_n,
    viewmats=vm[None], Ks=K[None], width=W, height=H, sh_degree=None,
    near_plane=0.01, far_plane=1e10, packed=False, render_mode="RGB")
print(f"法线渲染: shape={tuple(n_rgb.shape)} alpha 均值={float(a_n.mean()):.4f}")
if tuple(n_rgb.shape) != (1, H, W, 3):
    print("  **形状不符**")
    ok = False

# 4) 正则函数应同时返回两项
class _Cfg:
    lambda_depth_distortion = 0.05
    lambda_normal = 0.1
    depth_distortion_mode = "tv"


reg = two_dgs_regularizers(
    cfg=_Cfg(), means=means, quats=quats, scales=scales, opacities=opac,
    shs=shs, kk=K, vm=vm, width=W, height=H, alphas=alphas, sh_degree=0)
print(f"two_dgs_regularizers -> depth={reg.get('depth')}  normal={reg.get('normal')}"
      + (f"  normal_error={reg.get('normal_error')}" if reg.get("normal_error") else ""))
if reg.get("normal") is None:
    print("  **法线项为 None —— 法线正则没有生效**")
    ok = False
else:
    print(f"  法线一致性损失 = {float(reg['normal']):.4e}（有限且非负即正常）")

print("\n结论:", "gsplat 的深度/法线渲染均可用，2DGS 两项正则都能生效" if ok
      else "**有问题，见上面标记**")
raise SystemExit(0 if ok else 1)
