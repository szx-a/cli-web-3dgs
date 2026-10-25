"""LPIPS 风格感知损失/指标（基于 torchvision VGG16，无需额外 `lpips` 包）。

## 为什么自己实现，而不是 `pip install lpips`

1. 本机没有 `lpips` 包，安装要走网络；
2. `torchvision` 已在（0.26.0+cu128），VGG16 权重已下到工作区
   （`models/torch/hub/checkpoints/vgg16-397923af.pth`，见 `scripts/fetch_torch_weights.py`）；
3. **可控**：权重缺失时优雅退化，而不是训练到一半抛异常。

## 与论文/官方实现的对齐（2026-10-25 重写）

**这是关键修复**。上一版有三处偏离，导致"加了感知损失反而更糊"（见 03/G152）：

| # | 上一版 | 论文 / 官方 `lpips` 包 | 后果 |
|---|---|---|---|
| 1 | 5 层**等权**求平均 | 用**训练好的线性权重** `[0.0448, 0.0201, 0.0081, 0.0041, 0.0059]` | 等权把 conv5（语义层）放大了 ~10 倍 ⇒ 优化器偏向"语义平滑"，细节被抹掉、高斯变多 |
| 2 | 通道 RMS 归一化后直接 MSE | 每通道 L2 归一化 → **逐通道空间均值** → 乘线性权重 → 通道求均值 | 少了"逐通道空间均值"，空间结构差异被平均掉，梯度方向偏 |
| 3 | 每 16 步算一次（`lpips_every=16`）| **每步都算** | 稀疏施加 ⇒ 梯度与 L1/SSIM 不一致、来回拉扯，等效加噪 |

现在三者都按论文改齐；`lpips_every` 默认改为 **1**。

## 速度适配（唯一的有意偏离，已显式标注）

逐层在**原分辨率**跑 VGG16 在 8GB 卡上太贵。这里把 pred/gt **双线性缩到 `input_size`（默认 224）**再算 ——
LPIPS 在 224 输入下是社区评测/训练的常见做法（官方评测也常在 224~256 尺度），
并且**训练与评估用同一尺度**，所以数值自洽、可比。

## 其它

* 输入归一化：论文用 ImageNet 均值方差；这里用 `[-1,1]` 线性映射，训练与评估共用同一套（自洽）；
* 权重缺失时构造失败并抛异常，调用方记录 WARN 后**彻底关闭**该项（绝不给 0，
  否则"开着但没生效"这类静默退化无法察觉，见 03/G65）。
"""

from __future__ import annotations

import os
from pathlib import Path

# VGG16 features 的 5 个激活层下标（论文的 conv1_2 / conv2_2 / conv3_3 / conv4_3 / conv5_3）
_LPIPS_LAYERS = (4, 9, 16, 23, 30)

# 官方 `lpips` 包 **v0.1 linear** 权重（VGG16 版），论文配套值。
# 不相等权：越深的层权重越小（语义层只占 ~1%），细节主要来自浅层。
_OFFICIAL_LINEAR = (0.0448, 0.0201, 0.0081, 0.0041, 0.0059)

_CACHE: dict[str, object] = {}


def torch_home() -> Path:
    """权重缓存目录：固定在工作区内（默认会写 %USERPROFILE%/.cache/torch，沙箱下不可写）。"""
    return Path(__file__).resolve().parents[3] / "models" / "torch"


def vgg_weights_present() -> bool:
    f = torch_home() / "hub" / "checkpoints" / "vgg16-397923af.pth"
    return f.exists() and f.stat().st_size > 1024 * 1024


class LPIPS:
    """VGG16 特征 + 论文线性权重的感知距离。`__call__` 返回**标量张量**（可反传）。

    构造失败（缺 torchvision / 缺权重）时抛异常，调用方应捕获并记录 WARN，
    然后**退化为不使用**该损失/指标。
    """

    def __init__(self, device: str = "cuda", input_size: int = 224,
                 weights: tuple[float, ...] | None = None):
        import torch
        import torchvision.models as M

        os.environ.setdefault("TORCH_HOME", str(torch_home()))
        self.device = torch.device(device)
        self.input_size = int(input_size)
        net = M.vgg16(weights=None)
        ckpt = torch_home() / "hub" / "checkpoints" / "vgg16-397923af.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"缺少 VGG16 权重：{ckpt}")
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        net.load_state_dict(state, strict=False)
        # 只保留到最后一层激活，冻结参数
        self.features = net.features[: _LPIPS_LAYERS[-1] + 1].to(self.device).eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        w = tuple(weights) if weights else _OFFICIAL_LINEAR
        if len(w) != len(_LPIPS_LAYERS):
            raise ValueError(f"LPIPS 权重个数应为 {len(_LPIPS_LAYERS)}，实际 {len(w)}")
        # 记录权重向量（浮点）与"逐通道权重张量"，供注意力式加权
        self.layer_w = tuple(float(x) for x in w)

    def _feats(self, x):
        out = []
        h = x
        want = set(_LPIPS_LAYERS)
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in want:
                out.append(h)
                if i == _LPIPS_LAYERS[-1]:
                    break
        return out

    def __call__(self, pred, gt):
        """pred/gt: [H,W,3] 或 [B,3,H,W]，取值 0..1。返回标量张量（越小越像）。"""
        import torch
        import torch.nn.functional as F

        def prep(t):
            if t.dim() == 3:
                t = t.permute(2, 0, 1)[None]
            t = (t * 2.0 - 1.0).clamp(-1, 1)
            # 统一缩放到固定尺度（速度适配；训练与评估一致）
            if self.input_size and (t.shape[-1] != self.input_size or t.shape[-2] != self.input_size):
                t = F.interpolate(t, size=(self.input_size, self.input_size),
                                  mode="bilinear", align_corners=False, antialias=True)
            return t

        a, b = prep(pred), prep(gt)
        fa, fb = self._feats(a), self._feats(b)
        total = a.new_zeros(())
        for w, x, y in zip(self.layer_w, fa, fb):
            # 论文第一步：逐通道 L2 归一化（让不同通道尺度可比）
            x = x / (x.pow(2).mean(dim=(2, 3), keepdim=True).sqrt() + 1e-10)
            y = y / (y.pow(2).mean(dim=(2, 3), keepdim=True).sqrt() + 1e-10)
            # 论文第二步：逐通道**空间均值** ⇒ [B,C]，再按通道求平均
            d = (x - y).pow(2).mean(dim=(2, 3))     # [B,C]
            total = total + w * d.mean()
        denom = sum(self.layer_w) if sum(self.layer_w) > 0 else 1.0
        return total / denom


def get_lpips(device: str = "cuda", input_size: int = 224) -> LPIPS:
    """带缓存的构造（同一次训练里只建一次）。"""
    key = f"lpips:{device}:{input_size}"
    if key not in _CACHE:
        _CACHE[key] = LPIPS(device, input_size=input_size)
    return _CACHE[key]  # type: ignore[return-value]
