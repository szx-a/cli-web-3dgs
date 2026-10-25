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
    # 高斯数量上限（显存守护）。**必须与 config.DEFAULTS['train']['cap_max'] 保持一致** ——
    # 运行时以配置系统为准，dataclass 只是兜底；两处不一致会造成"改了不生效"的错觉。
    # （`tests/test_defaults_consistency.py` 会钉住这条。）
    cap_max: int = 600_000
    ssim_weight: float = 0.2
    # 是否使用**原始 3DGS 的单边 eps** 写法。默认 False（修正写法）。
    # 修正原因：本素材背景 42%~85% 是纯黑，单边 eps 会把常数图的 SSIM 拉到 ~0.9（应=1.0），
    # 实测系统性拉低 0.047（占 SSIM 项的 14%）且**随背景占比漂移**。
    # 保留开关仅用于受控对照（证明修正确实有效）。
    ssim_legacy_eps: bool = False
    densify_from: int = 500
    # 密度控制窗口的**上界**。这是本项目在真实数据上最敏感的一个参数。
    #
    # 官方默认是 15000（面向几百~上千张视图的大场景）。但在**实物环绕拍摄**
    # （几十~一两百张视图）上，后期加密纯粹是在过拟合，会显著劣化结果。
    # 真实素材实测（122 视图，600px，10k 迭代，只改这一个参数）：
    #     densify_until=15000(默认)  145,442 高斯   最终 20.88 dB（峰值 25.77 @8000，之后退化）
    #     densify_until= 4000         74,447 高斯   最终 28.62 dB（不再退化）
    #     densify_until= 2500         57,174 高斯   最终 28.71 dB（不再退化）
    # 即"少一半高斯、PSNR 反而高 7.8 dB"。故默认下调到 4000，
    # 大场景请显式调大（--set train.densify_until=15000）。
    densify_until: int = 4000
    densify_interval: int = 100
    # 加密判据：位置梯度范数超过阈值。
    # 实测校准：某场景 step 500 时 grad_norm 均值约 2e-4、最大约 1.2e-2；
    # 随着优化推进梯度会明显衰减（step 3000 已掉到 1e-5 量级），固定阈值会导致
    # "只在前几步加密过一次"。因此采用**分位数自适应阈值**：
    # 加密判据：位置梯度的范数阈值。
    # 官方 gsplat DefaultStrategy 用的是**固定绝对阈值**（grow_grad2d=0.0002，
    # 作用在按视图累积后的 2D 屏幕梯度上）。早期我改成"分位数自适应阈值"，
    # 结果梯度一衰减阈值就跟着降 -> **无限膨胀**（实测 6000 步涨到 6 万高斯并
    # 让 PSNR 从 33.75 掉到 21.47）。梯度随优化自然衰减本身就是停止加密的信号，
    # 不需要（也不应该）人为跟随。这里默认用固定阈值。
    densify_grad_threshold: float = 2e-4
    # 加密判据的来源（本轮审计的核心修正）：
    #   "grad2d"（默认）—— 官方口径：`means2d.grad` 屏幕空间梯度 / 可见次数。
    #       `densify_grad_threshold` 在此口径下的含义与官方 `grow_grad2d` 一致。
    #   "grad3d"       —— 旧行为：`means.grad` 世界空间梯度 + 固定阈值。
    #       保留以便 A/B 复现。
    # 实测差异（probe_densify_criterion.py）：两者量纲差 ~45 倍，
    # 且选中的高斯集合 **IoU 仅 0.065** —— 加密位置几乎完全不同。
    # 用 grad3d 时 densify_until=15000 会让高斯涨 4.3 倍而 PSNR **降 0.874 dB**。
    densify_criterion: str = "grad2d"
    # grad2d 口径下的阈值（官方 grow_grad2d 的常用值）。
    # 与 grad3d 的 2e-4 不是同一个量纲，**不要混用**。
    densify_grad2d_threshold: float = 2e-4
    # 光栅化模式（社区做法）：
    #   "classic"（默认）—— 3DGS 原版做法。
    #   "antialiased"    —— Mip-Splatting 的思路：给 2D 投影加低通滤波，
    #     抑制采样率不足导致的锯齿/闪烁。训练分辨率与展示分辨率不同、
    #     或视角密集时收益明显。
    # 保守起见默认 classic；实测对比见 docs/03。
    rasterize_mode: str = "classic"
    # 可选：>0 时改用"梯度的 (1-q) 分位"作阈值（仅用于实验对比）
    densify_quantile: float = 0.0
    densify_grad_max: float = 5e-3           # 自适应阈值的上限
    # 绝对梯度致密化（AbsGS）。官方 `DefaultStrategy` 支持 `absgrad`；
    # 开启后栅格化会额外产出 `means2d.absgrad`（**必须同时给 rasterization 传 absgrad=True**），
    # 致密判据改用绝对值和而不是"正负相消后的梯度"，能恢复更细的结构。
    # 默认关闭，需实测对照后再决定是否改默认。
    absgrad: bool = False
    # ---- 2DGS 参数化（压平高斯）----
    # 社区做法（2D Gaussian Splatting, SIGGRAPH'24）：把高斯**压平成 2D 圆盘**，
    # 于是"最短轴"就是该面片的**法线**，几何语义明确。3DGS 的椭球没有这个约束，
    # 这也是为什么直接把 2DGS 的两个损失项搬到 3DGS 上会有害（见 docs/03 G52）：
    # 损失项与参数化是一套的，只搬前者不匹配后者会与"拟合辐射场"冲突。
    #
    # 实现方式：只钳住**最短轴**的 log 尺度上限（其余两轴自由），
    # 即 scale_min = exp(min(log_scale, flatten_log_scale))。
    # 这样圆盘半径仍可自由学习，但厚度被压到极小 ⇒ 法线可信、深度沿射线集中。
    flatten_2dgs: bool = False
    # 最短轴的 log 尺度上限。exp(-6)≈0.0025，相对场景尺度是"极薄"；
    # 太大会失去压平效果，太小可能让光栅化数值不稳（有效像素过少）。
    flatten_log_scale: float = -6.0
    # 以下三个比例都相对 scene_scale（相机中心到均值距离），取自官方 DefaultStrategy：
    #   grow_scale3d  = 0.01 -> 尺度 <= 0.01*scene_scale 走"克隆"
    #   grow_scale3d  >  0.01 -> 走"分裂"
    #   prune_scale3d = 0.1  -> 尺度 > 0.1*scene_scale 直接裁剪（防发散的关键约束）
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1
    # 与官方一致：尺度裁剪只在 step > opacity_reset_interval 之后启用
    prune_scale_after_reset: bool = True
    # 分裂出的高斯按官方做法把不透明度调低，避免分裂后过度堆积
    revised_opacity: float = 0.1
    # 裁剪阈值：不透明度低于此值的高斯被删除（官方 0.005）
    prune_opacity: float = 0.005
    # 不透明度重置：**默认关闭**，这是实测后的决定，不是漏实现。
    #
    # 已按官方补齐的三件事（`step_post_backward` 的完整顺序）：
    #   * 重置的**值**是 `prune_opa * 2.0`（0.01），不是 `revised_opacity`（0.1）
    #   * 重置后**暂停精修** `pause_refine_after_reset` 步
    #   * 尺度裁剪以 `step > reset_every` 为门槛
    #
    # 但实测仍不划算（122 视图 / 15000 迭代 / mask_loss_until=1.0）：
    #
    #   A 组 densify_until=4000  + reset_every=2000 -> 中段崩到 13~17，末段 26.51
    #   B 组 densify_until=15000 + reset_every=3000 -> 峰值 23.92，末段 17.63，
    #                                                 高斯数 279,534（不重置仅 78,526）
    #
    # 根因：官方 `_update_state()` 的加密判据用的是**累积梯度 ÷ 可见次数**：
    #       grad2d.index_add_(...); grads = grad2d / count.clamp_min(1)
    # 而本项目用的是"当步梯度模长直接比绝对阈值"。
    # 重置把不透明度压到 0.01 后，所有高斯的梯度都会突然变大，
    # 这套判据就会**疯狂加密** -> 高斯爆炸（27.9 万 vs 7.9 万）。
    #
    # ⇒ 要启用重置，必须先补上官方的"梯度累积 ÷ 可见次数"统计量
    #   （即把 gsplat 的 DefaultStrategy 完整移植过来）。
    #   在那之前，默认 0（关闭）是更稳的选择。
    opacity_reset_interval: int = 0
    # ---- 不透明度重置的**替代方案**（用户方案第 5 项）----
    # 为什么需要替代：官方的周期重置是一次**不连续跳变**（把 logit 不透明度硬钳到
    # logit(0.01)）。实测后果：所有高斯的梯度突然变大 ⇒ 触发致密化 ⇒
    # 高斯数 279,534（不重置仅 78,526），并且"最终检查点"会被污染成一片黑。
    # 下面三个替代都**不做跳变**，因此不引入那一波梯度尖峰。
    #
    # 方案 C：梯度感知的不透明度衰减（本实现，最简、最稳）
    #   每 `opacity_decay_interval` 步把**实际不透明度**乘以 `opacity_decay`，
    #   让"贡献低"的高斯平滑淡出，而不是周期性地一刀切。
    #   实现要点：不透明度是 logit 参数化（`data` 是 logit），所以
    #       σ(x·k) = σ(x)/(σ(x) + k·(1-σ(x)))     （k 为衰减系数）
    #   是**精确**的乘法衰减在 logit 域的等价形式，不需要先转 sigmoid 再转回来。
    #   1.0 = 关闭。
    opacity_decay: float = 1.0
    opacity_decay_interval: int = 50
    # 方案 A/B 的实现位置见 `utilization_prune_*`：按"利用率"（梯度贡献）剪枝，
    # 只剪真正没用的，不动其他高斯。
    utilization_prune_interval: int = 0
    # 剪枝**只在这个步数之前**发生（0 = 同时约束在 iterations 内）。
    # ⚠️ 实测教训：第一版按 interval=400 周期性剪 20%，8000 步内剪了约 20 次，
    #    `0.8^20 ≈ 0.012` ⇒ 高斯 139,719 → **7,070**、PSNR → 21.04。
    #    用户方案的原意是"**每次重置后**剪一次"，本项目不用重置，
    #    于是它变成了"每 400 步都剪" —— 触发条件用错了。
    #    现在用 `*_until` 把剪枝限制在**少数几次**（配合大 interval）。
    utilization_prune_until: int = 0
    utilization_prune_quantile: float = 0.2
    # ---- 方案 A：恢复感知剪枝（recovery-aware pruning）----
    # 用户方案原意："每次重置后的 300 步，对不透明度**恢复最慢**的 20% 剪枝"。
    # 原理：正常高斯在所有视角贡献为正，被压下去后**恢复快**；
    #       而浮空物（伪影）只在少数视角贡献，不透明度震荡、**恢复慢**。
    #
    # 本项目的调整（**不启用重置**，因为重置实测 −0.59 dB）：用方案 C 的**平滑衰减**
    # 代替重置作为"打压"，然后看谁**没恢复回来**：
    #   1) 在衰减周期上拍一张不透明度的"快照"（被打压前）；
    #   2) 再等 `recovery_window` 步，比较当前不透明度与快照之比；
    #   3) 比值（恢复率）最低的 `recovery_prune_quantile` 剪掉。
    # 这样保留了方案 A 的核心判据（**恢复速度**），又不依赖被否掉的重置。
    recovery_prune_interval: int = 0
    recovery_prune_quantile: float = 0.2
    recovery_window: int = 300
    # 同 utilization_prune_until：限制剪枝只在少数几步发生
    recovery_prune_until: int = 0
    # 重置时压到的目标不透明度；<=0 表示用 prune_opacity*2.0（官方行为）
    reset_opacity: float = 0.0
    # 重置后暂停精修的步数（0=不暂停）。建议设为训练图片数。
    pause_refine_after_reset: int = 0

    # 密度控制策略：
    #   "builtin"  —— 本项目自写的简化实现（3D 梯度 + 固定阈值，实测校准过）
    #   "official" —— 直接复用官方 `gsplat.DefaultStrategy`（含 ops.py 的
    #                 duplicate/split/remove/reset_opa，以及 grad2d/count 统计量）
    #
    # 为什么要提供 official：自写实现的判据（3D 世界空间梯度 + 固定绝对阈值）
    # **不可跨分辨率迁移**，而且缺少配套的浮点抑制，导致渲染上出现半透明"雾"
    # （诊断图证实：几何正确、但背景被浮点抬高 -> PSNR 掉到 14 dB）。
    # 我试过零散抄官方的几个公式（grad2d/count、屏幕空间归一化），三次都失败
    # （13~17 / 23.9 / 9.4 dB）—— 因为官方那套是**整体自洽**的机制，
    # 拆一半移植比不移植更糟。所以正确做法是**直接复用官方类**。
    densify_strategy: str = "builtin"
    eval: bool = True
    # 相机轴约束：默认 False = 不翻转，与官方 gsplat 的 COLMAP 加载器一致。
    # 只保留用于对照实验；正常流程不要打开。
    flip_camera: bool = False
    # 少于这么多张图就不留评估集（留出 10% 会明显削弱小数据集训练），
    # 改为在训练视图上算 PSNR 并在指标里标注，避免"PSNR 恒为 0"的误读。
    min_imgs_for_holdout: int = 10
    eval_interval: int = 2000
    # 日志用的高频评估固定用几个留出视图（0=用全部留出视图）。
    # 存在的理由：训练视图是循环取的，而 eval_interval 与循环长度不互质时会产生
    # **度量混叠**（同一相位反复评估同一张难图 -> 曲线周期性骤降，误判训练不稳）。
    eval_quick_views: int = 3
    white_background: bool = False
    mask_loss_until: float = 1.0
    # 掩码损失口径：
    #   "composite"（默认）—— 把 GT 在掩码外合成为背景色，损失在**全图**上计算。
    #      前景监督真实像素、背景监督"必须渲染成背景色"、各视角 loss 尺度一致。
    #      （与 3DGS / gsplat 官方示例同源。修复了"掩码外无监督 ⇒ 渲染糊出去/破碎"。）
    #   "inside" —— 旧行为：只在掩码内取均值。loss 量级随前景占比漂移（12%~55% ⇒ 4.6 倍），
    #      且掩码外完全无监督。保留仅供复现旧结果与受控对照。
    mask_loss_mode: str = "composite"
    # ---- 正则项（参考 Faster-GS / 3DGS-MCMC；默认 0 = 不启用，保持既有行为不变）----
    # 开启后能显著压制"半透明浮点"（表现为点云/网格周围的雾状杂点）。
    # 3DGS-MCMC 论文与 Faster-GS 都用 0.01（两者同时开）。
    opacity_reg: float = 0.0
    scale_reg: float = 0.0
    max_scale_ratio: float = 0.05   # 每步尺度上限（× scene_scale），0=关闭
    bg_alpha_reg: float = 0.1       # 掩码外 alpha 正则权重（0=关闭）
    max_aspect: float = 50.0        # 长细比上限（0=关闭）
    lr_means_scene_scale: bool = False   # means 学习率乘以 scene_scale（官方做法）
    lr_shs: float = 0.0025               # SH 学习率（官方 = 1.25e-4）
    auto_extend_densify_when_reset: bool = True   # 重置与密度窗口联动
    mask_erode_px: int = 0                  # 损失用掩码腐蚀像素数（修轮廓白边）
    sh_rest_reg: float = 0.003              # f_rest 的 L2 正则权重（3e-3：彩度 −19%、SSIM −0.0005）
    sh_rest_max: float = 0.0                # f_rest 逐系数幅度上限（0=关）
    opacity_crisp_reg: float = 0.02          # 不透明度清晰度正则（推实；实测 SSIM −0.0003）
    opacity_crisp_warmup: float = 0.7        # 前 N% 迭代不施加清晰度正则（0~1）
    opacity_floor: float = 0.10              # 不透明度下限（实测几乎免费：op<0.1 85%→4%）
    opacity_floor_reg: float = 0.5           # 下限正则权重：惩罚 relu(floor-op)
    opacity_floor_warmup: float = 0.5        # 前 N% 迭代不施加
    # 训练结束时的最小不透明度清理（Faster-GS: MIN_OPACITY_AFTER_TRAINING = 1/255）。
    # 不透明度低于该值的高斯在收尾时直接删掉；0 表示不清理。
    min_opacity_final: float = 0.0
    # 不透明度学习率。官方 3DGS 用 0.05，但那**配合每 3000 步的不透明度重置**；
    # 我们默认关闭重置，0.05 会导致不透明度持续塌陷成"薄雾"（见 G43）。
    # Faster-GS / 3DGS-MCMC 用 0.025，这里对齐。
    lr_opacities: float = 0.025
    # ---- 学习率调度（用户方案第 1、2 项）----
    # lr_schedule: "cosine"（默认，线性预热 + 余弦退火）| "exp"（旧的纯指数衰减）
    # warmup_iters: 线性预热的步数。0 = 关闭预热。"前 5%~10% 步数"由调用方决定；
    #   默认给 0（保持旧行为，便于受控对照），显式打开后才生效 —— 这样
    #   "加了预热"与"没加预热"可以在**同一份数据**上直接对比。
    lr_schedule: str = "cosine"
    warmup_iters: int = 0
    # 余弦退火的下限（占峰值的比例）。用户方案建议衰减到 1/10 或更低。
    lr_min_ratio: float = 0.1
    # 四元数学习率。用户方案建议降到 5e-4，但**受控实测不支持**：
    # 同数据 20000 迭代下 1e-3 → 24.264 dB，5e-4 → 24.187 dB（反低 0.077）。
    # 保持 1e-3；若日后换参数化（如压平高斯）应重新验证。
    lr_quats: float = 1e-3
    # ---- 感知损失 LPIPS（用户方案第 3 项）----
    # ⚠️ 噪声底提醒：同配置重复训练时 PSNR 散布约 **0.09 dB**（实测 rep-1/2/3）。
    #    小于该量级的差异**不能**作为"有效/无效"的依据。
    # 历史：我曾据一轮对照把它判为有害并删掉代码，而那轮用的是 `sh_degree=0` 渲染
    # （丢高阶 SH ⇒ 低估 1.91 dB，且恰好抹掉感知损失要改善的视角相关细节）⇒ 假阴性。
    lambda_lpips: float = 0.0
    # 每 N 步计算一次（VGG 前向开销与显存都不小）
    lpips_every: int = 1                    # 社区做法：每步都算（见 03/G152）
    # ---- 多指标评估（用户方案第 6 项）----
    # 在 metrics.json 里同时记录 PSNR / SSIM / LPIPS，并额外保存 best_ssim / best_lpips
    # 的检查点。理由：PSNR 高但 SSIM/LPIPS 差的模型视觉上往往是模糊的，
    # 不能只用 PSNR 决定"哪个检查点用于导出"。
    eval_metrics: bool = True
    # 用哪个指标挑选"最佳检查点"（导出阶段优先使用它）：psnr | ssim | lpips
    best_metric: str = "psnr"
    # ---- 2DGS 训练侧正则（参考 2D Gaussian Splatting, SIGGRAPH'24）----
    # 为什么需要（本项目的实测动机）：导出网格时我们从深度图反投影取点，
    # 而 3DGS 的"累计深度"是沿射线的**加权平均** —— 一个像素的深度被几十个
    # 高斯混合出来，表面位置是模糊的，于是：
    #   * 反投影点云在表面法线方向上有厚度 -> 泊松得到"碎面"
    #   * 深度梯度算出的法线不稳（实测网格法线翻转 31~44%）
    # 2DGS 用两个正则解决：深度失真抑制（让同像素的高斯挤到同一深度）
    # 与法线一致性（渲染法线要等于深度图的几何法线）。两者都**默认关闭**，
    # 需要显式开启（见下面两个权重）。
    lambda_depth_distortion: float = 0.0
    # ⚠️ 实测结论：直接套 2DGS 的权重（0.05 / 0.1）在 3DGS 上**有害** ——
    # 公平对照下 PSNR 22.15 -> 18.18 dB、网格法线翻转 6.5% -> 35.2%、
    # scale 中位数被撑大 3.8 倍，且两项正则的数值都没有下降趋势。
    # 原因：2DGS 的权重与它"压平高斯 + 显式法线"的参数化自洽，
    # 而 3DGS 的椭球没有这个约束。要真正走 2DGS 路线需要改参数化，不是只加损失项。
    # 因此**默认关闭**；接口保留，便于换更小的权重或换参数化后再实验。
    lambda_normal: float = 0.0
    # 正则的施加间隔：每 N 步算一次。0/1 = 每步都算。
    # 深度正则需要**额外一次渲染**（render_mode="D"），所以默认隔几步算，控制开销。
    reg_interval: int = 8
    # 深度失真正则的形式：
    #   "tv"   -> |Δ_x D| + |Δ_y D|（一阶差分，惩罚深度图不平滑；实现简单、必收敛）
    #   "lap"  -> |Δ²D|（二阶，更接近 2DGS 的"深度应分段恒定"意图）
    depth_distortion_mode: str = "tv"
    # 训练侧深度语义："expected"=期望深度 ED（与导出一致）；"accum"=Σwz（仅对照）
    depth_mode: str = "expected"
    seed: int = 42
    log_every: int = 200


def _quat_to_rotmat(q: "torch.Tensor") -> "torch.Tensor":
    """四元数 (w,x,y,z) -> 旋转矩阵 [N,3,3]。用于取高斯的法线方向。"""
    import torch

    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    col0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
    col1 = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1)
    col2 = torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)
    return torch.stack([col0, col1, col2], dim=-2)


def gaussian_normals(quats: "torch.Tensor", scales: "torch.Tensor",
                     axis: str = "z") -> "torch.Tensor":
    """每个高斯的表面法线。

    **用旋转后的 +z 轴（第 3 列），不是"最短轴"。** 这是实测结论，推翻了原先的实现：

    原先用 `argmin(scales)` 取"最短轴"，理由是"2DGS 把高斯压平后最短轴就是圆盘法线"。
    但实测（scripts/compare_normal_sources.py，对**压平过的**模型直接比较邻域一致度）：

        候选                      mean_dot    翻转率
        +z 轴（旋转第 3 列）       0.9569      0.0%     ← 最优
        最短轴 argmin(scales)      0.5690      4.4%
        最长轴                     0.5586      5.1%

    而且"最短轴是 z 轴"只在 **19.2%** 的高斯上成立（其余 81% 落在两个水平轴上）——
    也就是说旧实现**大部分时候在监督一个错误的水平方向**，法线一致性正则基本没起作用，
    这也解释了它为什么"加不加都一样"（docs/03 G52）。

    几何原因：压平（`scatter(-1, argmin, cap)`）把**最小**的尺度压到上限，
    所以最短轴与 +z 轴本来就是同一根；但 `argmin` 返回的是**当时的索引**，
    在索引不等于 2 的高斯上它就指向了一个水平轴 —— 两种写法的差别只在这一点。
    直接用 +z 既正确又没有这个索引歧义。

    `axis="shortest"` 保留给"确实想取最短轴"的实验场景，不作为默认。
    """
    import torch

    R = _quat_to_rotmat(quats)                       # [N,3,3]，列是三个轴
    if axis == "shortest":
        idx = torch.argmin(scales, dim=-1).view(-1, 1, 1).expand(-1, 3, 1)
        n = torch.gather(R, 2, idx).squeeze(-1)
    else:
        n = R[:, :, 2]                               # +z 轴（圆盘法线）
    return n / (n.norm(dim=-1, keepdim=True) + 1e-8)


def depth_to_normal(depth: "torch.Tensor", K: "torch.Tensor"):
    """由**深度图**求几何法线（与导出阶段 `backproject_depth_to_points` 同一套公式）。

    相机系下 z(u,v)，切向量
        T_u = [zu*x/fx + x*zu/z + z, zu*y/fy, zu]
        T_v = [zv*x/fx, zv*y/fy + y*zv/z + z, 1]
        n   = normalize(T_v × T_u)
    其中 zu/zv 为深度对像素的中心差分偏导。

    返回 (法线 [H,W,3], 有效掩码 [H,W])。深度 <= 0 或梯度退化处标记为无效。
    """
    import torch

    H, W = depth.shape
    z = depth
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    vs = torch.arange(H, device=depth.device, dtype=depth.dtype).view(-1, 1).expand(H, W)
    us = torch.arange(W, device=depth.device, dtype=depth.dtype).view(1, -1).expand(H, W)
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z

    zu = torch.zeros_like(z)
    zv = torch.zeros_like(z)
    zu[:, 1:-1] = 0.5 * (z[:, 2:] - z[:, :-2])
    zv[1:-1, :] = 0.5 * (z[2:, :] - z[:-2, :])

    eps = 1e-8
    # 视线方向（相机系：从光心指向该像素），先算好 —— 下面的退化兜底要用。
    # ⚠️ 这个变量必须**定义在使用之前**：第一版把它放在了兜底之后，
    # 单次调用会 NameError，而在循环里会**静默沿用上一轮的残留值**，
    # 表现为"兜底法线在整幅图上都是一个常数方向"（单测发现）。
    forward = torch.stack([x, y, z], dim=-1)

    # ---- 由深度图求表面法线（正确的切向量推导）----
    #
    # 参数化：P(u,v) = z(u,v) · K⁻¹[u,v,1]，记 m = (x/z, y/z, 1) = K⁻¹[u,v,1]
    #   P_u = z_u·m + z·(1/fx, 0, 0)
    #   P_v = z_v·m + z·(0, 1/fy, 0)
    #   n   = P_u × P_v
    #
    # ⚠️ 这里曾经写错过（而且**导出阶段也是同一个错误公式**）：
    # 把 P_v 的 z 分量写成了常数 1（那是 dz/dv 吗？不是，1 是 K⁻¹ 第三分量的系数）。
    # 后果：**正对相机的平面**（z_u = z_v = 0，最常见的情形）算出的法线会
    # 偏离真实方向 26.6°（实测 [0, 0.447, -0.894]，而正确值是 [0, 0, -1]）。
    # 用正确的 P_u × P_v 代入 z_u=z_v=0 可验证：结果 = -z²·fx·fy·ẑ ∝ [0,0,-1] ✓
    m = torch.stack([x / z.clamp_min(eps), y / z.clamp_min(eps), torch.ones_like(z)], dim=-1)
    pu = zu.unsqueeze(-1) * m + z.unsqueeze(-1) * torch.stack(
        [torch.full_like(z, 1.0 / fx), torch.zeros_like(z), torch.zeros_like(z)], dim=-1)
    pv = zv.unsqueeze(-1) * m + z.unsqueeze(-1) * torch.stack(
        [torch.zeros_like(z), torch.full_like(z, 1.0 / fy), torch.zeros_like(z)], dim=-1)
    n = torch.cross(pu, pv, dim=-1)
    ln = n.norm(dim=-1, keepdim=True)
    # 兜底：极端情况下（数值退化）退回视线方向
    degenerate = (ln[..., 0] <= 1e-12) & (z > 1e-6)
    if bool(degenerate.any()):
        fallback = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        n = torch.where(degenerate.unsqueeze(-1), fallback, n)
    valid = z > 1e-6
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # 统一朝向相机（视线为 -forward）：法线要与视线反向
    flip = (n * (-forward)).sum(-1, keepdim=True) < 0
    n = torch.where(flip, -n, n)
    return n, valid


def normal_consistency_loss(n_render_cam: "torch.Tensor", n_geo: "torch.Tensor",
                            weight: "torch.Tensor"):
    """法线一致性损失：`1 - |dot(渲染法线, 深度几何法线)|`，按 `weight` 加权平均。

    单独抽成函数是为了**可单测**：这个量曾经被我写成恒等式（恒为 0，等于没加正则），
    而单测能立刻抓到"两种明显不同的输入给出相同损失"。

    两个输入都应是**单位向量**、同一坐标系（相机系）、形状 [H,W,3]。
    取绝对值是因为法线的正负方向对"表面朝向"等价（两面都可渲染）。
    """
    import torch

    a = n_render_cam / n_render_cam.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b = n_geo / n_geo.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dot = (a * b).sum(-1).abs().clamp(0.0, 1.0)
    w = weight
    denom = w.sum()
    if float(denom) <= 0:
        return None
    return ((1.0 - dot) * w).sum() / denom


def depth_distortion_loss(depth: "torch.Tensor", alpha: "torch.Tensor",
                          mode: str = "tv"):
    """深度失真抑制：惩罚深度图的局部不平滑（`tv` 一阶 / `lap` 二阶），按 alpha 加权。

    2DGS 原式是 `L_d = Σ_{i,j} w_i w_j |z_i - z_j|`（需要逐像素的高斯深度列表），
    而 gsplat 的 `render_mode="D"` 只给累计量 `Σwᵢzᵢ`。这里用深度图差分做等价约束：
    同样是"让同一像素附近的高斯挤到同一深度"，只需一次额外渲染、无需改 CUDA 核。
    """
    import torch

    if mode == "lap":
        dx = depth[:, 2:] - 2 * depth[:, 1:-1] + depth[:, :-2]
        dy = depth[2:, :] - 2 * depth[1:-1, :] + depth[:-2, :]
        ax, ay = alpha[:, 1:-1], alpha[1:-1, :]
    else:
        dx = depth[:, 1:] - depth[:, :-1]
        dy = depth[1:, :] - depth[:-1, :]
        ax, ay = alpha[:, 1:], alpha[1:, :]
    if not dx.numel() and not dy.numel():
        return None
    total = (dx.abs() * ax).mean()
    if dy.numel():
        total = total + (dy.abs() * ay).mean()
    return total


def two_dgs_regularizers(cfg, means, quats, scales, opacities, shs, kk, vm,
                         width: int, height: int, alphas, sh_degree: int):
    """2DGS 的两项正则：**深度失真抑制** + **法线一致性**。

    动机（本项目实测）：3DGS 的累计深度 `Σwᵢzᵢ / Σwᵢ` 是沿射线的**加权平均**，
    一个像素的深度由几十个高斯混合而成 -> 表面位置本身是模糊的。于是导出网格时
    "深度反投影"拿到的点云在法线方向上有厚度，泊松重建出来就是碎面，
    由深度梯度算的法线也不稳（实测网格法线翻转 31~44%）。

    2DGS（SIGGRAPH'24）用两项正则把表面"钉住"：
      * **深度失真**：同一像素内高斯的深度应尽量集中。2DGS 原式是
        `L_d = Σ_{i,j} w_i w_j |z_i - z_j|`，需要逐像素的高斯深度列表，
        而 gsplat 的 `render_mode="D"` 只给累计量 `Σwᵢzᵢ`。
        这里用**深度图的差分**做等价约束（`tv`：一阶 / `lap`：二阶），
        效果同样是"让深度分段平滑/恒定"，且只需一次额外渲染、无需改 CUDA 核。
      * **法线一致性**：渲染出的高斯法线（最短轴）应与**深度图的几何法线**一致。

    返回 {"depth": Tensor|None, "normal": Tensor|None}（已经过 alpha 加权）。
    """
    import torch
    from gsplat import rasterization

    want_depth = float(getattr(cfg, "lambda_depth_distortion", 0.0) or 0.0) > 0
    want_normal = float(getattr(cfg, "lambda_normal", 0.0) or 0.0) > 0
    out: dict = {"depth": None, "normal": None}
    if not (want_depth or want_normal):
        # 两个正则都关闭时**直接返回**，不要白跑一次深度渲染（每 reg_interval 步一次，
        # 虽然不大但纯属浪费；实测单测也会因此误判"触发了额外渲染"）。
        return out

    # 1) 额外渲染一次深度。
    #
    # ⚠️ **这里以前用 `render_mode="D"`，那是个真实的自洽性缺陷（2026-10-25 修）**：
    #   * `D` 返回的是 `Σ wᵢ zᵢ`（**未除以 alpha**，gsplat 文档明确），它不是"深度"，
    #     而是"深度×不透明度"；alpha 低的地方数值被同比压低。
    #   * 而导出阶段（`gsdata/mesh_export.py`）用的是 **`RGB+ED`（期望深度 `Σwz/Σw`）**。
    #   于是训练时优化法线/失真用的深度，和导出时反投影用的深度**不是同一个量**，
    #   两者相差 1/alpha 倍 —— 这正是"逻辑没写自洽"的典型：各段都"能跑"，但语义不一致。
    #
    # 现在统一到 ED（与导出、与 2DGS/GOF 的做法一致）；`depth_mode="accum"` 仅保留做对照。
    depth_mode = str(getattr(cfg, "depth_mode", "expected") or "expected").lower()
    if depth_mode == "accum":
        colors_d, alphas_d, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=shs, viewmats=vm[None], Ks=kk[None], width=width, height=height,
            sh_degree=sh_degree, near_plane=0.01, far_plane=1e10, packed=False,
            render_mode="D",
        )
        depth = colors_d[0][..., 0]                # Σwᵢzᵢ（未归一化，仅对照用）
        alpha = alphas_d[0][..., 0].detach()
    else:
        colors_ed, alphas_d, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=shs, viewmats=vm[None], Ks=kk[None], width=width, height=height,
            sh_degree=sh_degree, near_plane=0.01, far_plane=1e10, packed=False,
            render_mode="RGB+ED",
        )
        depth = colors_ed[0][..., 3]               # 期望深度 Σwᵢzᵢ / Σwᵢ
        alpha = alphas_d[0][..., 0].detach()

    # 低 alpha 区域（背景/边缘）没有可信深度：置 0，让下游用法线的 valid 掩码排除。
    depth = torch.where(alpha > 0.1, depth, torch.zeros_like(depth))

    # 2) 深度失真抑制
    if want_depth:
        out["depth"] = depth_distortion_loss(
            depth, alpha,
            mode=str(getattr(cfg, "depth_distortion_mode", "tv") or "tv").lower())

    # 3) 法线一致性：渲染出的高斯法线 vs 深度图的几何法线
    #
    # ⚠️ 这里不能用"几何法线与视线正交"之类的代理量 —— 几何法线在
    # `depth_to_normal` 里已经统一朝向相机，那个量**恒等于 1**，损失恒为 0，
    # 看起来"加了正则"其实什么都没做（第一版就是这样，单测立刻抓到）。
    # 正确做法：把高斯法线当作颜色**再渲染一次**，得到逐像素的渲染法线，
    # 再与深度几何法线比较。这就是 2DGS / GOF 的法线一致性损失。
    if float(getattr(cfg, "lambda_normal", 0.0) or 0.0) > 0:
        n_geo, valid = depth_to_normal(depth, kk)
        n_gauss = gaussian_normals(quats, scales)          # [N,3] 世界系
        R = vm[:3, :3]
        # 渲染法线要在**相机系**下表达（法线不随平移变化，只旋转）
        n_cam = n_gauss @ R.t()                            # [N,3]
        # ⚠️ `sh_degree=None` 时 gsplat 要求 colors 是 **[N, D]`（不带中间的 SH 系数维），
        # 传 [N,1,3] 会触发 `AssertionError: torch.Size([N,1,3])`。
        # 第一版就是这样，被 try/except 吞掉 -> 法线项静默为 None（日志里看不到），
        # 表现为"加了法线正则但完全没生效"。现在按形状分支处理，并且**失败时会告警**。
        col_n = (n_cam * 0.5 + 0.5).reshape(-1, 3)
        try:
            n_rgb, a_n, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=col_n, viewmats=vm[None], Ks=kk[None],
                width=width, height=height, sh_degree=None,
                near_plane=0.01, far_plane=1e10, packed=False, render_mode="RGB",
            )
            n_render = n_rgb[0] * 2.0 - 1.0                # [H,W,3]
            w = (valid & (a_n[0][..., 0] > 0.5)).float()
            out["normal"] = normal_consistency_loss(n_render, n_geo, w)
        except Exception as exc:  # noqa: BLE001
            # 失败时**明确为 None 并记录原因**，而不是返回一个恒为 0 的假损失。
            # 调用方会在日志里看到 normal 缺失（曾因静默 except 而长期未察觉）。
            out["normal"] = None
            out["normal_error"] = f"{type(exc).__name__}: {exc}"
    return out


def effective_densify_until(iterations: int, densify_until: int,
                            reset_interval: int, auto_extend: bool = True) -> int:
    """把「不透明度重置」与「密度控制窗口」联动起来，返回**实际生效**的窗口上界。

    ## 为什么要联动（实测，2026-10-22）

    官方 3DGS / gsplat 的 `reset_every=3000` 与 `refine_stop_iter=15000` 是**一套机制**：
    重置把不透明度打回低位并裁掉一批高斯，**后续 densify 再把容量补回**。

    我们的 `densify_until=4000`。4000 之后一旦开启重置，就变成**只删不补**：

        | 配置 | 掩码内 SSIM | 高斯数 |
        |---|---|---|
        | 基线（无重置）| **0.9478** | 152,578 |
        | 重置 3000（窗口 4000，坏）| **0.9429** | **97,594（−36%）** |
        | 重置 3000（自动延到 9000）| **0.9456** | 207,384 |

    ⇒ 联动能消除"净删容量"，但重置本身在这份数据上仍不优于基线，所以默认仍关闭。

    规则：`reset_interval > 0` 且 `auto_extend` 时，
    `until = max(until, min(iterations*0.6, reset_interval*3))`；否则原样返回。

    ⚠️ 顺序很重要（单测抓到过一次）：必须先取 `min(cap, need)` 再与外层 `max`，
    否则"窗口本来就比 cap 长"时会被**缩短** —— 那等于主动减容量，与初衷相反。
    """
    if not auto_extend or reset_interval <= 0:
        return int(densify_until)
    need = int(reset_interval) * 3
    cap = int(max(iterations, 1) * 0.6)
    return int(max(int(densify_until), min(cap, need)))


def lr_at(step_idx: int, peak: float, min_ratio: float,
          warmup: int, total: int) -> float:
    """线性预热 + 余弦退火的学习率（**模块级**，便于用解析值直接单测）。

    为什么预热：Adam 早期只用很少的样本估计二阶矩，**初期步长极不稳定**；
    四元数（旋转流形）与尺度（对数域）对步长尤其敏感，前几百步的震荡会把点云
    "甩"到坏区域，之后很难回来。前 5%~10% 线性升温能显著稳定早期训练。
    为什么余弦而非指数：余弦**中段衰减慢、末段加速**，契合"先探索、后固化精细结构"。

    取值（可手算，测试就按这些端点断言）：
        warmup=W, total=T, peak=P, min_ratio=r
        step 0       -> P/W            （预热第一步，非 0）
        step W-1     -> P              （预热末步到峰值）
        step W       -> P              （预热结束仍为峰值）
        step (W+T)/2 -> (P + P·r)/2    （余弦中点）
        step T-1     -> ≈ P·r          （退火到下限）

    ⚠️ 不能用 `torch.optim.lr_scheduler.CosineAnnealingLR`：密度控制会**替换参数对象**
    并迁移优化器状态，调度器持有的引用会失效（本项目在 ExponentialLR 上已踩过，03/G31）。
    """
    min_lr = peak * float(min_ratio)
    if warmup > 0 and step_idx < warmup:
        return peak * (step_idx + 1) / warmup
    prog = (step_idx - warmup) / max(total - warmup, 1)
    prog = min(max(prog, 0.0), 1.0)
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * prog))


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


def ssim_simple(img1, img2, legacy_eps: bool = False):
    """可微 SSIM（11x11 高斯窗）。

    ⚠️ 这里的**分母写法与原始 3DGS 不同**，是刻意修正过的，不要"照官方改回去"：

    原版 3DGS（`utils/image_utils.py::_ssim`）与本项目旧版都是**单边 eps 在分母**：
        ssim = (2μ1μ2 + c1)(2σ12 + c2) / ((μ1²+μ2²+c1)(σ1²+σ2²+c2) + 1e-8)
    对**常数图**（μ=σ=0，例如纯黑背景）这个式子给出
        c1·c2 / (c1·c2 + 1e-8) = 1e-8 / 2e-8 = **0.5**（单通道，三通道平均≈0.9）
    而两张全黑图的 SSIM **必须等于 1.0**。

    原版 3DGS 不修它是因为其场景（Mip-NeRF 360 / Tanks&Temples）几乎没有大面积纯色区；
    **本项目是抠像后黑底素材，实测背景占 42%~85% 的像素是近黑**，
    于是每张图的 SSIM 被系统性拉低（实测 -0.047，占 SSIM 项的 14%），
    而且**偏差随背景占比漂移**（前景占比 12%~55%）——
    等于在损失里注入一个"与质量无关、随视角变化"的常数项，
    与 G70 修掉的 loss 尺度漂移是同一类病。

    现在改为与 skimage / pytorch_msssim 一致的写法（eps 只作数值安全网、不进分子），
    保证：相同图 → 1.0，全黑对全黑 → 1.0。
    验证：`scripts/compare_ssim.py`（对 skimage 逐点比对）与
    `scripts/probe_ssim_bias.py`（在真实训练数据上量化偏差）。
    """
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
    if legacy_eps:
        # 复现**原始 3DGS** 的单边 eps 写法（含常数图偏置），仅供受控对照使用。
        # 见本函数 docstring：它在纯黑背景上把 SSIM 拉到 ~0.9（应=1.0）。
        return (((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8)).mean()
    # 对称写法：eps 只防止 0/0，不改变"相同图 = 1"这一性质
    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / den.clamp_min(1e-12)
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


def load_colmap_data(data_dir: Path, downscale: int = 1, white_background: bool = False,
                     flip_camera: bool = False):
    """读取去畸变后的 COLMAP 数据，返回相机参数、位姿与点云。

    **相机位姿不需要任何轴翻转** —— 这一点我错了很久，最后靠读官方实现才纠正：

    官方 gsplat（examples/datasets/colmap.py）的做法是：
        w2c = 由 COLMAP 的 qvec/tvec 拼成的 [[R|t],[0|1]]     # world -> camera
        camtoworlds = np.linalg.inv(w2c)                      # 直接求逆，不做翻转
    而本模块的 `Image.cam_to_world()` 返回 `(R.T, -R.T @ t)`，
    正是 `inv([[R|t],[0|1]])` 的解析解 —— **两者完全等价**。
    所以 `gsplat` 的 `rasterization` 直接接收 `viewmats = inv(c2w)` 即可。

    早期我照抄 gsplat 示例里的 `c2w[0:3, 1:3] *= -1`，
    以及后来"修"成的 `c2w[0:3, 1] *= -1`，**都是多余的**，会让几何错位，
    表现为渲染与真值无法对齐、PSNR 卡在 13~14 dB 上不去。

    flip_camera 仅保留用于对照实验（默认 False = 不翻转，与官方一致）。
    """
    from ..gsdata.colmap_io import read_model

    model_dir = find_model_dir(data_dir)
    if not (model_dir / "cameras.bin").exists():
        raise StageError("train", f"找不到 COLMAP 稀疏模型：{model_dir}", "先跑 colmap 阶段")

    rec = read_model(model_dir)
    if not rec.images:
        raise StageError("train", "COLMAP 模型中没有已注册图像", "检查 colmap 阶段的日志与 report.json")

    cam_by_id = rec.cameras
    images = sorted(rec.images, key=lambda im: im.name)

    # 自动裁剪的偏移：`gsdata/auto_crop.py` 把图与掩码都裁过之后会写 `crop.json`，
    # 这里把**内参主点**相应平移（焦距不变）。裁剪对相机几何的唯一影响就是主点。
    # 若没有 crop.json 则 (0,0)，行为与之前完全一致。
    try:
        from ..gsdata.auto_crop import read_crop_shift

        _crop = read_crop_shift(data_dir) or (0.0, 0.0)
    except Exception:  # noqa: BLE001
        _crop = (0.0, 0.0)
    crop_dx, crop_dy = float(_crop[0]), float(_crop[1])

    c2ws, ks, names, widths, heights = [], [], [], [], []
    for im in images:
        cam = cam_by_id[im.camera_id]
        k = cam.intrinsics()
        if crop_dx or crop_dy:
            # 裁剪是"左上角移到 (dx,dy)" ⇒ 同一像素的新坐标比旧坐标小 dx,dy
            k[0, 2] -= crop_dx
            k[1, 2] -= crop_dy
        if downscale > 1:
            k[0, 0] /= downscale
            k[1, 1] /= downscale
            k[0, 2] = (k[0, 2] + 0.5) / downscale - 0.5
            k[1, 2] = (k[1, 2] + 0.5) / downscale - 0.5
        r, t = im.cam_to_world()
        c2w = np.eye(4)
        c2w[:3, :3] = r
        c2w[:3, 3] = t
        # 与官方一致：c2w 直接由 COLMAP 的 w2c 求逆得到，**不做轴翻转**。
        # 详见本函数 docstring。flip_camera 只用于对照实验。
        if flip_camera:
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


def _init_scales_nn(means, k: int = 3):
    """按官方 3DGS 的做法算初始尺度：到 k 个最近邻的平均平方距离，再开方。

    官方（inria `gaussian_model.create_from_pcd`）：
        dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)   # 到最近邻的平方距离均值
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    这里用分块 cdist 实现，避免一次性构造 N×N 矩阵（大场景会吃满显存）。
    """
    import torch

    pts = means.detach().float()
    n = pts.shape[0]
    if n <= 1:
        return torch.full((n, 1), 0.01, device=pts.device)
    kk = min(k, n - 1)
    out = torch.empty(n, device=pts.device)
    chunk = max(1, min(n, 4096))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d = torch.cdist(pts[s:e], pts)                      # [c, n]
        # 排除自身（对角线），取最小的 kk 个
        d[torch.arange(e - s, device=pts.device), torch.arange(s, e, device=pts.device)] = float("inf")
        nn = d.topk(kk, dim=1, largest=False).values        # [c, kk]
        out[s:e] = nn.pow(2).mean(dim=1)
    dist = torch.sqrt(torch.clamp_min(out, 1e-8))
    return dist.clamp_min(1e-4).unsqueeze(-1)


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

    data = load_colmap_data(cfg.data_dir, downscale=1, white_background=cfg.white_background,
                            flip_camera=cfg.flip_camera)
    names = data["names"]
    ctx.info(f"已注册视图 {len(names)} 个，稀疏点 {len(data['points'])}")

    # 分辨率：按长边缩放到 cfg.resolution
    #
    # ⚠️ `resolution` 的语义是"长边**缩放**到该值"，**放大也照做** ——
    # 而默认 `resolution=1600` 在 720px 的素材上会把图**放大 2.2 倍**：
    # 既不增加任何信息，又让训练更慢、显存更紧张。
    # 实测本项目 IMG2 素材原图 720×720（去畸变后 721×721），我却一直用 900 ~ 1600 训练。
    # 这里以**数据集原生长边**为上限：只缩小、不放大。
    native = max(int(data["widths"][0]), int(data["heights"][0]))
    max_side = cfg.resolution if cfg.resolution > 0 else None
    if max_side and native and max_side > native:
        ctx.info(f"分辨率上限下调：请求 {max_side} > 原生长边 {native}，"
                 f"按原生 {native} 训练（避免无意义的放大）")
        max_side = native
    imgs, masks = _load_images(cfg.data_dir, names, max_side)
    h, w = imgs[0].shape[:2]
    ctx.info(f"训练分辨率：{w}x{h}（原生长边 {native}），共 {len(imgs)} 张")

    # ---- 训练用掩码腐蚀（修"轮廓白边"，2026-10-24）----
    #
    # 实测本数据集的掩码**整体偏大**：掩码内靠边 4px 环的真值亮度 **0.89**，
    # 而掩码内部只有 0.43~0.51、掩码外 0.91 ⇒ 那一圈其实是**亮背景**（墙/天空）。
    # 训练把它当物体，模型就学出**轮廓高光**（用户反复说的"白边/发光"）。
    #
    # 这里只腐蚀**损失用**的掩码；`evaluate()` 仍用原始掩码，保证指标可比。
    masks_loss = masks
    _erode = int(getattr(cfg, "mask_erode_px", 0) or 0)
    if _erode > 0:
        import torch.nn.functional as _F

        masks_loss = []
        for _m in masks:
            k = 2 * _erode + 1
            e = _F.max_pool2d((~_m).float()[None, None], k, stride=1, padding=_erode)[0, 0]
            masks_loss.append(~(e > 0.5))
        kept = float(sum(m.sum() for m in masks_loss)) / max(float(sum(m.sum() for m in masks)), 1)
        ctx.info(f"损失用掩码腐蚀 {_erode}px：保留 {kept*100:.1f}% 前景像素"
                 f"（评估仍用原掩码）")

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

    # 划分训练/评估。
    #
    # 踩坑：原来是 `if len(imgs) >= 10 else 0`，于是只有 7 个视图时评估集为空，
    # evaluate() 直接返回 0.0，日志里永远是 "PSNR=0.00 dB"，看起来像渲染坏了。
    # 正确做法：**视图太少时不留评估集**（留出 10% 会明显削弱训练），
    # 改为在训练视图上算 PSNR，并在指标里标注 "train_views"，避免误读。
    perm = torch.randperm(len(imgs), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    can_holdout = cfg.eval and len(imgs) >= cfg.min_imgs_for_holdout
    n_eval = max(1, len(imgs) // 10) if can_holdout else 0
    eval_idx = set(perm[:n_eval])
    train_idx = [i for i in range(len(imgs)) if i not in eval_idx]
    # 评估用的视图集合：有留出就用留出，否则退回训练视图（并标记）
    psnr_idx = sorted(eval_idx) if eval_idx else list(range(len(imgs)))
    psnr_on_train = not eval_idx
    # ⚠️ 度量混叠（本项目一个很隐蔽的坑，见 docs/03 G32）：
    # 训练视图是按 `train_idx[step % len(train_idx)]` **循环**取的，
    # 而评估间隔 eval_interval 往往与循环长度 len(train_idx) 不互质
    # （本机实测 1000 % 110 = 10），于是
    #     6000 % 110 == 12000 % 110 == 60
    # —— 两次评估恰好落在**循环里的同一相位**，看到的是同一张"难图"。
    # 表现是 PSNR 曲线在固定步数反复骤降（6000/12000/...），
    # 让人误判"训练不稳定"。实测内置密度控制与官方 DefaultStrategy **都**在同样步数降，
    # 说明问题在**度量**而不是训练。
    # 对策：日志用的高频评估**固定取前若干张**（确定性），
    # 完整留出集只在最后一次评估时用（作为最终可信指标）。
    quick_n = int(cfg.eval_quick_views)
    if quick_n > 0 and len(psnr_idx) > quick_n:
        psnr_idx_log = psnr_idx[:quick_n]
    else:
        psnr_idx_log = psnr_idx
    if psnr_idx_log != psnr_idx:
        ctx.info(f"评估：日志用固定 {len(psnr_idx_log)} 个视图（避免与训练循环混叠），"
                 f"最后一次用全部 {len(psnr_idx)} 个留出视图")
    if psnr_on_train:
        ctx.warn(f"仅 {len(imgs)} 个视图（< {cfg.min_imgs_for_holdout}），不留评估集；"
                 f"PSNR 将在 {len(psnr_idx)} 个训练视图上计算（仅供趋势参考，不是泛化指标）")

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

    # 尺度初始化：官方 3DGS/gsplat 用 **最近邻距离** 的平均平方距离再开方
    #   dist2 = mean(||p - knn(p)||^2)      # k=3
    #   scales = log(sqrt(dist2))
    # 踩坑：原来用的是"到点云质心的距离 × 0.5"。
    # 对一块平面场景，质心到每个点的距离约等于平面的半径（本例 ~1.0），
    # 于是初始高斯有半个场景那么大 -> 整幅图被少数几个巨大高斯糊住，
    # 渲染出来就是一片带颜色的雾。改成最近邻距离后才与官方一致
    # （可以用 --set train.init_scale_mode=knn|centroid 对照）。
    init_dist = _init_scales_nn(means.detach(), k=3)
    scales = torch.nn.Parameter(torch.log(init_dist).repeat(1, 3).to(device))
    ctx.info(f"初始尺度（最近邻均值）= {float(init_dist.mean()):.5f}，"
             f"中位数 {float(init_dist.median()):.5f}")
    quats = torch.nn.Parameter(torch.zeros(n, 4, device=device))
    quats.data[:, 0] = 1.0
    # 不透明度初值（官方也是 0.1）。实测：训练完中位只有 0.07、
    # op<0.1 占 67% ⇒ 模型收敛到“半透明层叠”的解，新视角就透。
    # 初值抬高会把优化推向“少一点但更实”的解，这里开放成配置。
    _op_init = float(getattr(cfg, "opacity_init", 0.1) or 0.1)
    _op_init = min(max(_op_init, 0.01), 0.99)
    opac = torch.nn.Parameter(torch.logit(torch.full((n, 1), _op_init, device=device)))
    sh_dim = (cfg.sh_degree + 1) ** 2
    shs = torch.nn.Parameter(torch.zeros(n, sh_dim, 3, device=device))
    rgb0 = torch.from_numpy(cols).float().to(device) / 255.0
    shs.data[:, 0, :] = (rgb0 - 0.5) / 0.28209479177387814  # SH DC 反解

    # scene_scale：**一次性**从相机位置估出（相机中心到其均值的平均距离），
    # 与官方 DefaultStrategy.initialize_state 一致。
    # 关键：它必须是常量。早期我在分裂判据里用"当前尺度中位数"当参照，
    # 而每次分裂又把尺度除以 1.6 -> 阈值自增强 -> 尺度与位置双双发散
    # （实测 log尺度到 739、坐标范围到 215）。
    _cam_centers = c2ws[:, :3, 3]
    scene_scale = float(torch.linalg.norm(_cam_centers - _cam_centers.mean(0), dim=1).mean()) + 1e-6
    ctx.info(f"scene_scale = {scene_scale:.4f}（相机中心平均距离）；"
             f"克隆上限 {cfg.grow_scale3d * scene_scale:.5f}，裁剪阈值 {cfg.prune_scale3d * scene_scale:.5f}")

    # 参数用字典管理，和官方 gsplat 一致：密度控制会**替换**参数对象并迁移优化器状态，
    # 所以不能像早期版本那样把参数绑死在闭包里的局部变量上。
    params: dict[str, torch.nn.Parameter] = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opac,
        "shs": shs,
    }
    # 各参数的学习率。
    # ⚠️ opacities 从官方默认的 5e-2 **降到 2.5e-2**：官方 3DGS 的 0.05 配合"每 3000 步重置
    # 不透明度"才成立；我们**默认关闭重置**（实测重置会污染末次检查点），
    # 此时 0.05 会让不透明度一路被压向 0，退化成"一大团极低不透明度的薄雾"：
    # 轮廓指标（PSNR / IoU）仍然正常，但深度完全模糊 -> 深度反投影失效 -> 网格成碎片。
    # 实测（img2-04，242 帧 / 35000 迭代）：opacity 中位数 0.0165 -> 0.0138，
    # 最终 >0.02 的高斯只剩 0.2%。
    # Faster-GS 与 3DGS-MCMC 都用 0.025（见其 Trainer.py 的 LEARNING_RATE_OPACITIES），
    # 这里对齐。
    # 与官方对齐：官方 means 学习率 = 1.6e-4 * scene_scale。
    # 我们长期用固定 1.6e-4：场景尺度 ~5 时位置每步更新量偏小 5 倍，
    # 几何难以收敛，只能靠放大/拉长高斯去覆盖像素 ⇒ 针状高斯。
    _lr_means = 1.6e-4 * (scene_scale if bool(getattr(cfg, "lr_means_scene_scale", False)) else 1.0)
    _lr_shs = float(getattr(cfg, "lr_shs", 2.5e-3) or 2.5e-3)
    _LR = {"means": _lr_means, "scales": 5e-3,
           # 四元数学习率：原为 1e-3。按用户方案降到 5e-4 ——
           # Adam 的每步更新量约等于 lr（因为它按二阶矩归一化），而旋转是**流形**参数，
           # 1e-3 偏大：椭球会在空间里反复乱转，连带让法线/朝向类统计量一直抖，
           # 这也是"法线一致性正则不收敛"的一个可疑来源（见 03/G65）。
           "quats": float(getattr(cfg, "lr_quats", 5e-4) or 5e-4),
           "opacities": float(getattr(cfg, "lr_opacities", 0.025) or 0.025),
           "shs": _lr_shs}
    _gamma = 0.01 ** (1 / max(cfg.iterations, 1))   # 旧策略（纯指数）保留，供 A/B 复现

    # ---- 消除「容量冻结 vs 重置」冲突：重置开启时自动延长密度窗口 ----
    #
    # 官方 `reset_every=3000` 与 `refine_stop_iter=15000` 是**联动**的：
    # 重置把不透明度打回低位并裁掉一批高斯，后续 densify 再把容量补回。
    # 我们 `densify_until=4000` 时，4000 之后的每次重置都是**净删容量**
    # （实测 152,578 → 97,594，SSIM 0.9478 → 0.9429）。
    # 用户要求"有冲突就改"：这里把两个参数联动起来。逻辑抽成纯函数便于单测。
    _want = effective_densify_until(
        int(cfg.iterations), int(cfg.densify_until),
        int(getattr(cfg, "opacity_reset_interval", 0) or 0),
        bool(getattr(cfg, "auto_extend_densify_when_reset", True)))
    if _want > int(cfg.densify_until):
        ctx.warn(
            f"重置间隔 {cfg.opacity_reset_interval} 与密度窗口 {cfg.densify_until} "
            f"不匹配（重置后无法补回容量）⇒ 自动把窗口延长到 **{_want}**")
        cfg.densify_until = _want

    # ---- 学习率调度：线性预热 + 余弦退火（用户方案第 1 项）----
    # 曲线本体是模块级的 `lr_at()`（便于用解析端点单测）；
    # 这里只把配置读成常量，避免闭包内重复读 cfg。
    _WARM = int(getattr(cfg, "warmup_iters", 0) or 0)
    _TOTAL = max(int(cfg.iterations), 1)
    _MINR = float(getattr(cfg, "lr_min_ratio", 0.1) or 0.1)

    def _lr_at(step_idx: int, peak: float) -> float:
        return lr_at(step_idx, peak, _MINR, _WARM, _TOTAL)

    _SCHED = str(getattr(cfg, "lr_schedule", "cosine") or "cosine").lower()

    def make_optimizers():
        """为**当前**的参数对象建优化器（只在开头调用一次）。

        密度控制时不重建优化器，而是迁移状态 + 原地重建同 gamma 的调度器
        （见 _update_param_with_optimizer 与 official gsplat 的 ops.py）。
        """
        opts = {k: torch.optim.Adam([v], lr=_LR[k], eps=1e-15) for k, v in params.items()}
        return opts

    def schedule_lrs(step_idx: int) -> None:
        """按 `lr_schedule` 设置各参数学习率。

        * `cosine`（默认）—— 线性预热 + 余弦退火到峰值的 `lr_min_ratio`
        * `exp`           —— 旧的纯指数衰减，保留以便复现历史结果与做 A/B
        """
        if _SCHED == "exp":
            factor = _gamma ** step_idx
            for k, opt in optimizers.items():
                for group in opt.param_groups:
                    group["lr"] = _LR[k] * factor
            return
        for k, opt in optimizers.items():
            lr = _lr_at(step_idx, _LR[k])
            for group in opt.param_groups:
                group["lr"] = lr
    optimizers = make_optimizers()
    sched = None  # 用 schedule_lrs 手动等价实现，避免替换参数后调度器失效
    # 利用率统计（方案 B）：累积 |∂L/∂opacity|。关闭时保持 None，不产生任何开销。
    # 形状随高斯数变化（密度控制会增删高斯），所以每次用前先对齐长度。
    util_acc = (torch.zeros(int(opac.shape[0]), device=device)
                if int(getattr(cfg, "utilization_prune_interval", 0) or 0) > 0 else None)
    # ⚠️ 用**可变容器**承载需要在嵌套函数里改写的状态。
    # 直接写 `util_acc = ...` 会让它变成 `densify_and_prune` 的**局部变量**
    # （实测 UnboundLocalError），因为嵌套函数里的赋值总是创建局部名。
    util_state: dict = {"acc": util_acc}
    # 方案 A 的状态：衰减前的快照（步骤 -> 不透明度张量），用于算"恢复率"
    recov_state: dict = {"snap": None, "snap_step": -1}

    # ---- 可选：直接复用官方 `gsplat.DefaultStrategy` ----
    # 这是"移植官方机制"的正确形态：整套复用（含 ops.py 的
    # duplicate/split/remove/reset_opa 与 grad2d/count 统计量），而不是零散抄公式。
    # 我试过零散移植三次都失败（见 docs/03 G31），根因就是机制的整体性。
    gs_strategy = None
    gs_state: dict = {}
    if str(cfg.densify_strategy).lower() == "official":
        try:
            from gsplat import DefaultStrategy

            gs_strategy = DefaultStrategy(
                prune_opa=cfg.prune_opacity,
                grow_grad2d=cfg.densify_grad_threshold,
                grow_scale3d=cfg.grow_scale3d,
                prune_scale3d=cfg.prune_scale3d,
                # ⚠️ **必须显式传 cap_max**（2026-10-25 实测）：不传时 gsplat 用自己的默认上限，
                # 而我们的 `train.cap_max`（默认 60 万）根本没生效 ——
                # 质量预设实测在 step 10000 就涨到 **251 万高斯**、显存 7.5GB、ETA 124 分钟，
                # 是一次典型的"参数没接上导致致密化失控"。
                cap_max=int(cfg.cap_max),
                refine_start_iter=cfg.densify_from,
                refine_stop_iter=cfg.densify_until,
                reset_every=(cfg.opacity_reset_interval or 3000),
                refine_every=cfg.densify_interval,
                pause_refine_after_reset=cfg.pause_refine_after_reset,
                revised_opacity=bool(cfg.revised_opacity),
                # AbsGS 的绝对梯度致密化：官方 strategy 会把 absgrad 透传给栅格化，
                # 于是 densify 判据用 `means2d.absgrad`（绝对值和）而不是普通梯度。
                absgrad=bool(cfg.absgrad),
                key_for_gradient="means2d",
            )
            gs_strategy.check_sanity(params, optimizers)
            gs_state = gs_strategy.initialize_state(scene_scale=scene_scale)
            ctx.info(f"密度控制：**官方 DefaultStrategy**（grow_grad2d={cfg.densify_grad_threshold}，"
                     f"refine {cfg.densify_from}~{cfg.densify_until} 每 {cfg.densify_interval} 步，"
                     f"reset_every={cfg.opacity_reset_interval or 3000}，"
                     f"pause_refine_after_reset={cfg.pause_refine_after_reset}）")
        except Exception as exc:  # noqa: BLE001
            gs_strategy = None
            ctx.warn(f"官方 DefaultStrategy 初始化失败，退回内置密度控制："
                     f"{type(exc).__name__}: {exc}")

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
        s = torch.exp(scales)
        if not cfg.flatten_2dgs:
            return s
        # 2DGS 参数化：只限制**最短轴**，让高斯退化成"圆盘"（圆盘半径仍自由学习）。
        #   1) argmin 取最短轴下标 —— 与高斯的旋转一致，故法线语义稳定
        #   2) 把该轴厚度压到上限
        #   3) scatter 放回原位置
        axis = torch.argmin(s, dim=-1, keepdim=True)          # [N,1]
        thin = torch.gather(s, -1, axis)                      # [N,1]
        cap = float(np.exp(cfg.flatten_log_scale))
        thin2 = torch.minimum(thin, torch.full_like(thin, cap))
        return s.scatter(-1, axis, thin2)

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
            # 官方语义：判据是 **平均每次可见的梯度**（grad2d / count），
            # 量纲已归一化，所以 2e-4 这类阈值在不同分辨率/场景下可迁移。
            # 调用方传入的就是这个平均量（见 train() 里的 grad2d/count 累加）。
            # densify_quantile>0 时才启用分位自适应（实验用）。
            # 阈值按**判据来源**取：grad2d 与 grad3d 不是同一量纲，不能混用。
            # 实测两者 p50 相差 ~45 倍（1.6e-4 vs 3.5e-6），混用会让加密量完全失控。
            _crit = str(getattr(cfg, "densify_criterion", "grad2d") or "grad2d").lower()
            _base_thr = (float(getattr(cfg, "densify_grad2d_threshold", 2e-4))
                         if _crit == "grad2d" else cfg.densify_grad_threshold)
            if cfg.densify_quantile > 0:
                thr = _quantile(grad_norm, 1.0 - cfg.densify_quantile)
                thr = min(max(thr, _base_thr), cfg.densify_grad_max)
            else:
                thr = _base_thr
            is_grad_high = grad_norm > thr

            # 官方语义：按**绝对尺度阈值**（0.01 * scene_scale）区分克隆/分裂。
            # 之前用"当前尺度中位数"作参照是错的：每次分裂把尺度除以 1.6，
            # 中位数跟着下降 -> 阈值自增强 -> 无界分裂 + 尺度爆炸（实测 log尺度到 739）。
            scales_max = torch.exp(params["scales"].data).max(dim=-1).values
            is_small = scales_max <= cfg.grow_scale3d * scene_scale
            is_dup = is_grad_high & is_small
            is_split = is_grad_high & (~is_small)

            n_before = params["means"].shape[0]
            if cfg.cap_max and n_before >= cfg.cap_max:
                is_dup = torch.zeros_like(is_dup)
                is_split = torch.zeros_like(is_split)

            n_dup, n_split = _grow(params, optimizers, is_dup, is_split,
                                   revised_opacity=cfg.revised_opacity)
            refresh_params()

            # 裁剪规则一：不透明度过低（官方 prune_opa）
            is_prune = torch.sigmoid(params["opacities"].data).squeeze(-1) < cfg.prune_opacity
            # 裁剪规则二：尺度过大（官方 prune_scale3d）——**这条是防发散的关键，早期漏实现**。
            # 官方用 `step > reset_every` 作门槛；若关闭了重置，退回用 densify_from
            # （即"训练进入中段后"才启用尺度裁剪）。
            _reset_iv = (cfg.opacity_reset_interval if cfg.opacity_reset_interval > 0
                         else cfg.densify_from)
            use_scale_prune = (not cfg.prune_scale_after_reset) or step > _reset_iv
            if use_scale_prune:
                scales_now = torch.exp(params["scales"].data).max(dim=-1).values
                is_prune = is_prune | (scales_now > cfg.prune_scale3d * scene_scale)
            n_prune = int(is_prune.sum())
            if n_prune:
                _apply_keep(params, optimizers, ~is_prune)
                refresh_params()

            # ⚠️ 形状同步：密度控制会增删高斯，而 `util_acc` 是在**反向**时按当时的高斯数
            # 累加的 ⇒ 到这一步长度可能已经不匹配。
            # 实测：`step 800` 同时是致密化步（500+100k）与利用率剪枝步（400 的倍数），
            # 打印出"高斯 38541，统计量长度 36889" ⇒ 剪枝被长度守护条件跳过，
            # 表现为"参数生效了但功能从不执行"。
            # ⚠️ 形状同步**移到反向之前**（见循环里 `util_state["acc"]` 的同步）。
            # 曾放在这里：致密化窗口内每 100 步都增删高斯 ⇒ 累加器每步被清零
            # ⇒ 阈值恒为 0、剪枝永不生效（实测"参数生效了但功能从不执行"）。
            # 现在只在反向之前按梯度长度重建一次，之后正常累积。

            smax = torch.exp(params["scales"].data).max(dim=-1).values
            ctx.info(f"step {step} 密度：阈值={thr:.2e} 克隆 {n_dup} 分裂 {n_split} "
                     f"裁剪 {n_prune} -> {params['means'].shape[0]} 个高斯"
                     f"（尺度 max={float(smax.max()):.4f} / scene_scale={scene_scale:.3f}"
                     f"{'，已启用尺度裁剪' if use_scale_prune else ''}）")
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
    # 感知损失状态：函数缓存 + "不可用就永久关闭"标志
    # （权重缺失/前向失败是**确定性**的，不需要每步重试）
    _lpips_fn = None
    _lpips_disabled = False
    # 最佳检查点跟踪（导出阶段优先用它，见 stages/export.py:_find_ply）
    # best_psnr 存的是"选择指标"的当前最优值（按 best_metric 可能是 -LPIPS），
    # best_metrics_now 存该检查点的**完整**指标，供 metrics.json 记录（多指标评估）
    best_psnr = float("-inf")
    best_step = 0
    best_ply: Path | None = None
    best_metrics_now: dict = {}

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

        # 校验并归一化光栅化模式（非法值回退 classic 并告警，不静默）
        _rm = str(getattr(cfg, 'rasterize_mode', 'classic') or 'classic').lower()
        if _rm not in ('classic', 'antialiased'):
            ctx.warn(f"未知的 rasterize_mode={_rm!r}，回退 classic")
            _rm = 'classic'
        _raster_mode = _rm
        colors, alphas, meta = rasterization(
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
            # 社区做法：antialiased 给 2D 投影加低通滤波（Mip-Splatting 思路）。
            # 非法值在下面统一校验后回退 classic。
            rasterize_mode=_raster_mode,
            # ⚠️ AbsGS 的绝对梯度**必须同时在这里传 absgrad=True**，否则
            # `meta["means2d"].absgrad` 是 None，官方 strategy 取 `.clone()` 会失败。
            # 官方文档明确写了这一点（"the rasterization function should be called
            # with absgrad=True as well so that the absolute gradients are computed"），
            # 我第一次只给 strategy 传了、漏了这里。
            absgrad=bool(cfg.absgrad),
        )
        # 官方的密度控制判据用的是 **2D 屏幕空间**的 means2d 梯度，不是 3D 的 means.grad。
        # 两者量纲完全不同：`grow_grad2d=2e-4` 只在归一化后的屏幕空间里才有意义。
        # 所以必须 retain_grad，让 backward 之后还能读到 means2d.grad。
        if isinstance(meta, dict) and meta.get("means2d") is not None:
            meta["means2d"].retain_grad()
        render = colors[0]
        if cfg.white_background:
            render = render + (1 - alphas[0])
        else:
            render = render + background * (1 - alphas[0])

        # ---- 掩码处理：**先改 GT，再算损失**（本轮的重要算法修正）----
        #
        # 原实现是 `l1 = l1[m].mean()`（只在掩码内取均值），有两个真实后果：
        #   1) **loss 量级随前景占比变化**：实测各视角前景占比 12%~55%，
        #      同样的像素误差下 loss 量级相差 4.6 倍，前景小的视角被系统性欠训练；
        #   2) **更严重：掩码外完全没有监督** ⇒ 模型可以在背景区域任意生成高斯
        #      （渲染成什么都不受惩罚），于是渲染内容远超真实范围。
        #      实测证据：`preview_*.png` 里渲染/真值的内容高度比达 **2.31 倍**
        #      （渲染把牌坊画满整幅、真值只有顶部）——
        #      这正是"该空洞的地方出现破碎网格"的来源。
        #
        # 正确做法（3DGS / gsplat 官方示例同源）：把 GT 在**掩码外合成为已知背景色**，
        # 损失仍在**全图**上计算。于是：
        #   * 前景：照常监督真实像素；
        #   * 背景：监督"必须渲染成背景色"⇒ 抑制背景里的游离高斯；
        #   * 每个视角的 loss 尺度一致（不再随前景占比漂移）。
        #
        # ⚠️⚠️ **这里曾经有一个致命顺序错误**（本轮查出）：
        #
        #     l1 = (render - gt).abs()          # ← 用**原始** gt 算好了
        #     if masks and ...:
        #         gt = torch.where(...)          # ← 只替换了 gt
        #     l1 = l1.mean()                     # ← 用的还是**旧** l1
        #
        # 于是 `composite` 对 **L1 完全没有生效**（只有 SSIM 用了被替换的 gt，
        # 而 ssim_weight 通常只有 0.2）⇒ 掩码形同虚设，背景照样被拟合。
        # 实测证据（`scripts/diag_alpha_background.py`）：
        #
        #     掩码外渲染亮度 **0.904**（应为黑）、alpha 0.998
        #     `mask_loss_mode` 改成 inside 后亮权重 61.2% → 59.6%（几乎无变化）
        #
        # 现在：**先**把 gt 在掩码外合成背景色，**再**算 l1。
        _mask = None
        if masks and step < int(cfg.iterations * cfg.mask_loss_until):
            # 损失用腐蚀后的掩码（mask_erode_px>0 时），避免把掩码外沿的亮背景
            # 学成轮廓高光；评估仍用 masks（见 evaluate）。
            _mask = masks_loss[idx]
            if max_side and _mask.shape[0] != gt.shape[0]:
                _mask = torch.nn.functional.interpolate(
                    _mask[None, None].float(), size=gt.shape[:2], mode="nearest"
                )[0, 0] > 0.5
            # ⚠️ 插值分支产生的张量在 **CPU** 上（`m` 原本是 CPU 上的 bool 张量），
            # 而 `gt` 在 CUDA 上。旧代码只在 else 分支做了 `.to(device)`，
            # 于是"掩码尺寸与图像不一致"时会崩：
            #     RuntimeError: Expected all tensors to be on the same device,
            #                   but found at least two devices, cuda:0 and cpu!
            # 这个分支只在**图像被缩放**时触发（例如 576x1280 的素材去畸变后变
            # 577x1309），所以小数据集一直没暴露它。这里统一搬设备。
            _mask = _mask.to(device)
            _mode = str(cfg.mask_loss_mode).lower()
            if _mode != "inside" and _mask.any() and not bool(_mask.all()):
                # 默认：掩码外合成背景色 —— **必须在算 l1 之前**
                gt = torch.where(_mask[..., None], gt,
                                 background.to(device=gt.device, dtype=gt.dtype))

        l1 = (render - gt).abs()
        if (_mask is not None and str(cfg.mask_loss_mode).lower() == "inside"
                and _mask.any()):
            # 旧行为（仅用于对照/复现）：只在掩码内取均值
            l1 = l1[_mask] if l1[_mask].numel() else l1
        l1 = l1.mean()

        ssim_val = ssim_simple(render.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None],
                               legacy_eps=bool(getattr(cfg, "ssim_legacy_eps", False)))
        loss = (1.0 - cfg.ssim_weight) * l1 + cfg.ssim_weight * (1.0 - ssim_val)

        # ---- 视线级不透明度下限（本项目创新项，2026-10-25）----
        #
        # ## 与"逐高斯不透明度正则"的根本区别（数学）
        #
        # 逐高斯方案（含我们的 `opacity_crisp_reg` / `opacity_floor`）把 α 当作被约束对象，
        # 但**观感上的"透"是逐像素的累计量**：一条视线穿过 N 个高斯后
        #
        #     T(u) = 1 - Π_j (1 - α_j)          # 即 gsplat 返回的 `render_alphas`
        #
        # 标准 3DGS 只用颜色损失间接约束 T：只要 Σ α_j c_j 合成出正确颜色，T 可以任意小
        # —— 这就是"颜色对得上、换视角却透"的根源（实测 op 中位 0.012、100% 高斯 <0.1）。
        #
        # 这里**直接约束累计透射率**：
        #
        #     L_ray = (1/|M|) Σ_{u∈M} relu(τ - T(u))
        #
        # 其梯度（对第 j 个高斯的 logit o_j，α_j = σ(o_j)）：
        #
        #     ∂L_ray/∂o_j = -α_j(1-α_j) · Π_{k≠j}(1-α_k) · [T < τ]
        #                 = -α_j(1-α_j) · (1-T)/(1-α_j) · [T < τ]
        #                 = -α_j · (1-T) · [T < τ]
        #
        # 关键差异：
        #   * **逐高斯下限** `relu(f - α_j)` 的梯度是 `-α_j(1-α_j)`，在 α→0 时 **趋近 0** ——
        #     而 α≈0.01 恰恰是要修的状态 ⇒ 梯度条件极差（实测把权重从 0.02 提到 0.05，分布**纹丝不动**）；
        #   * **视线级** 的梯度是 `-α_j(1-T)`：`(1-T)` 是"当前这条视线还缺多少不透明度"，
        #     T 越小梯度越大 ⇒ **正是缺得越多、推得越狠**，条件数好得多。
        #
        # 另一个好处：它天然只约束**表面像素**（掩码内），不会去推背景里的游离高斯
        # （那些像素的 T 本来就该接近 0，被掩码排除）。
        _ray_w = float(getattr(cfg, "ray_opacity_reg", 0.0) or 0.0)
        if _ray_w > 0:
            _ray_wu = float(getattr(cfg, "ray_opacity_warmup", 0.0) or 0.0) * float(cfg.iterations)
            if step >= _ray_wu:
                _tau = float(getattr(cfg, "ray_opacity_floor", 0.9) or 0.9)
                _T = alphas[0].reshape(-1)                   # [H*W] 累计不透明度
                if _mask is not None and bool(_mask.any()):
                    _wpx = _mask.reshape(-1).to(_T.dtype)
                else:
                    _wpx = torch.ones_like(_T)
                _viol = torch.relu(_tau - _T)
                loss = loss + _ray_w * (_viol * _wpx).sum() / (_wpx.sum().clamp_min(1.0))

        # ---- SH 正则（治"斜视彩虹"，2026-10-24）----
        #
        # 实测：任意轨道视角下出现青/洋红/绿斑块，而 `sh_degree=0`（仅 DC）**完全消失**
        # ⇒ 高阶 SH 在未见视角过拟合。两种抑制手段：
        #   * `sh_rest_reg`：对 `f_rest` 加 L2，压住幅值（温和，推荐先试 1e-3~1e-2）；
        #   * `sh_rest_max`：逐系数硬上限（更直接，但可能让颜色变"平"）。
        # 只作用在 **f_rest**（DC 决定基色，不能压）。
        if float(getattr(cfg, "sh_rest_reg", 0.0) or 0.0) > 0 and shs.shape[1] > 1:
            _rest = shs[:, 1:, :]
            loss = loss + float(cfg.sh_rest_reg) * (_rest ** 2).mean()

        # ---- 不透明度"清晰度"正则（2026-10-25）----
        #
        # 实测：不透明度中位 0.066、op<0.1 占 67% ⇒ 表面靠**大量半透明层**叠出来，
        # 训练视角颜色对得上（29.9 dB），但换视角就"透"、屋顶显薄。
        # 而**离线剪掉**淡高斯会掉 7.5 dB（它们在承载画质）⇒ 只能在训练时把 op 推向 0/1。
        # `op*(1-op)` 在 0 和 1 处最小、在 0.5 处最大，正好惩罚"半透明堆叠"。
        _crisp_w = float(getattr(cfg, "opacity_crisp_reg", 0.0) or 0.0)
        if _crisp_w > 0:
            # 退火（opacity_crisp_warmup）：前 N% 迭代**不施加**，
            # 先让 L1/SSIM 把颜色和结构做对，后段再把不透明度推向 0/1，
            # 避免一开始就把高斯压成硬块（上一轮全程施加就出现亮斑）。
            _wu = float(getattr(cfg, "opacity_crisp_warmup", 0.0) or 0.0) * float(cfg.iterations)
            if step >= _wu:
                _op_now = current_opacities().reshape(-1)
                loss = loss + _crisp_w * (_op_now * (1.0 - _op_now)).mean()

        # ---- 不透明度**下限**正则（2026-10-25）----
        #
        # 与上面的 `op*(1-op)` 互补，且更对症：
        #   * `op*(1-op)` 在 op→0 时**也变小** ⇒ 优化器可以"把高斯推到更淡"来满足它；
        #   * 实测 img5-v1 **85% 的高斯 op<0.1**（中位 0.031）⇒ 需要的正是"别再淡下去"。
        # 所以这里用单边惩罚 `relu(floor - op)`：只有低于下限才罚，且只能靠**抬高**来解除。
        _floor = float(getattr(cfg, "opacity_floor", 0.0) or 0.0)
        _floor_w = float(getattr(cfg, "opacity_floor_reg", 0.0) or 0.0)
        if _floor > 0 and _floor_w > 0:
            _fwu = float(getattr(cfg, "opacity_floor_warmup", 0.0) or 0.0) * float(cfg.iterations)
            if step >= _fwu:
                _op_f = current_opacities().reshape(-1)
                loss = loss + _floor_w * torch.relu(_floor - _op_f).mean()

        # ---- 正则项（参考 Faster-GS / 3DGS-MCMC 的做法）----
        # 为什么需要：3DGS 在"背景无人约束"或"视图偏少"时会用**大量半透明浮点**
        # 去凑像素 —— 表现为点云/网格周围一圈雾状杂点（用户反馈的"杂乱"）。
        # 官方 3DGS 与 3DGS-MCMC 都加了不透明度 L1 正则来抑制它：
        #   MCMC 论文用 LAMBDA_OPACITY_REGULARIZATION=0.01（配合 scale 正则 0.01），
        #   Faster-GS 沿用同一套（见其 Trainer.py 顶部常量）。
        # 这里按"平均不透明度"做 L1：对**每个高斯**都要为它的不透明度付出代价，
        # 于是"开着但不贡献有效像素"的浮点会被优化掉。
        if cfg.opacity_reg > 0:
            loss = loss + float(cfg.opacity_reg) * current_opacities().mean()
        if cfg.scale_reg > 0:
            # 尺度正则：抑制个别高斯长到过大（也是浮点的常见形态）
            loss = loss + float(cfg.scale_reg) * current_scales().mean()

        # ---- 掩码外 alpha 正则（`bg_alpha_reg`，默认 0 = 关闭）----
        #
        # 为什么必须补这一项（2026-10 实测）：`composite` 只把**掩码外的 GT 涂成背景色**，
        # 也就是只约束"那里必须是黑的"，**没有约束"那里不许有几何"**。
        # 模型于是用**不透明的黑色高斯**把背景填满：
        #
        #     实测 img3-final：掩码外 alpha 均值 **0.70**（正确应≈0）、
        #     且 1,161 个高斯（0.8%）从未进入任何掩码，却占 **14.1% 视觉权重**。
        #
        # 在查看器里转一圈，这层黑几何就是用户说的"背面混乱"。
        # 做法：对掩码外的 alpha 直接加惩罚（掩码内不罚）。
        #
        # ⚠️ 双掩码（2026-10-24）：**颜色监督**用紧掩码（`masks_loss`，避免把掩码
        # 外沿的亮背景学成轮廓白边），但**背景 alpha 惩罚**仍用**原掩码**。
        # 否则模型会在收紧的那一圈里把 alpha 压到 0 ⇒ 几何整体内缩、
        # 薄结构被削（实测内部 PSNR 26.48 → 24.15 dB，−2.2）。
        # 默认 `mask_erode_px=0` 时两者相同，行为不变。
        if (cfg.bg_alpha_reg > 0 and _mask is not None
                and bool(_mask.any()) and not bool(_mask.all())):
            _mask_alpha = _mask
            if _erode > 0 and masks:
                _ma = masks[idx]
                if max_side and _ma.shape[0] != gt.shape[0]:
                    _ma = torch.nn.functional.interpolate(
                        _ma[None, None].float(), size=gt.shape[:2], mode="nearest"
                    )[0, 0] > 0.5
                _mask_alpha = _ma.to(device)
            _bg = (~_mask_alpha).to(alphas[0].dtype)
            loss = loss + float(cfg.bg_alpha_reg) * (alphas[0][..., 0] * _bg).mean()

        # ---- 感知损失 LPIPS（用户方案第 3 项）----
        # 动机：L1 是逐像素平均、SSIM 是局部结构，两者都偏"平滑"，对纹理细节
        # （石雕纹样、瓦当）不敏感，渲染容易发糊。LPIPS 用预训练 VGG 的深层特征度量差异。
        #
        # ⚠️ 历史教训（不要凭一轮实验就否定它）：我曾据一次对照把它判为"有害"并**删掉代码**，
        # 那一轮测量用的是 `sh_degree=0` 渲染（丢掉高阶 SH ⇒ 低估 1.91 dB），
        # 而 `sh_degree=0` 抹掉的正是**视角相关**外观 —— 感知损失要改善的恰恰是这部分。
        # 也就是说：**当时那把尺子看不见优化目标**，结论是假阴性。
        # 因此现在保留开关（默认 0），并附上噪声底：同配置 PSNR 散布约 0.09 dB，
        # 小于它的差异不能作为结论。
        if float(getattr(cfg, "lambda_lpips", 0.0) or 0.0) > 0 and not _lpips_disabled:
            if _lpips_fn is None:
                try:
                    from .perceptual import get_lpips
                    _lpips_fn = get_lpips(str(device),
                                          input_size=int(getattr(cfg, "lpips_size", 224) or 224))
                    ctx.info(f"感知损失已启用：λ_lpips={float(cfg.lambda_lpips)} "
                             f"每 {int(cfg.lpips_every)} 步计算一次")
                except Exception as exc:  # noqa: BLE001
                    _lpips_disabled = True
                    ctx.warn(f"LPIPS 不可用，已**永久关闭**该损失项"
                             f"（{type(exc).__name__}: {exc}）；"
                             f"如需启用请先跑 scripts\\fetch_torch_weights.py")
            if _lpips_fn is not None and (step % max(1, int(cfg.lpips_every)) == 0):
                try:
                    lp_val = _lpips_fn(render.clamp(0, 1), gt)
                    loss = loss + float(cfg.lambda_lpips) * lp_val
                    if step % 500 == 0:
                        ctx.emit(f"step {step} LPIPS={float(lp_val):.4f} "
                                 f"（λ={float(cfg.lambda_lpips)}）", level="debug")
                except Exception as exc:  # noqa: BLE001
                    _lpips_disabled = True
                    ctx.warn(f"LPIPS 前向失败，已永久关闭该损失项：{exc}")

        # ---- 2DGS 正则（深度失真抑制 + 法线一致性）----
        # 见 TrainConfig 里 `lambda_depth_distortion` 的说明：3DGS 的累计深度是沿射线的
        # 加权平均，表面位置本身是模糊的 —— 这正是"反投影点云有厚度、网格碎"的根源。
        # 这两个正则让"同一像素上的高斯挤到同一深度"，并让渲染法线与深度几何法线一致。
        #
        # 代价：深度正则需要**额外一次 `render_mode="D"` 渲染**，所以按 `reg_interval`
        # 隔步计算（默认 8 步一次），避免拖慢训练。
        want_depth_reg = float(cfg.lambda_depth_distortion) > 0
        want_normal_reg = float(cfg.lambda_normal) > 0
        reg_every = max(1, int(cfg.reg_interval or 1))
        if (want_depth_reg or want_normal_reg) and (step % reg_every == 0):
            reg = two_dgs_regularizers(
                cfg=cfg, means=means, quats=quats, scales=current_scales(),
                opacities=torch.clamp(current_opacities().squeeze(-1), 0.0, 1.0),
                shs=shs, kk=kk, vm=vm, width=gt.shape[1], height=gt.shape[0],
                alphas=alphas, sh_degree=cfg.sh_degree,
            )
            if reg.get("depth") is not None and want_depth_reg:
                loss = loss + float(cfg.lambda_depth_distortion) * reg["depth"]
            if reg.get("normal") is not None and want_normal_reg:
                loss = loss + float(cfg.lambda_normal) * reg["normal"]
            if step % 500 == 0 and reg:
                parts = [f"{k}={float(v):.4e}" for k, v in reg.items() if v is not None]
                if parts:
                    ctx.emit(f"step {step} 2DGS 正则：" + "  ".join(parts), level="debug")

        # 官方策略要求在 backward 之前 retain 它需要的梯度（means2d）
        if gs_strategy is not None:
            gs_strategy.step_pre_backward(params, optimizers, gs_state, step, meta or {})

        # ---- 反向 ----
        loss.backward()
        # 缓存本步的位置梯度范数：密度控制需要它，但那时 .grad 已被 zero_grad 清掉。
        #
        # ⚠️ 已知缺口（如实记录，避免后人重复试错）：
        # 官方 `DefaultStrategy` 的加密判据用的是 **2D 屏幕空间**的 `means2d` 梯度，
        # 并做两层归一化后才和 `grow_grad2d=2e-4` 比较：
        #     grads[..., 0] *= width / 2.0 * n_cameras    # 换算到 [-1,1] 屏幕空间
        #     grads = grad2d / count.clamp_min(1)         # 再除以"被看到的次数"
        # 本项目现在用的是 **3D 世界空间**的 `means.grad` + 固定阈值
        # `train.densify_grad_threshold`（实测校准过：在 700px / 122 视图下
        # 2e-4 给出 7.4 万高斯、PSNR 26.3 dB，是当前可用的最好结果）。
        #
        # 我曾尝试移植官方那套统计量（grad2d/count + 屏幕空间归一化 + 可见次数），
        # 三次都失败：加密量暴涨（每轮 2000+ 克隆/分裂）、尺度 max 冲到 5.3
        # （远超 `prune_scale3d*scene_scale`≈0.36）、PSNR 崩到 9.4 dB。
        # 原因不只是"少了个 /count"—— 官方的 `duplicate/split/remove/reset_opa`
        # 与统计量是一整套自洽机制（在 `gsplat/strategy/ops.py`），
        # 拆一半移植比不移植更糟。
        #
        # ⇒ 要真正对齐官方，应整体移植 `DefaultStrategy`（约 200 行 +
        #   `ops.py`），而不是零散补几处。在那之前，这里的实测校准值更可靠。
        # ---- 致密化判据（官方口径 vs 旧的 3D 口径）----
        #
        # 官方 `DefaultStrategy` 用**屏幕空间**梯度并除以可见次数：
        #     g = meta["means2d"].grad;  g[...,0] *= width/2*n_cameras
        #     g = g.norm(dim=-1);  g = g / count.clamp_min(1)
        #     densify if g > grow_grad2d
        # 我们旧版用 `means.grad.norm()`（3D 世界空间）+ 固定 2e-4。
        # 实测（scripts/probe_densify_criterion.py）：两者量纲差 ~45 倍，
        # 且选出的高斯集合 **IoU 仅 0.065** —— 加密位置几乎完全不同。
        # 后果已验证：densify_until=15000 时高斯 4.3 倍，PSNR 反降 0.874 dB，
        # 近透明高斯 21.2% -> 36.8%（"雾"变多，细节没变多）。
        grad_norm_cache = None
        _crit = str(getattr(cfg, "densify_criterion", "grad2d") or "grad2d").lower()
        if _crit == "grad2d":
            _m2 = meta.get("means2d") if isinstance(meta, dict) else None
            _radii = meta.get("radii") if isinstance(meta, dict) else None
            if _m2 is not None and _m2.grad is not None:
                with torch.no_grad():
                    gg = _m2.grad.detach().clone()          # [C,N,2]
                    # ⚠️ 官方（`gsplat/strategy/default.py::step_post_backward`）是：
                    #     grads[..., 0] *= width  / 2.0 * n_cameras
                    #     grads[..., 1] *= height / 2.0 * n_cameras
                    #     state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
                    #     grads = state["grad2d"] / count.clamp_min(1)
                    # **两个分量都要缩放**（各自按自己的像素维度换算到 [-1,1]）。
                    # 我的第一版只缩放了 x、且没乘 n_cameras —— 对竖屏素材
                    # （577x1309）y 方向梯度被低估约 650 倍 ⇒ 加密几乎不触发。
                    # 实测后果：406 视图、30000 迭代只得到 **22,675** 个高斯
                    # （每视图 125 个，社区 1000~5000）。
                    _H, _W = float(gt.shape[0]), float(gt.shape[1])
                    _nc = float(gg.shape[0])
                    gg[..., 0] *= _W / 2.0 * _nc
                    gg[..., 1] *= _H / 2.0 * _nc
                    g2d = gg.norm(dim=-1)                    # [C,N]
                    # 除以"被看到次数"（radii>0 视为可见；取不到就按 1）
                    if _radii is not None:
                        vis = (_radii.detach().reshape(-1, 2) > 0).any(dim=-1).float()
                        vis = vis.clamp_min(1.0)
                    else:
                        vis = torch.ones(g2d.shape[-1], device=g2d.device)
                    # 逐视角累积后取最大（官方是累加进 state，这里每步重算，取 max
                    # 与"本步最需要加密的视角"等价，且不引入跨步状态）
                    grad_norm_cache = (g2d / vis[None]).max(dim=0).values.detach().clone()
            else:
                ctx.warn("拿不到 means2d 梯度（densify_criterion=grad2d），"
                         "本轮退回 3D 梯度；如需稳定请检查 gsplat 版本")
        # 兜底：grad2d 没算出来（拿不到 means2d 梯度）时才用 3D 梯度。
        #
        # ⚠️ 这里**曾经有一个孤儿 `elif`**：那是打 `densify_criterion` 补丁时
        # 留下的旧代码（原本属于"if means.grad is not None"那条链）。补丁之后
        # 它变成挂在新 `if` 上的 `elif means.grad is not None:`，
        # 语义从"拿不到 means2d"错成了"**grad2d 算成功了**" ⇒
        # **正确的 2D 梯度被 3D 梯度覆盖**，还打出一条假告警
        # "未取到 means2d 梯度…"。实测两个作业各 **30000 次**（每步一次），
        # 导致 grad2d 从未真正生效：`img3-auto3` 只得到 8,563 个高斯、PSNR 19.34。
        if grad_norm_cache is None and means.grad is not None:
            with torch.no_grad():
                grad_norm_cache = means.grad.norm(dim=-1).detach().clone()
            ctx.warn("未取到 means2d 梯度，密度控制退回 3D 梯度"
                     "（量纲不同，阈值需另行标定）")

        # ---- 利用率统计（方案 B）：用 gsplat 的 `tiles_per_gauss`，不是梯度 ----
        #
        # ⚠️ 我曾经用 `|∂L/∂opacity|` 当"利用率"，结果把 **95% 的高斯剪掉**、
        #    PSNR 崩到 21.86。原因：**梯度小 ≠ 没用**，而通常表示"**已经拟合好了**"。
        #    于是我把已经学好的高斯剪掉、留下还在被纠正的 —— 方向反了。
        #
        # 正确语义：**这个高斯到底有没有被哪个像素需要**。
        # gsplat 的 `meta["tiles_per_gauss"]`（形状 [n_cam, N]）就是它在本视角与
        # 多少个屏幕 tile 相交 —— 正是官方密度控制用的 `count`(grad2d/count) 的对应物，
        # 而且**免费**（渲染时已算好，不需要额外 pass）。
        # 一个长期 tiles=0 的高斯不会被任何像素采样到 ⇒ 它确实没用。
        if util_state["acc"] is not None and isinstance(meta, dict):
            tpg = meta.get("tiles_per_gauss")
            if tpg is not None:
                with torch.no_grad():
                    t = tpg.reshape(-1).float()
                    if util_state["acc"].numel() != t.numel():
                        util_state["acc"] = torch.zeros_like(t)
                    util_state["acc"] += t

        # 探针：确认渲染确实覆盖了像素、且位置参数拿到梯度。
        # 曾经出现 alpha 和 = 0 / means.grad = 0（相机位姿翻转写错，高斯全在相机背后）。
        # 训练"能跑完"不等于"在工作"，这是最省事的健康断言。
        if step == 1:
            ga = means.grad
            gn = float(ga.abs().sum()) if ga is not None else -1.0
            # ⚠️ 必须 `.detach()`：alphas 仍挂在反向图上，直接 float() 会每步
            # 打一条 UserWarning（"Converting a tensor with requires_grad=True
            # to a scalar may lead to unexpected behavior"），日志被噪音淹没。
            alpha_sum = float(alphas[0].detach().sum())
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

        # ---- 每步尺度上限（`max_scale_ratio`，默认 0 = 关闭）----
        #
        # 为什么需要（2026-10 实测，docs/03 待编号）：尺度裁剪 `prune_scale3d`
        # **只在致密化窗口内**执行（见下面的 `densify_from <= step <= densify_until`
        # 门控）。默认 `densify_until=4000`，于是后 26,000 步**没有任何尺度约束**，
        # 优化器可以把高斯无限拉长去凑残差：
        #
        #     实测 img3-final：最长的一个高斯 scale=[15.806, 0.0, 0.033]、
        #     opacity=1.0，而整个模型只有 4.16 米宽；
        #     aniso>100 的高斯只占 10.2% 数量，却承担 **94.8% 的视觉权重**。
        #
        # 后果正是用户看到的三件事：训练视角像面、自由视角是**细丝**、
        # 用这些针跑泊松得到**皱成一团的网格**。
        #
        # 这里按"场景尺度的比例"钳制 log 尺度（与 `prune_scale3d` 同一个量纲），
        # 每步生效、不依赖致密化窗口。默认 0 = 不改变既有行为。
        # ---- 每步长细比上限（max_aspect，默认 0 = 关闭）----
        #
        # max_scale_ratio 只限最长轴；最短轴仍可以趋近 0。
        # 实测：最细一个高斯 scale=[0.0, 0.0064, 0.0177]、opacity=1.0、
        # aniso=**7509** —— 一个零厚度、不透明的薄片，从侧面看就是一道丝。
        # 这里把 smin 抬到 smax/max_aspect（在 log 域里 clamp）。
        #
        # ⚠️ **与 2DGS 压平互斥**（2026-10-22，用户要求"有冲突就改"）：
        # `flatten_2dgs` 的目标就是把最短轴压到接近 0（变成圆盘），
        # 而本钳制每步又把最短轴抬回来 —— 两者对撞，压平的收益被完全抵消。
        # 既然它们语义冲突，就不该同时生效：压平开启时**跳过长细比钳制**
        # （压平本身已经限制了"针"，而它是沿法线的薄，不是无方向细丝）。
        _flatten_on = bool(getattr(cfg, "flatten_2dgs", False))
        if _flatten_on and float(getattr(cfg, "max_aspect", 0.0) or 0.0) > 1.0:
            if not locals().get("_aspect_conflict_logged"):
                ctx.warn("flatten_2dgs 与 max_aspect 语义冲突：本次跳过 max_aspect "
                         "（压平需要最短轴→0）。如需细丝治理请二选一。")
                _aspect_conflict_logged = True
        if (not _flatten_on
                and getattr(cfg, "max_aspect", 0.0) and float(cfg.max_aspect) > 1.0):
            with torch.no_grad():
                _s = params["scales"].data
                _smax = _s.max(dim=1, keepdim=True).values
                _floor = _smax - float(np.log(float(cfg.max_aspect)))
                _s.clamp_(min=_floor)

        if getattr(cfg, "max_scale_ratio", 0.0) and float(cfg.max_scale_ratio) > 0:
            with torch.no_grad():
                _cap = float(np.log(float(cfg.max_scale_ratio) * scene_scale))
                params["scales"].data.clamp_(max=_cap)

        # ---- 结构更新（后向之后进行，官方称之为 post-backward step）----
        # 官方 simple_trainer.py 的顺序就是：
        #   optimizer.step() -> optimizer.zero_grad(set_to_none=True) -> strategy.step_post_backward(...)
        if gs_strategy is not None:
            # 官方整套：grow -> prune -> 清统计量 ->（按周期）重置不透明度
            n_before = int(params["means"].shape[0])
            gs_strategy.step_post_backward(params, optimizers, gs_state, step,
                                           meta or {}, packed=False)
            n_after = int(params["means"].shape[0])
            if n_after != n_before:
                refresh_params()
                if step % 500 == 0 or abs(n_after - n_before) > 5000:
                    ctx.info(f"step {step} 官方密度控制：{n_before} -> {n_after} 个高斯")
            else:
                refresh_params()
        # 判据必须在这里用缓存量：zero_grad 之后 means.grad 已是 None，
        # 直接读会恒为 None -> 一次都不加密（早期踩过）。
        # 传入的是 backward 之后缓存的位置梯度范数（zero_grad 之后就读不到了）。
        elif (cfg.densify_from <= step <= cfg.densify_until
                and step % cfg.densify_interval == 0
                and (cfg.opacity_reset_interval <= 0
                     or step % cfg.opacity_reset_interval >= cfg.pause_refine_after_reset)):
            densify_and_prune(step, grad_norm_cache)

        # ================= 不透明度重置：位置见"评估与保存"之后 =================
        # ⚠️ 顺序红线：重置必须放在**保存检查点与评估之后**。
        # 踩坑（很隐蔽，且污染所有下游产物）：重置把 logit 不透明度钳到
        # `logit(prune_opa*2)`（默认 logit(0.01)）。而 step 15000 同时满足
        # `15000 % 3000 == 0`（重置步）与"最后一步" ——
        # 若重置在保存之前，"最终检查点"存下的就是**刚被重置完的状态**：
        #     实测 iteration_15000 有 90.6% 的高斯不透明度被钉在 0.01、max 也是 0.01，
        #     从该 PLY 渲染一片黑、逐视图 PSNR 仅 ~10 dB；
        #     而同一时刻 trainer 内存里的模型（重置前）报 22~24 dB。
        # 所以循环末尾（评估/保存之后）才做重置。

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
            # 最后一次评估用**全部**留出视图（可信的最终指标）；
            # 中途用固定的少数视图（确定性，避免与训练循环相位混叠）。
            is_final = (step == cfg.iterations)
            ev_idx_now = psnr_idx if is_final else psnr_idx_log
            # 全部用关键字传参：位置传参曾因顺序写错而静默失效（形状恰好兼容）
            eval_res = evaluate(
                rasterization=rasterization,
                means=means,
                quats=quats,
                scales=current_scales(),
                opacities=current_opacities(),
                shs=shs,
                imgs=imgs,
                masks=masks,
                ks=ks,
                viewmats=viewmats,
                eval_idx=ev_idx_now,
                sh_degree=cfg.sh_degree,
                background=background,
                device=device,
                max_side=max_side,
                # 只在最后一次评估时导出"渲染 vs 真值"对比图（诊断位姿约定用）
                dump_dir=(cfg.out_dir / "diagnostics" / "render_vs_gt" if is_final else None),
                # 多指标评估：最后一次评估才算 LPIPS（VGG 前向不便宜），
                # 日志用的高频评估只算 PSNR/SSIM。
                want_lpips=bool(getattr(cfg, "eval_metrics", True)) and is_final,
            )
            cur_psnr = float(eval_res.get("psnr", 0.0))
            _cur_masked = eval_res.get("psnr_masked")
            _cur_ssim_m = eval_res.get("ssim_masked")
            cur_ssim = float(eval_res.get("ssim", 0.0))
            cur_lpips = eval_res.get("lpips")
            psnr_log.append({"step": step, "psnr": round(cur_psnr, 3),
                         "psnr_masked": (round(float(_cur_masked), 3)
                                          if _cur_masked is not None else None),
                             "ssim": round(cur_ssim, 4),
                             "ssim_masked": (round(float(_cur_ssim_m), 4)
                                              if _cur_ssim_m is not None else None),
                             "lpips": (round(float(cur_lpips), 4)
                                       if cur_lpips is not None else None),
                             "views": len(ev_idx_now), "final": is_final})
            tag = "（训练视图，非泛化指标）" if psnr_on_train else ""
            tag += "（全留出集）" if is_final and not psnr_on_train else ""
            extra = f" SSIM={cur_ssim:.4f}"
            if cur_lpips is not None:
                extra += f" LPIPS={float(cur_lpips):.4f}"
            ctx.emit(f"step {step} 评估 PSNR={cur_psnr:.2f} dB{extra} "
                     f"[{len(ev_idx_now)} 视图]{tag}", level="info",
                     progress=step / cfg.iterations)

            # ---- 挑选"最佳检查点"（用户方案第 6 项）----
            # 默认按 PSNR；可改为 ssim / lpips（后者越小越好）。
            # 动机：PSNR 高但 SSIM/LPIPS 差的模型往往视觉上模糊，不适合作为导出依据。
            bm = str(getattr(cfg, "best_metric", "psnr") or "psnr").lower()
            if bm == "ssim":
                cur_key = cur_ssim if cur_lpips is not None or True else cur_ssim
                better = cur_ssim > best_psnr + 1e-9
            elif bm in ("lpips", "lpips_inv") and cur_lpips is not None:
                # lpips 越小越好 ⇒ 用负值比较，保持 best_psnr 变量语义为"越大越好"
                cur_key = -float(cur_lpips)
                better = cur_key > best_psnr + 1e-9
            else:
                cur_key = cur_psnr
                better = cur_psnr > best_psnr + 1e-6
            if better:
                best_psnr = float(cur_key)
                best_step = step
                best_metrics_now = {"psnr": cur_psnr, "ssim": cur_ssim,
                                    "lpips": (float(cur_lpips)
                                              if cur_lpips is not None else None),
                                    "selected_by": bm}
                best_dir = gs_dir / f"iteration_{step}"
                save_ply(
                    path=best_dir / "point_cloud.ply",
                    means=means,
                    scales=current_scales(),
                    quats=quats,
                    opacities=current_opacities(),
                    shs=shs,
                    sh_degree=cfg.sh_degree,
                )
                best_ply = best_dir / "point_cloud.ply"

        if step % 5000 == 0 or step == cfg.iterations:
            # ⚠️ 顺序红线：**保存检查点必须在"不透明度重置"之前**。
            # 踩坑（很隐蔽，且影响所有下游产物）：重置会把 logit 不透明度钳到
            # `logit(0.01)`，而 step 15000 恰好是 `15000 % 3000 == 0` 的重置步 ——
            # 于是"最后保存的检查点"= "刚被重置完的状态"：
            #     实测 iteration_15000 有 90.6% 的高斯不透明度被钉在 0.01、max 也是 0.01
            #     从该 PLY 渲染出来**一片黑**，逐视图 PSNR 只有 ~10 dB，
            #     而 trainer 内存里的同一模型（重置前）报 22~24 dB。
            # 导出阶段优先用 best_ply（可以避开），但"最终模型"会被这个坏快照污染。
            # 关键字传参：位置传参出过一次静默错位（evaluate 的 PSNR 因此恒为 0）
            save_ply(
                path=gs_dir / f"iteration_{step}" / "point_cloud.ply",
                means=means,
                scales=current_scales(),
                quats=quats,
                opacities=current_opacities(),
                shs=shs,
                sh_degree=cfg.sh_degree,
            )
            ctx.info(f"已保存检查点 iteration_{step}")

        # ================= 不透明度重置的替代方案（用户方案第 5 项）=================
        #
        # 为什么要替代官方重置：它是一次**不连续跳变**（把 logit 不透明度硬钳到
        # logit(0.01)）。实测后果是"所有高斯的梯度同时变大" ⇒ 触发致密化 ⇒
        # 高斯数 279,534（不重置仅 78,526），且"最终检查点"被污染成一片黑。
        # 下面两个替代都**不做跳变**，因此不引入那波梯度尖峰。
        # 同样遵守顺序红线：放在"保存 + 评估"之后。
        if step > 0 and step != cfg.iterations:
            # 方案 C：梯度感知的不透明度衰减（平滑淡出低贡献高斯）
            _decay = float(getattr(cfg, "opacity_decay", 1.0) or 1.0)
            _dint = int(getattr(cfg, "opacity_decay_interval", 50) or 50)
            if 0.0 < _decay < 1.0 and _dint > 0 and step % _dint == 0:
                with torch.no_grad():
                    # 不透明度是 logit 参数化：σ(x·k) = σ(x)/(σ(x)+k(1-σ(x)))
                    # 这是"实际不透明度乘以 k"在 logit 域的**精确**等价形式，
                    # 不需要先 sigmoid 再 logit（那会引入数值误差与额外拷贝）。
                    s = torch.sigmoid(opac.data)
                    new_s = s / (s + _decay * (1.0 - s)).clamp_min(1e-12)
                    opac.data.copy_(torch.logit(new_s.clamp(1e-6, 1 - 1e-6)))

            # 方案 A/B：按**利用率**剪枝（只剪真正没用的，不动其他高斯）
            # 判据用 |∂L/∂opacity| 的累积：一个高斯若长期对渲染颜色没有影响，
            # 它的不透明度梯度就长期接近 0 ⇒ 它不"被需要"。
            # 与"随机/一刀切重置"的区别在于：**只删掉确实无贡献的**。
            _uint = int(getattr(cfg, "utilization_prune_interval", 0) or 0)
            _uuntil = int(getattr(cfg, "utilization_prune_until", 0) or 0)
            _acc = util_state["acc"]
            if (_uint > 0 and _acc is not None and step % _uint == 0 and step > _uint
                    and (_uuntil <= 0 or step <= _uuntil)):
                _q = float(getattr(cfg, "utilization_prune_quantile", 0.2) or 0.2)
                n_now = int(params["means"].shape[0])
                # ⚠️ 长度对齐：致密化**就在本步之前**可能增删高斯，而累加器是反向时按
                # 更早的长度算的。实测 `step 800` 出现 "高斯 38961，统计量长度 37278"
                # ⇒ 若要求严格相等，剪枝会被永久跳过（"参数生效了但功能从不执行"）。
                # 这里**对齐到较短长度**：前 m 个高斯在本次致密化中未被重排
                # （`_grow` 是 append 式、`_apply_keep` 用 `torch.where` 保序），
                # 所以前缀对应关系仍然有效，只对前缀做判定。
                m = min(n_now, int(_acc.numel()))
                if m >= 64:
                    thresh = _quantile(_acc[:m], _q)
                    keep_m = _acc[:m] > thresh
                    _nz = int((_acc[:m] > 0).sum())
                    print(f"[利用率剪枝] step {step} 高斯 {n_now} 统计 {int(_acc.numel())} "
                          f"对齐 {m}：非零 {_nz} 阈值={thresh:.3f} "
                          f"保留={int(keep_m.sum())}/{m}", flush=True)
                    # 保护：至少留 50%，避免把有效结构剪掉
                    if int(keep_m.sum()) >= max(16, int(0.5 * m)):
                        # 只对前缀做判定；致密化**新增**的那部分高斯无法判断
                        # （它们还没有任何梯度历史），一律保留，避免误删。
                        if m < n_now:
                            keep = torch.cat([
                                keep_m.to(device),
                                torch.ones(n_now - m, dtype=torch.bool, device=device)])
                        else:
                            keep = keep_m.to(device)
                        _apply_keep(params, optimizers, keep)   # 原地修改，返回 None
                        # ⚠️ `_apply_keep` 只改 `params` 字典，`train()` 里的
                        # `means`/`quats`/`scales`/`opac` 等局部变量仍指向**旧**张量。
                        # 不重新绑定，后面的保存/评估会用错数据（形状甚至可能恰好兼容
                        # 而不报错）。密度控制路径也是这么做的（见 densify_and_prune）。
                        refresh_params()
                        n_after = int(params["means"].shape[0])
                        ctx.info(f"step {step}：利用率剪枝（阈值 {thresh:.3e}，"
                                 f"分位 {_q}）{n_now} -> {n_after} 个高斯")
                        # ⚠️ 形状红线：`grad_norm_cache` 是按**旧**高斯数算出来的，
                        # 下一轮 densify_and_prune 会按它做布尔运算 ⇒ 不重算就
                        # `size of tensor a (38861) must match b (29757)`（实测崩溃）。
                        # 剪枝后最稳妥的做法是**置空**：`grad_norm_cache=None` 会让
                        # densify 那一轮退回固定阈值，不会用错长度的张量。
                        grad_norm_cache = None
                        util_state["acc"] = torch.zeros(n_after, device=device)
                        _sh_needs_refresh = True
                elif util_state["acc"].numel() != n_now:
                    util_state["acc"] = torch.zeros(n_now, device=device)

            # ---- 方案 A：恢复感知剪枝（recovery-aware pruning）----
            # 判据是"**恢复速度**"，不是"梯度大小"也不是"tile 数"：
            #   正常高斯的贡献在所有视角都是正的 ⇒ 被衰减后**很快被拉回来**；
            #   浮空物只在少数视角有贡献 ⇒ 不透明度震荡、**恢复慢**。
            # 用户原方案以"重置"为打压手段；本项目不用重置（−0.59 dB），
            # 所以改用方案 C 的**平滑衰减**作为打压，判据与剪枝对象完全保持原意。
            _rint = int(getattr(cfg, "recovery_prune_interval", 0) or 0)
            _runtil = int(getattr(cfg, "recovery_prune_until", 0) or 0)
            if _rint > 0 and (_runtil <= 0 or step <= _runtil):
                _snap = recov_state["snap"]
                # ① 该拍快照了吗？（在衰减之前拍，之后才能看"恢复了多少"）
                if step % _rint == 0 or _snap is None:
                    with torch.no_grad():
                        recov_state["snap"] = torch.sigmoid(opac.data).detach().clone()
                        recov_state["snap_step"] = step
                # ② 快照后已过 recovery_window 步 ⇒ 算恢复率并剪最慢的那批
                elif (step - recov_state["snap_step"] >= int(cfg.recovery_window)
                      and _snap is not None and _snap.numel() == int(opac.data.numel())):
                    with torch.no_grad():
                        now_o = torch.sigmoid(opac.data)
                        # 恢复率 = 当前 / 快照（快照为 0 的用极小值避免除零）
                        ratio = now_o / _snap.clamp_min(1e-6)
                        q = float(getattr(cfg, "recovery_prune_quantile", 0.2) or 0.2)
                        thr = _quantile(ratio, q)
                        keep_r = ratio > thr
                        n_before = int(opac.data.numel())
                        n_nonzero = int((_snap > 1e-6).sum())
                        print(f"[恢复感知剪枝] step {step} 快照@step "
                              f"{recov_state['snap_step']}：有效率 {n_nonzero}/{n_before} "
                              f"阈值={thr:.3f} 保留={int(keep_r.sum())}", flush=True)
                        if int(keep_r.sum()) >= max(16, int(0.5 * n_before)):
                            _apply_keep(params, optimizers, keep_r.to(device))
                            refresh_params()
                            grad_norm_cache = None
                            util_state["acc"] = torch.zeros(
                                int(params["means"].shape[0]), device=device)
                            _sh_needs_refresh = True
                    recov_state["snap"] = None
                    recov_state["snap_step"] = -1

        # ================= 不透明度重置（必须在保存/评估之后）=================
        # ⚠️ 顺序红线（踩过，且会污染所有下游产物）：
        # 重置把 logit 不透明度钳到 `prune_opa*2` 的 logit（默认 logit(0.01)）。
        # 而 step 15000 恰好满足 `15000 % 3000 == 0`，**同时也是最后一步** ——
        # 若把重置放在保存之前，"最终检查点"保存的就是**刚被重置完的状态**：
        #     实测 iteration_15000 有 90.6% 的高斯不透明度被钉在 0.01、max 也是 0.01，
        #     从该 PLY 渲染出来一片黑、逐视图 PSNR 仅 ~10 dB；
        #     而同一时刻 trainer 内存里的模型（重置前）报 22~24 dB。
        # 两个对策：
        #   1) 重置整体后移到"保存 + 评估"之后（本段的位置）
        #   2) **最后一步跳过重置** —— 最后一步重置没有任何恢复机会，
        #      只会让最终快照变差，没有任何好处
        if (gs_strategy is None and cfg.opacity_reset_interval and step > 0
                and step % cfg.opacity_reset_interval == 0
                and step != cfg.iterations):
            with torch.no_grad():
                reset_val = float(cfg.reset_opacity if cfg.reset_opacity > 0
                                  else cfg.prune_opacity * 2.0)
                opac.data.clamp_(max=float(torch.logit(torch.tensor(reset_val, device=device))))
            if cfg.pause_refine_after_reset > 0:
                ctx.info(f"step {step}：不透明度重置为 {reset_val:.3f}，"
                         f"暂停精修 {cfg.pause_refine_after_reset} 步")

    # ---- 收尾：保存最终模型与指标 ----
    final_dir = gs_dir / f"iteration_{cfg.iterations}"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_ply = final_dir / "point_cloud.ply"
    save_ply(
        path=final_ply,
        means=means,
        scales=current_scales(),
        quats=quats,
        opacities=current_opacities(),
        shs=shs,
        sh_degree=cfg.sh_degree,
    )

    # ---- 收尾清理：删掉不透明度低于阈值的高斯（Faster-GS 的 MIN_OPACITY_AFTER_TRAINING）----
    # 为什么放在最后单独做：这些高斯对渲染几乎无贡献，但会出现在导出的点云/网格里
    # （表现为"周围一圈雾状杂点"）。训练中途删它们会扰动密度控制，收尾删最稳妥。
    # Faster-GS 用 1/255 ≈ 0.0039。
    pruned_final = 0
    if cfg.min_opacity_final > 0:
        with torch.no_grad():
            keep = current_opacities().reshape(-1) >= float(cfg.min_opacity_final)
            pruned_final = int((~keep).sum())
            if pruned_final:
                _apply_keep(params, optimizers, keep)
                refresh_params()
                final_ply.parent.mkdir(parents=True, exist_ok=True)
                save_ply(
                    path=final_ply,
                    means=params["means"],
                    scales=current_scales(),
                    quats=params["quats"],
                    opacities=current_opacities(),
                    shs=params["shs"],
                    sh_degree=cfg.sh_degree,
                )
                ctx.info(f"收尾清理：删除 {pruned_final} 个不透明度 < "
                         f"{cfg.min_opacity_final:.4f} 的高斯，剩 {params['means'].shape[0]} 个")

    metrics = {
        "iterations": cfg.iterations,
        "gaussians": int(means.shape[0]),
        "resolution": [int(w), int(h)],
        "train_views": len(train_idx),
        "eval_views": len(eval_idx),
        # 评估视图数（无留出时 = 训练视图数）；psnr_on_train 说明这个 PSNR 不是泛化指标
        "psnr_views": len(psnr_idx),
        "psnr_on_train": psnr_on_train,
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "final_loss": loss_log[-1]["loss"] if loss_log else None,
        "psnr": psnr_log[-1]["psnr"] if psnr_log else None,
        # 掩码内 PSNR：掩码生效后全图 PSNR 会被"该黑的背景"主导而失去意义
        # （实测 26.3 → 3.4 dB 而目视更好），物体区域画质看这个。
        "psnr_masked": (psnr_log[-1].get("psnr_masked") if psnr_log else None),
        # 多指标评估（用户方案第 6 项）：PSNR 高不等于视觉好，
        # 同时记录 SSIM / LPIPS，便于判断"哪个检查点真的更好"。
        "ssim": (psnr_log[-1].get("ssim") if psnr_log else None),
        "ssim_masked": (psnr_log[-1].get("ssim_masked") if psnr_log else None),
        "lpips": (psnr_log[-1].get("lpips") if psnr_log else None),
        "psnr_curve": psnr_log,
        "loss_curve": loss_log[-200:],
        "ply": str(final_ply),
        # 最佳检查点（导出优先用它）：3DGS 会"过训练"，最后一步往往不是最好的
        "best_psnr": (round(best_psnr, 3) if best_psnr != float("-inf") else None),
        "best_step": best_step or None,
        "best_ply": (str(best_ply) if best_ply else str(final_ply)),
        # 最佳检查点的完整指标 + 它是由哪个指标选出来的
        "best_metrics": (best_metrics_now or None),
        "best_metric_used": str(getattr(cfg, "best_metric", "psnr") or "psnr"),
    }
    (cfg.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    msg = (f"训练完成：{metrics['gaussians']} 个高斯，用时 {metrics['elapsed_min']} 分钟"
           + (f"，PSNR {metrics['psnr']:.2f} dB" if metrics["psnr"] else ""))
    if metrics["best_psnr"] and metrics["psnr"] and metrics["best_psnr"] > metrics["psnr"] + 0.1:
        msg += (f"；**最佳检查点在 step {metrics['best_step']}"
                f"（{metrics['best_psnr']:.2f} dB），导出将采用它**")
    ctx.success(msg)
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


def _apply_keep(params: dict, optimizers: dict, keep) -> None:
    """按 keep 掩码裁剪高斯（官方 remove 模式）。

    ⚠️ 这里**不能**用模块级 `@torch.no_grad()` 装饰器：本模块在没有 torch 的环境里
    也要能被 import（安装器的 core-only 安装、`gsflow doctor` 的环境自检都会 import 它），
    而 `torch` 在那种情况下是 None ⇒ 装饰器求值直接
    `AttributeError: 'NoneType' object has no attribute 'no_grad'`，连 CLI 都起不来。
    （这个 bug 是安装器在干净目录实测时抓到的。）
    """
    import torch

    with torch.no_grad():
        _apply_keep_inner(params, optimizers, keep, torch)


def _apply_keep_inner(params: dict, optimizers: dict, keep, torch) -> None:
    sel = torch.where(keep)[0]

    def param_fn(name, p):
        return torch.nn.Parameter(p[sel].clone(), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return v[sel].clone()

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


def _grow(params: dict, optimizers: dict, duplicate_mask, split_mask,
          revised_opacity: float = 0.1) -> tuple[int, int]:
    """加密：对 duplicate_mask 克隆、对 split_mask 分裂（官方 duplicate/split 模式）。

    同 `_apply_keep`：**不能用模块级装饰器** —— 无 torch 环境下 import 本模块就会崩
    （安装器的 core-only 路径实测）。
    """
    import torch

    with torch.no_grad():
        return _grow_inner(params, optimizers, duplicate_mask, split_mask,
                           revised_opacity, torch)


def _grow_inner(params: dict, optimizers: dict, duplicate_mask, split_mask,
                revised_opacity: float, torch) -> tuple[int, int]:
    """实际执行加密（调用方已进入 `torch.no_grad()`）。

    split 的两个细节照官方：
      * 尺度除以 1.6（`split` 的官方系数）；
      * 分裂出的高斯**不透明度调低**（官方 `revised_opacity`，默认 0.1），
        避免分裂后过度堆积。
    """
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
        # SH 的 DC 分量一起复制，高阶分量为 0（与官方 split/duplicate 一致）
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
        # 分裂：位置抖动 2*scale，尺度除以 1.6，不透明度调低（官方做法）
        scales = torch.exp(params["scales"].data[sel_s])
        offsets = torch.randn_like(params["means"].data[sel_s]) * 2.0 * scales

        def param_fn(name, p):
            extra = p[sel_s].clone()
            if name == "means":
                extra = extra + offsets
            elif name == "scales":
                extra = torch.log(torch.exp(extra) / 1.6)
            elif name == "opacities":
                # logit(revised_opacity)：分裂出的高斯不透明度调低（官方 0.1）
                import math

                extra = torch.full_like(extra, math.log(revised_opacity / (1.0 - revised_opacity)))
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
             eval_idx, sh_degree, background, device, max_side, dump_dir=None,
             want_lpips: bool = False):
    """在评估视图上算 **PSNR / SSIM /（可选）LPIPS**，返回 dict。

    为什么返回多指标（用户方案第 6 项）：**PSNR 高不等于视觉好**。
    PSNR 是逐像素 MSE 的对数，对"整体偏一点但纹理全糊"的渲染很宽容；
    而这类模型导出的网格/贴图观感明显更差。所以同时记录 SSIM 与 LPIPS，
    并允许用它们挑选"最佳检查点"（`TrainConfig.best_metric`）。

    dump_dir 的用途：位姿约定（cam_to_world 的轴翻转）一旦写错，
    渲染在数值上仍可能有非零 PSNR，只看 PSNR 难以判断。
    把两者并排存成 PNG 后可以直接肉眼确认几何是否对齐。

    踩坑：调用方曾用位置参数按错误顺序传入（opacities 落到 scales 上、
    background 落到 masks 上），因为各张量形状恰好兼容，**不会报错**，
    只会让 PSNR 恒为 0。所以这里一律关键字调用，并把顺序写清楚。

    返回：{"psnr": float, "ssim": float, "lpips": float|None, "views": int}
    """
    import torch

    if not eval_idx:
        return {"psnr": 0.0, "ssim": 0.0, "lpips": None, "views": 0}
    psnrs: list[float] = []
    psnrs_in: list[float] = []   # 掩码内 PSNR（物体区域画质）
    ssims_in: list[float] = []   # 掩码内 SSIM（同上）
    ssims: list[float] = []
    lpipss: list[float] = []
    lp = None
    if want_lpips:
        try:
            from .perceptual import get_lpips
            lp = get_lpips(str(device))
        except Exception as exc:  # noqa: BLE001
            print(f"[evaluate] LPIPS 不可用，已跳过（{type(exc).__name__}: {exc}）")
            lp = None
    with torch.no_grad():
        for i in eval_idx:
            gt = imgs[i].to(device)
            colors, alphas, _ = rasterization(
                means=means,
                quats=quats / (quats.norm(dim=-1, keepdim=True) + 1e-8),
                scales=scales,
                opacities=torch.clamp(opacities.squeeze(-1), 0.0, 1.0),
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
            # ---- 掩码内 PSNR：训练把掩码外压成背景色之后，**全图 PSNR 必然失效** ----
            #
            # 背景被有意压黑后，全图 MSE 主要来自"该黑的地方渲成了亮的模型"，
            # 与"物体画得好不好"几乎无关。实测修复掩码后全图 PSNR 直接从
            # 26.3 dB 掉到 **3.4 dB**，而目视质量明显更好 —— 这就是指标与目标脱节。
            # 所以必须同时给出**只统计掩码内像素**的 PSNR（物体区域的画质）。
            if masks is not None and i < len(masks):
                _m = masks[i]
                if _m.shape[0] != gt.shape[0]:
                    _m = torch.nn.functional.interpolate(
                        _m[None, None].float(), size=gt.shape[:2], mode="nearest"
                    )[0, 0] > 0.5
                _m = _m.to(device)
                if bool(_m.any()):
                    d2 = (render - gt) ** 2
                    mse_in = d2[_m].mean().clamp_min(1e-10)
                    psnrs_in.append(float(10.0 * torch.log10(1.0 / mse_in)))
            try:
                s = ssim_simple(render.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
                ssims.append(float(s))
                # 掩码内 SSIM：掩码生效后全图 SSIM 会被"该黑的背景"主导
                # （实测 0.952 → 0.368，而目视质量明显更好）⇒ 物体区域要看这个。
                if masks is not None and i < len(masks):
                    _m2 = masks[i]
                    if _m2.shape[0] != gt.shape[0]:
                        _m2 = torch.nn.functional.interpolate(
                            _m2[None, None].float(), size=gt.shape[:2], mode="nearest"
                        )[0, 0] > 0.5
                    _m2 = _m2.to(device)
                    if bool(_m2.any()):
                        s_in = ssim_simple(
                            (render * _m2[..., None]).permute(2, 0, 1)[None],
                            (gt * _m2[..., None]).permute(2, 0, 1)[None])
                        ssims_in.append(float(s_in))
            except Exception:  # noqa: BLE001
                pass
            if lp is not None:
                try:
                    lpipss.append(float(lp(render.clamp(0, 1), gt)))
                except Exception:  # noqa: BLE001
                    lp = None
            if dump_dir is not None:
                _dump_compare(dump_dir, i, render, gt)
    return {
        "psnr": sum(psnrs) / len(psnrs),
        # 掩码内 PSNR：训练把掩码外压成背景色之后，**全图 PSNR 不再代表画质**
        # （实测修复掩码后全图从 26.3 → 3.4 dB，而目视质量更好）。
        # 物体区域画质要看这个。
        "psnr_masked": (sum(psnrs_in) / len(psnrs_in)) if psnrs_in else None,
        "ssim": (sum(ssims) / len(ssims)) if ssims else 0.0,
        "ssim_masked": (sum(ssims_in) / len(ssims_in)) if ssims_in else None,
        "lpips": (sum(lpipss) / len(lpipss)) if lpipss else None,
        "views": len(psnrs),
    }


def _dump_compare(dump_dir, view_idx: int, render, gt) -> None:
    """把 "渲染 | 真值 | 差异" 横向拼成一张 PNG（仅用于诊断）。"""
    try:
        from PIL import Image
        import torch

        r = (render.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        g = (gt.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        diff = (np.abs(r.astype(np.int16) - g.astype(np.int16))).astype(np.uint8)
        diff = np.repeat(diff, 3, axis=2) if diff.ndim == 2 else diff
        strip = np.concatenate([r, g, diff], axis=1)
        dump_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(strip).save(dump_dir / f"view_{view_idx:03d}_render_gt_diff.png")
    except Exception:  # 诊断用，失败不影响训练
        pass



def save_ply(path: Path, means, scales, quats, opacities, shs, sh_degree: int) -> Path:
    """导出标准 3DGS PLY（与官方 inria 格式一致，可被 SuperSplat/three.js 读取）。

    ⚠️ 关于两个"隐藏的表示约定"，这里踩过坑：

      * **scale_* 存的是 log 尺度**，不是真实尺度。
        官方 `save_ply` 写的是 `self.scaling`（即 log 尺度），读的时候要 `exp()`。
        本项目调用方传进来的 `scales` 是 `exp(log_scale)` 之后的**真实尺度**，
        所以这里必须先取 log 再写；否则读回时会再 exp 一次，
        误差达 200 倍（实测尺度从 0.005 变成 1.0，渲染成一片糊）。

      * **opacity 存的是 logit**，不是 sigmoid 后的概率。
        调用方传进来的 `opacities` 已经是 sigmoid 后的值，这里要转回 logit。

    这样才能与官方 inria / SuperSplat / three.js 的 PLY 约定互通。
    """
    from plyfile import PlyData, PlyElement
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        xyz = means.detach().cpu().numpy().astype(np.float32)
        normals = np.zeros_like(xyz)
        f_dc = shs[:, 0, :].detach().cpu().numpy().astype(np.float32)
        sh_dim = (sh_degree + 1) ** 2
        f_rest = shs[:, 1:sh_dim, :].detach().cpu().numpy().transpose(0, 2, 1).reshape(len(xyz), -1).astype(np.float32)
        # 真实尺度 -> log 尺度；概率 -> logit（与官方 PLY 约定对齐）
        sc = torch.log(torch.clamp(scales.detach(), min=1e-12)).cpu().numpy().astype(np.float32)
        op = torch.logit(torch.clamp(opacities.detach(), min=1e-6, max=1 - 1e-6)).cpu().numpy().astype(np.float32)
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
