"""配置系统：默认值 + config/config.toml + 运行时覆盖。

三层优先级（后者覆盖前者）：
  1. DEFAULTS（本文件内）
  2. config/config.toml（可提交，不含任何密钥）
  3. 运行时覆盖（CLI 的 --set 点路径 / GUI 表单）

凭据单独走 config/credentials.json（见 credentials.py），永不写进 TOML。
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

from . import paths
from .errors import ConfigError

CONFIG_FILE = paths.CONFIG_DIR / "config.toml"

DEFAULTS: dict[str, Any] = {
    "project": {
        "name": "gsflow",
        # 数据与作业产物目录（可为 D:\\3dgs-workspace\\workspace 等 junction 路径）
        "workspace_dir": str(paths.WORKSPACE_DIR),
    },
    "tools": {
        # 留空表示按 PATH -> tools/ 自带目录 的顺序自动探测
        "ffmpeg": "",
        "ffprobe": "",
        "colmap": "",
        "magick": "",
    },
    # ---- 拍摄模式（决定"要不要保留背景"）----
    # auto（默认）—— 按数据自动判定：**背景区域的帧间差异**。
    #   转台（相机固定）⇒ 背景几乎不动；相机绕行 ⇒ 背景有明显视差。
    #   实测：img3-01 比值 0.015（转台）、maskloss-v2 0.489（绕行），相差 34 倍。
    # turntable —— 物体在转台上转、相机与光源固定 ⇒ **保留背景**（mask.mode=keep），
    #   让静态背景锚定"相机没动"；导出阶段再按掩码/密度把物体裁出来。
    # orbit     —— 相机绕物体行走 ⇒ 抠背景 + 黑底合成是安全的（现状）。
    # 映射只在 gsdata/capture_mode_params.py 一处定义，CLI 与 GUI 共用（避免两端漂移）。
    "capture": {
        "mode": "auto",
    },
    "frames": {
        # 抽帧模式：fps（定帧率）或 count（按目标张数均匀抽）
        "mode": "fps",
        "fps": 2.0,
        "count": 120,
        # 缩放长边；0 表示不缩放
        "long_side": 1920,
        # 可选裁剪 `x,y,w,h`（空=不裁）。**在缩放之前执行**。
        # 为什么需要：物体在画面里很小时，网络容量与高斯预算大量浪费在空白上。
        # 实测 IMG3（576×1280）：牌坊只占 8.5% 像素、联合包围盒仅 16.7%；
        # 裁到 489×316 后前景占比升到 79.6%（9.4 倍），等效分辨率大幅提升。
        # 求法：`python scripts\measure_foreground_bbox.py --job <作业>`
        # （它用全部掩码的**并集**包围盒，保证任何一帧都不被裁掉。）
        "crop": "",
        "blur_threshold": 0.0,   # >0 时按拉普拉斯方差剔除模糊帧
        "max_frames": 0,          # >0 时对最终帧数再限流（均匀抽样）
    },
    "mask": {
        # 分割后端：classical | fastsam | rembg | sam2 | sam3
        #
        # 怎么选（实测对比，见 docs/03 的 G13）：
        #   * classical —— 零依赖的 Otsu/背景模型分割。**背景干净、主体与背景对比明显时
        #     又快又好**：合成基准上 4.3 秒完成、前景占比 14.5%、COLMAP 注册 36/36。
        #     缺点是靠颜色/亮度阈值，复杂背景会失效。
        #   * fastsam  —— 语义分割，能按"物体"切分，**真实照片/有杂物背景时更稳**，
        #     代价是慢（同数据 12.5 秒，且首次需下载权重）。
        #     在"主体是一块铺满纹理的平面"这类非自然场景上，它反而把纹理当背景切掉，
        #     实测前景占比 22.8%、注册率掉到 16/36。
        #   * rembg / sam2 / sam3 —— 通用抠图 / 提示式分割，权重需另行下载。
        #
        # 默认给 fastsam：真实使用场景（手持环绕拍摄）里语义分割更可靠。
        # 若你的素材背景干净（纯色墙、桌面、影棚），换 classical 会明显更快更准。
        "backend": "fastsam",
        # 清理模式：
        #   black   —— 背景填黑（COLMAP 最稳；适合**相机绕行**拍摄）
        #   inpaint —— 纹理补全
        #   keep    —— **保留原始背景**（适合**转台**拍摄：物体转、相机固定）
        # 为什么要 keep：转台拍法里静态背景是"相机确实没动"的唯一证据。
        # 合成黑底会抹掉它 ⇒ 重建只能靠物体自身的旋转解释像素 ⇒
        # 把物体转当成相机转 ⇒ 光照/遮挡不自洽 ⇒ 细节糊、网格碎。
        # 实测（scripts/detect_turntable_capture.py）：img3-01 背景差异 0.34
        # （比值 0.013）= 转台；maskloss-v2 17.88（比值 0.489）= 相机绕行。
        # 判定与策略见 gsdata/capture_mode.py 与 scripts/check_capture_mode.py。
        "mode": "black",
        "dilate_px": 4,
        "feather_px": 2,
        "min_area_ratio": 0.002,  # 小于此面积占比的零散区域丢弃
        "keep_largest": True,     # 只保留最大连通前景
        "batch_size": 4,
        "mask_black_threshold": 8,
        # 分割失败（未检出前景）时：keep=原图直通（下游仍可用）| skip=丢弃该帧
        "on_failure": "keep",
        # 掩码收紧像素数（0=不改变）。实测掩码偏大会学出轮廓白边（docs/03 G134）。
        "erode_px": 0,
    },
    "colmap": {
        "cameras": "SIMPLE_RADIAL",
        "single_camera": True,
        "max_image_size": 2048,
        "max_num_features": 8192,
        "use_gpu": True,
        "guided_matching": True,
        # 匹配数上限：显式给一个非零值。COLMAP 在某些参数组合下
        # （例如 guided_matching 被过滤掉时）会让 max_num_matches 变成 0，
        # 触发 `Check failed: max_num_matches > 0` 直接崩溃。
        "max_num_matches": 32768,
        # SfM 后端：auto=先探外部 COLMAP，不可用则 pycolmap；
        # colmap/pycolmap 可强制。（实测本机 COLMAP 3.12 的 matcher 选项绑定坏了，见 docs/03 G138）
        "sfm_backend": "auto",
        # 给 COLMAP 哪套图：auto=clean 优先；raw=原始帧（黑底会让注册爆跌时用）；clean=强制
        "images": "auto",
        "min_num_matches": 32,
        "multiple_models": False,
        # matcher: auto | sequential | exhaustive | vocab_tree
        "matcher": "auto",
        "sequential_overlap": 12,
        # 闭环检测需要 vocabulary tree 文件（COLMAP 官方 vocab_tree 下载后填绝对路径）；
        # 留空则关闭——没有 vocab tree 时开启 loop_detection 会让 COLMAP 直接崩溃。
        "vocab_tree_path": "",
        "use_masks": True,
        "quality_min_registered_ratio": 0.75,
        "quality_max_reproj_error": 2.0,
    },
    "train": {
        # 训练后端：gsplat（默认，免编译）| official（需自编译 CUDA 光栅化器）
        "backend": "gsplat",
        "iterations": 30000,
        "resolution": 1600,       # 长边；0 = 原图
        "sh_degree": 3,
        "data_device": "cuda",
        # 高斯数量上限（显存守护）。实测：8GB 卡上 557k 高斯训练正常，
        # 1.5M 会明显吃紧；真实大场景可上调，小场景可下调。
        "cap_max": 600000,
        "eval": True,
        "fp16": False,
        "seed": 42,
        # 密度控制（对齐官方 gsplat DefaultStrategy 的语义）
        "densify_from": 500,
        # 密度窗口上界：**实物环绕拍摄（几十~一两百视图）建议 2500~4000**。
        # 真实素材实测只改这一个参数：15000 -> 最终 PSNR 20.88 dB（145k 高斯）；
        # 4000 -> 28.62 dB（74k 高斯）；2500 -> 28.71 dB（57k 高斯）。
        # 官方默认 15000 面向几百~上千视图的大场景，此时后期加密不是过拟合；
        # 视图少时它纯粹在过拟合，所以默认下调。大场景请显式调回 15000。
        "densify_until": 4000,
        "densify_interval": 100,
        # 0 = 关闭"分位自适应阈值"，用固定绝对阈值（与 trainer 默认一致）。
        # 实测分位自适应会导致无界增长，已回退。
        "densify_quantile": 0.0,
        # ---- 自动裁剪（默认开，**在 frames 阶段做**）----
        # 实测 IMG3（576×1280 竖屏）：物体只占 **8.5%** 像素，联合包围盒仅 16.7%。
        # 不裁剪 ⇒ PSNR 20.74 dB；裁到 489×316（前景 79.6%）⇒ **28.78 dB（+8.04）**。
        #
        # ⚠️ **必须在 COLMAP 之前裁**（即 frames 阶段）。实测"先跑 COLMAP 再裁"：
        # COLMAP 会把**整个房间**都重建出来（点云 bbox 13.7×1.6×8.5、
        # 离质心最远 10.63，而正确做法只有 5.7×1.0×5.4 / 4.51），
        # 初始化高斯散布全场景 ⇒ PSNR **19.97 dB**（对照 28.78）。
        #
        # 判据用**帧间时间方差**（`gsdata/auto_crop.detect_bbox_temporal`），不需要掩码：
        # 转台时物体在动（高方差）、背景静止（低方差）；相机绕行时全图都变
        # ⇒ 包围盒≈全图 ⇒ 自然不裁（安全）。
        "auto_crop": True,
        # 在 **train 阶段再裁一次**：**默认关闭**。那个位置太晚
        # （COLMAP 已经建完整个场景，裁图只会让图像与相机不一致），
        # 只在明确知道自己在做什么时打开。
        "auto_crop_in_train": False,
        # 前景联合包围盒面积 / 原图面积 小于该值才裁（否则说明本来就够满）
        "auto_crop_threshold": 0.75,
        # ---- 两项**社区标准质量杠杆**（此前都没用上）----
        # 光栅化模式（Mip-Splatting 思路）：
        #   "classic"（默认）—— 3DGS 原版。
        #   "antialiased"    —— 给 2D 投影加低通滤波，抑制采样不足造成的锯齿/闪烁。
        #     训练分辨率与展示分辨率不同、或视角密集时收益明显。
        # 保守起见默认 classic；实测见 docs/03 G106。
        "rasterize_mode": "classic",
        # AbsGS：用 `means2d.absgrad`（绝对值和）做致密化判据，而不是普通梯度和。
        # 官方 DefaultStrategy 支持；开启后必须同时给 rasterization 传 absgrad=True
        # （代码里已连带处理，否则 meta["means2d"].absgrad 会是 None）。
        "absgrad": False,
        "densify_grad_threshold": 2e-4,
        # ---- 致密化判据来源（本轮审计的核心修正）----
        # "grad2d"（默认）= 官方口径：`means2d.grad` 屏幕空间梯度 / 可见次数
        # "grad3d"        = 旧行为：`means.grad` 世界空间梯度 + 固定阈值
        # 实测（scripts/probe_densify_criterion.py）：两者量纲差 ~45 倍
        # （p50 1.60e-4 vs 3.55e-6），且选中的高斯集合 **IoU 仅 0.065**
        # ⇒ 加密位置几乎完全不同。用 grad3d 时 densify_until=15000
        # 会让高斯涨 4.3 倍而 PSNR 反降 0.874 dB、近透明高斯 21%→37%。
        "densify_criterion": "grad2d",
        # grad2d 口径下的阈值（官方 grow_grad2d 常用值）。
        # **与 grad3d 的 2e-4 不同量纲，不要混用。**
        "densify_grad2d_threshold": 2e-4,
        "densify_grad_max": 5e-3,
        # 绝对梯度致密化（AbsGS：https://ty424.github.io/AbsGS.github.io/）。
        # 官方 `DefaultStrategy(absgrad=...)` 支持：开启后栅格化会额外产出
        # `means2d.absgrad`，致密判据用**绝对值和**而不是"正负相消后的梯度"。
        # 优点是能恢复更细的结构（Faster-GS README 也专门列出这一项），
        # 代价是 backward 多一份开销。默认关闭，需实测对照后再决定。
        "absgrad": False,
        # 密度控制实现：
        #   builtin  —— 本项目的自定义实现（已验证、默认）。`absgrad` 在此路径**不生效**。
        #   official —— 直接复用 gsplat `DefaultStrategy`（含 AbsGS 的 absgrad 支持）。
        # 注意：`absgrad` 只有配合 official 才有意义；走 builtin 时它会被忽略。
        "densify_strategy": "builtin",
        # ---- 2DGS 参数化（压平高斯）----
        # 把高斯压平成 2D 圆盘（只钳最短轴），"最短轴 = 表面法线"这一语义才成立。
        # 这是 2DGS(SIGGRAPH'24) 与 3DGS 的**根本差别**；只搬损失项而不改参数化
        # 会与"拟合辐射场"冲突（实测有害，见 docs/03 G52）。默认关闭，需实测对照。
        "flatten_2dgs": False,
        "flatten_log_scale": -6.0,
        # 以下三个比例都相对 scene_scale（相机中心平均距离），取自官方 gsplat DefaultStrategy。
        # prune_scale3d 是**防发散的关键约束**：漏掉它会让尺度无界增长（实测 log尺度到 739）。
        "grow_scale3d": 0.01,          # 尺度 <= 0.01*scene_scale 走克隆，大于走分裂
        "prune_scale3d": 0.1,          # 尺度 > 0.1*scene_scale 直接裁剪
        "prune_scale_after_reset": True,  # 尺度裁剪只在 step > opacity_reset_interval 后启用
        "revised_opacity": 0.1,        # 分裂出的高斯不透明度（官方 revised_opacity）
        "prune_opacity": 0.005,
        # ⚠️ 不透明度重置：**默认关闭（0）**，这是实测结论。
        # 这里曾长期是 3000，而 trainer 的 dataclass 默认早已改为 0 —— **两份默认值不一致**，
        # 运行时以本文件为准，导致"我在 trainer 里关掉了重置"根本没生效：
        # 训练一直每 3000 步压一次不透明度，末次重置后的状态被存成最终检查点，
        # 实测 PLY 的不透明度中位数被压到 0.012（正常应 ~0.5），渲染出一张**近乎全白**的图。
        # 教训见 docs/03 G39：默认值只能有**一处**权威来源。
        "opacity_reset_interval": 0,
        # ---- 不透明度重置的**替代方案**（用户方案第 5 项）----
        # 官方周期重置是不连续跳变（硬钳 logit 到 logit(0.01)）⇒ 梯度同时变大
        # ⇒ 触发致密化失控（实测 279,534 高斯 vs 不重置 78,526），且污染末次检查点。
        # 下面两个替代都**不做跳变**，所以不引入那波梯度尖峰。
        #
        # 方案 C：梯度感知的不透明度衰减 —— 每 opacity_decay_interval 步把**实际**
        #   不透明度乘以 opacity_decay，让低贡献高斯平滑淡出。
        #   1.0 = 关闭（默认）；建议 0.999 ~ 0.9999。
        "opacity_decay": 1.0,
        "opacity_decay_interval": 50,
        # 方案 A/B：按**利用率**剪枝 —— 累积 |∂L/∂opacity|，周期性剪掉最低的
        #   `utilization_prune_quantile` 比例。只删长期无贡献的高斯，
        #   其余不动（不像重置那样影响全部高斯）。0 = 关闭（默认）。
        "utilization_prune_interval": 0,
        "utilization_prune_quantile": 0.2,
        # 剪枝**只在这个步数之前**发生（0 = 不额外限制）。
        # ⚠️ 实测教训：第一版按 interval=400 周期性剪 20%，8000 步内剪了约 20 次，
        #    0.8^20 ≈ 0.012 ⇒ 高斯 139,719 → **7,070**、PSNR → 21.04（比不剪低 2.1 dB）。
        #    用户方案的原意是"**每次重置后**剪一次"；本项目不用重置，
        #    于是它退化成"每 400 步都剪" —— **触发条件用错了，不是方案无效**。
        #    现在配合大 interval（4000）+ `*_until` 把剪枝限制在少数几次。
        "utilization_prune_until": 0,
        # ---- 方案 A：恢复感知剪枝（recovery-aware pruning）----
        # 判据是"**恢复速度**"：正常高斯在所有视角贡献为正，被衰减后很快被拉回；
        # 浮空物只在少数视角有贡献，不透明度震荡、**恢复慢** ⇒ 剪掉最慢的那批。
        # 用户原方案以"重置后 300 步"为窗口；本项目不用重置（实测 −0.59 dB），
        # 改用方案 C 的平滑衰减做打压，判据与剪枝对象保持原意。
        # 0 = 关闭（默认）。建议配合 opacity_decay 一起用。
        "recovery_prune_interval": 0,
        "recovery_prune_quantile": 0.2,
        # 快照后等多少步再判断"恢复了多少"（用户方案给 300）
        "recovery_window": 300,
        # 同 utilization_prune_until：限制剪枝只在少数几步发生
        "recovery_prune_until": 0,
        # ---- 当前选用的"不透明度治理方案"名称：none / reset / utilization / recovery / decay ----
        # 仅作**标记与界面显示**用；真正的行为由上面各开关决定（不做隐式开关联动）。
        "opacity_cleanup": "none",
        # 掩码内损失生效的迭代占比：1.0 = 全程约束在掩码内。
        # （同样是曾与 trainer 不一致的一项：这里 0.25、trainer 1.0。）
        "mask_loss_until": 1.0,
        # ---- 掩码损失的**口径**（本轮修正的算法错误）----
        # 旧行为（mask_mode="inside"）：`l1 = l1[mask].mean()`，只在掩码内取均值。
        #   两个后果：(1) loss 量级随前景占比漂移（实测各视角 12%~55% ⇒ 差 4.6 倍）；
        #   (2) **掩码外完全没有监督**，模型可在背景任意生成高斯 ⇒ 渲染"糊出去"、
        #       该空洞处被无约束高斯填成破碎网格。
        # 新行为（"composite"，默认）：把 GT 在掩码外**合成为已知背景色**，
        #   损失仍在**全图**上计算 —— 前景监督真实像素、背景监督"必须渲染成背景色"、
        #   每个视角 loss 尺度一致（与 3DGS / gsplat 官方示例同源）。
        # 保留 "inside" 仅用于复现旧结果/对照实验。
        "mask_loss_mode": "composite",
        # 正则项（参考 Faster-GS / 3DGS-MCMC 的做法）。
        # 开启后能压制"半透明浮点" —— 即点云/网格周围那圈雾状杂点。
        # 论文里配合 MCMC 密度控制使用 0.01；默认 0 表示不启用（保持既有行为不变，
        # 待实测确认收益后再考虑改默认）。
        "opacity_reg": 0.0,
        "scale_reg": 0.0,
        # 每步钳制 log 尺度上限 = max_scale_ratio × scene_scale（0 = 关闭）。
        # 存在的理由：`prune_scale3d` **只在致密化窗口内**执行，窗口结束后
        # 高斯可以无限拉长。实测（img3-final, 30k）：最长高斯 scale=[15.81, 0.0, 0.033]、
        # opacity=1.0，而模型只有 4.16 米宽；aniso>100 的高斯占数量 10.2% 却承担
        # **94.8% 的视觉权重** ⇒ 自由视角出现细丝、泊松网格皱成一团（docs/03 待编号）。
        "max_scale_ratio": 0.05,
        # 掩码外 alpha 正则（0=关闭）。
        # 为什么需要：composite 只约束**颜色**（该黑处渲黑），不约束
        # "别在那儿画东西"。实测掩码外 alpha 均值 0.70（应≈0）——
        # 模型外面裹着一层不透明黑几何，转视角就是"背面混乱"。
        "bg_alpha_reg": 0.1,
        # 长细比上限（0=关闭）。
        # max_scale_ratio 只限了**最长轴**，没限**最短轴**：实测仍有
        # scale=[0.0, 0.0064, 0.0177]、opacity=1.0、aniso=**7509** 的
        # "零厚度薄片" —— 转到侧面就是一道丝。
        # 这里把 smin 抬到 smax/max_aspect（log 域）。
        "max_aspect": 50.0,
        # ==== 学习率与官方对齐（2026-10-22）====
        # 官方 3DGS/gsplat 的 means 学习率是 **1.6e-4 * scene_scale**；
        # 我们一直用固定 1.6e-4 ⇒ 场景尺度约 5 时**位置几乎不动**，
        # 只能靠拉长/放大高斯去凑像素（针状高斯的根源之一）。
        "lr_means_scene_scale": False,
        # 官方 SH 学习率 = 2.5e-3 / 20 = **1.25e-4**；我们用 2.5e-3（大 20 倍）
        # ⇒ 颜色按视角过拟合，也是各类正则/感知损失测不出效果的可疑来源。
        "lr_shs": 0.0025,
        # ==== 消除“容量冻结 vs 重置”冲突（2026-10-22）====
        # 官方 reset_every=3000 与 refine_stop_iter=15000 是联动的：重置剪掉后靠 densify 补回。
        # 我们密度窗口只到 4000，重置就变成“净删 36% 容量”。
        # 开启后：发现重置间隔 > densify_until/2 时，自动把窗口
        # 延长到 min(iterations*0.6, reset*3)，并记一条日志。
        "auto_extend_densify_when_reset": True,
        # ==== 掩码腐蚀（修“白边”，2026-10-24）====
        # 实测：掩码普遍偏大，掩码内部靠边 4px 环的真值亮度 **0.89**，
        # 而掩码内部只有 0.43~0.51、掩码外 0.91 ⇒ 那一圈实际是**亮背景**。
        # 训练时把它当物体 ⇒ 学出轮廓高光（白边）。这里在**损失**上腐蚀
        # N 个像素（评估仍用原掩码，保证可比）。
        "mask_erode_px": 0,
        # ==== SH 正则（治斜视彩虹，2026-10-24）====
        # 实测：任意轨道视角下出现青/洋红/绿色块，而 sh_degree=0（仅 DC）完全消失
        # ⇒ 高阶 SH 在未见视角过拟合。这里给 _rest 加 L2 与幅度上限。
        "sh_rest_reg": 0.003,   # 默认 3e-3：彩度 −19%、SSIM 仅 −0.0005（docs/03 G137）
        "sh_rest_max": 0.0,
        # 收尾清理：训练结束时删掉不透明度低于此值的高斯。
        # Faster-GS 用 MIN_OPACITY_AFTER_TRAINING = 1/255 ≈ 0.0039。
        "min_opacity_final": 0.0,
        # 不透明度学习率。官方 3DGS 用 0.05，但那**配合每 3000 步的重置**；
        # 我们默认关闭重置，0.05 会让不透明度一路被压向 0、
        # 退化成"一大团极低不透明度的薄雾"（轮廓指标正常但深度失效 → 网格成碎片）。
        # Faster-GS / 3DGS-MCMC 用 0.025，这里对齐。
        "lr_opacities": 0.025,
        # ---- 学习率调度（用户方案第 1、2 项）----
        # cosine = 线性预热 + 余弦退火（默认）；exp = 旧的纯指数衰减
        "lr_schedule": "cosine",
        # 预热步数。默认 0 = 不预热，保持与历史结果可比；
        # 按迭代数的 5%~10% 设置（例如 20000 迭代 → 1000~2000）。
        "warmup_iters": 0,
        # 余弦退火下限（占峰值比例）；0.1 = 衰减到峰值的 1/10
        "lr_min_ratio": 0.1,
        # 四元数学习率。用户方案建议降到 5e-4，但**受控实测不支持**：
        # 同数据 20000 迭代，1e-3 → PSNR 24.264，5e-4 → 24.187（反而低 0.077 dB）。
        # 按"实测优先于直觉"的原则保持 1e-3；界面/CLI 仍可改。
        "lr_quats": 1e-3,
        # ---- 感知损失 LPIPS（用户方案第 3 项）----
        # L1+SSIM 偏平滑、对纹理细节（石雕纹样、瓦当）不敏感；LPIPS 用 VGG 深层特征度量。
        # 默认 0（关闭）：开启前必须有**同数据对照**证据。
        #
        # ⚠️ 两个必须记住的测量前提（我曾因忽略它们得出错误结论）：
        #   1) **噪声底**：同配置重复训练 PSNR 散布约 **0.09 dB**，
        #      小于它的差异不能作为"有效/无效"的依据；
        #   2) **渲染必须用完整 SH**：诊断脚本若用 `sh_degree=0`，会低估 1.91 dB
        #      并且抹掉**视角相关**外观 —— 那正是感知损失要改善的部分，
        #      用那种尺子测必然得到假阴性（见 gsdata/shio.py 的说明）。
        "lambda_lpips": 0.0,
        # 每 N 步计算一次 LPIPS（VGG 前向开销不小）
        "lpips_every": 4,
        # ---- 多指标评估（用户方案第 6 项）----
        # metrics.json 同时记录 PSNR / SSIM / LPIPS，并保存"最佳检查点"的完整指标。
        "eval_metrics": True,
        # 用哪个指标挑最佳检查点（导出阶段优先用它）：psnr | ssim | lpips
        "best_metric": "psnr",
        # ---- 2DGS 训练侧正则（参考 2D Gaussian Splatting, SIGGRAPH'24）----
        # 动机：3DGS 的累计深度是沿射线的**加权平均**，一个像素的深度由几十个高斯
        # 混合而成 —— 表面位置本身是模糊的。导出网格时"深度反投影"拿到的点云在法线
        # 方向上有厚度，泊松重建就是碎面；由深度梯度算的法线也不稳。
        #
        # ⚠️ **实测：直接套 2DGS 的权重在 3DGS 上是有害的，因此默认关闭（0）。**
        # 公平对照（img2-2dgs-base vs img2-2dgs-reg，同为 200 帧 / 12000 迭代 / 700px）：
        #   指标              无正则        正则 0.05/0.1
        #   PSNR              22.15 dB      18.18 dB      （−3.97 dB）
        #   高斯数            425,744       201,187
        #   scale 中位数      0.00334       0.01266       （大 3.8 倍，几何被撑开）
        #   网格相邻面法线     0.7426        0.2570
        #   法线翻转比例       6.5%          35.2%         （**更差 5.4 倍**）
        # 且两项正则的数值**都没有下降趋势**（depth 2.37e-2→2.79e-2；normal 0.40→0.30 震荡），
        # 说明权重过大、正则在与重建损失对抗，而不是在引导它。
        #
        # 为什么 2DGS 能用这套权重而我们不能（如实记录，避免后人重复）：
        #   2DGS 把高斯**压平成 2D 圆盘**并显式优化法线，所以"深度应集中""法线应一致"
        #   与它的参数化是自洽的；3DGS 的椭球没有这个约束，直接加这两项会与
        #   "拟合辐射场"的目标冲突。要真正走 2DGS 路线需要**改参数化**（压平 + 显式法线），
        #   而不是只加两个损失项。
        "lambda_depth_distortion": 0.0,
        "lambda_normal": 0.0,
        "reg_interval": 8,             # 每 N 步施加一次正则（深度正则需要额外一次渲染）
        "depth_distortion_mode": "tv",  # tv（一阶）| lap（二阶）
        "extra_args": "",
    },
    "export": {
        # 产物格式（逗号分隔）：
        #   glb_points  点云 glTF（纯 Python 写出，离线可用）
        #   mesh        三角网格：同时产出 model_mesh.obj / .ply / .glb
        #   viewer_html 单文件离线查看器
        #   spz / splat 需额外依赖；obj 为点云 OBJ
        # 默认给出用户要求的三种查看产物：点云(ply/glb) + 网格 + （可选）带贴图网格
        "formats": ["glb_points", "mesh", "viewer_html", "splat"],
        "glb_max_points": 800000,
        "glb_shrink": 1.0,
        "viewer_cdn": "",         # 留空则用本地静态查看器

        # ---- 网格提取（深度反投影 -> 泊松 -> 平滑 -> QEM 简化）----
        # 网格提取方法：
        #   poisson（社区标准流程：深度反投影 → 泊松 → 裁剪 → 平滑 → QEM）
        #   2dgs   （在 poisson 基础上加**多视角一致性投票**，见下面 consistency 段）
        #   tsdf   （体素融合，另一条独立路线）
        #
        # **默认取 2dgs**：公平对照（同模型 img2-2dgs-base、同 80,000 面、同 80 视角，
        # 唯一差别是投票开关）显示它在三项质量指标上更好：
        #   指标                 poisson     2dgs
        #   相邻面法线点积        0.7430      0.8143   （+9.6%）
        #   法线翻转比例           6.5%        3.4%    （减半）
        #   深度误差中位数         1.75%       1.58%   （改善）
        #   轮廓 IoU              0.671       0.650   （−0.021）
        # 即"用 0.021 的 IoU 换法线质量减半 + 深度更准"，而法线翻转正是
        # 用户反馈"网格杂乱、尖锐"的直接来源。
        # ⚠️ 注意：早期曾测得"2dgs 的 IoU 明显更低（0.684）"，那批数据是在
        # **背面剔除方向写反**（见 docs/03 G51 附近）时得到的，反投影点本身无效，已作废。
        "mesh_method": "2dgs",
        "mesh_width": 640,              # 渲深度用的分辨率
        "mesh_max_views": 80,           # 参与融合的视角上限
        # 泊松八叉树深度。**默认 8**（原为 9，按实测翻转率改）：
        #   同一模型、同一份输入点云，只改 depth，相邻面法线翻转率：
        #     dd-0 : depth 8 -> 1.7%   depth 9 -> 17.8%   depth 10 -> 43.5%
        #     dd-15: depth 8 -> 2.1%   depth 9 -> 22.4%
        #   面数几乎不变（7.6~7.9 万），轮廓 IoU 只掉 0.014（0.793 -> 0.779）。
        #   ⇒ **更深不等于更好**：depth=10 反而把法线翻转推到 43.5%（过拟合到噪声点）。
        #   机理：翻转来自几何自交（该网格 `is_orientable()=False`，
        #   且 `orient_triangles()` 完全无效 ⇒ 不是可修的绕序），而更浅的八叉树
        #   产出**更平滑**的表面，自交更少。这与"靠投票删面"降翻转不同（那个 IoU 掉 0.060）。
        #   代价提醒：depth=10 在大规模输入点（530 万）下还会让平滑阶段跑 1 小时以上（见 03/G46）。
        "mesh_poisson_depth": 8,
        # 自动聚焦：网格重建的包围盒**只用不透明度过滤，不按分位数裁**。
        #
        # ⚠️ 这里原来是 `0.95`，是一个**真实事故**的源头（用户："网络依然破碎"）：
        # 3DGS 把容量放在**细节密集**处（门楼雕花），屋檐/翼角面积大但纹理平滑
        # ⇒ 分到的高斯很少。按"点的分位数"裁，会先把稀疏但真实的结构裁掉：
        #
        #     不透明度>0.3 的高斯 24,629 个，铺满 5.83 × 2.16 × 2.85
        #     但 95% 分位框只有 1.67 × 1.05 × 0.49
        #     最终网格 bbox 只有高斯的 25% / 22% / 15%，体积占 0.85%
        #
        # ⇒ 导出的网格只是中央一小块扭曲碎片。改成 1.0（等价于不做分位数裁剪），
        # 背景污染由 `mesh_crop_opacity` 的低不透明度过滤负责。
        # 详见 `gsdata/mesh_export.auto_crop_bounds` 与 docs/03 G119。
        "mesh_crop_quantile": 1.0,
        "mesh_density_quantile": 0.001, # 密度裁剪；**极敏感**，0.03 会把质量砍半
        "mesh_trim_sigma": 2.5,         # 按"到输入点距离"裁掉泊松桥接面；
                                        # 阈值 = 输入点间距 × 该值。0=关闭。
                                        # 这一项专治"该空的地方是实的"
        "mesh_target_triangles": 400000,  # QEM 简化目标面数（0=不简化）
        # 预简化倍数：平滑**不在原始高密度网格上做**，先 QEM 降到 target×ratio 再平滑。
        # 为什么：Taubin 平滑是 CPU 上对全体顶点与邻接的迭代，代价随规模超线性增长。
        # 实测泊松 depth=10 产出 2771 万面时"先平滑"跑了 61 分钟仍未完成（内存 7.9GB），
        # 而先预简化可把规模界定在目标面数量级。设为 1.0 = 关闭预简化。
        "mesh_presimplify_ratio": 10.0,
        "mesh_smooth_iters": 5,         # Taubin 平滑迭代
        "mesh_largest_only": True,      # 只保留最大连通块

        # ---- 纹理烘焙（可选，"实感"的最后一步）----
        # ⚠️ 依赖 xatlas UV 展开，而它在大网格上可能极慢（实测 10 万面级别会卡住），
        #    所以默认关闭 + 带超时保护（超时即跳过，不影响其它产物）。
        "mesh_texture": False,
        "mesh_texture_size": 2048,
        "mesh_texture_timeout": 300,    # UV 展开超时（秒）；超时跳过纹理烘焙
        # 拒绝采到"黑背景"的像素。`mask.mode=black` 的训练图背景是纯黑，
        # 网格轮廓边缘的采样点会采到 (0,0,0) ⇒ 纹理黑斑
        # （实测 final-v1 黑色 texel 35.99%）。转台模式（mask.mode=keep）本来就没有背景，
        # 该选项对它是无害的空操作。若素材本身是深色物体、担心误杀，可设为 False。
        "mesh_texture_reject_black": True,
        # 按"到输入点距离"裁掉泊松桥接出来的面（阈值 = 输入点间距 × 该值；0=关闭）
        "mesh_trim_sigma": 2.5,
        # 删除"从未被任何相机看到"的网格面（泊松重建的内部垃圾几何）。
        # 实测 img3-crop：**25.6% 的面从未被看到**，且散布在整个物体内部
        # （包围盒 ≈ 全网格、到中心距离 ≈ 全网格），不是背面遮挡。
        # 它们白占 UV 空间却烘不出颜色 ⇒ 纹理黑洞（实测 43.7% chart 内部为黑）。
        # ⚠️ 判据是"是否被相机看到过"，不是"落在前景掩码内的比例"——
        # 后者在转台素材上原理错误，我曾据此误删物体细节（docs/03 G96）。
        "mesh_drop_unseen": True,
        "mesh_min_views": 1,           # 保留"被至少这么多视角看到"的面

        # ---- 「2dgs」模式的多视角一致性投票（仅 mesh_method=2dgs 时生效）----
        # 为什么需要：单视角深度反投影会把"只有 1 个视角看到的伪影"也变成点，
        # 表现为表面毛刺、以及**法线翻转 ~35%**（同一表面被相邻视角各投一份、
        # 两份法线方向相反，体素下采样后平均成接近零，光照大片发黑）。
        # 判据：体素投票，要求 >= min_views 个视角支持。
        "mesh_consistency_min_views": 2,       # 至少多少个视角支持该体素
        # 投票体素 = 下采样体素 × 该值。**必须比下采样体素大**，否则相邻视角对同一
        # 表面的采样永远落不进同一格，投票会误删真实表面（实测 1.5 时删掉 52% 的点、
        # IoU 从 0.840 掉到 0.684）。
        # 实测（img2-opfix，80 视角，模型与其它参数完全相同）：
        #   1.5 -> 保留 48.0%，法线一致 0.5865，IoU 0.684
        #   3.0 -> 保留 84.0%，法线一致 0.3041，IoU 0.815   <- 默认，性价比最好
        #   对照 poisson（不投票）：法线一致 0.1019，IoU 0.840
        "mesh_consistency_voxel_scale": 3.0,
        "mesh_consistency_normal_dot": 0.0,    # 与该体素平均法线的点积下限（-1=不筛）
    },
    "metrics": {
        "enabled": True,
        "num_views": 8,
    },
    "web": {
        "host": "127.0.0.1",
        "port": 8787,
        "open_browser": False,
    },
    "runtime": {
        "max_parallel_jobs": 1,   # 单卡串行，避免抢显存
        "keep_intermediate": True,
        "log_tail_lines": 400,
    },
}

_SETTABLE_PREFIXES = tuple(DEFAULTS.keys())


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:  # pragma: no cover
        raise ConfigError(f"配置文件 {path} 解析失败：{exc}") from exc


def _dump_toml(data: dict[str, Any]) -> str:
    """极简 TOML 写出器：只支持本项目用到的 表 + 标量 + 字符串数组。"""
    lines: list[str] = []

    def fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, str):
            return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(v, list):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        raise ConfigError(f"不支持写出的 TOML 类型：{type(v)!r}")

    for section, values in data.items():
        if not isinstance(values, dict):
            continue
        lines.append(f"[{section}]")
        for k, v in values.items():
            if isinstance(v, dict):
                continue
            lines.append(f"{k} = {fmt(v)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class Settings:
    """配置容器：支持点路径读写，例如 s.get('colmap.max_num_features')。"""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    # --- 读取 ---
    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self._data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def section(self, name: str) -> dict[str, Any]:
        val = self._data.get(name, {})
        return copy.deepcopy(val) if isinstance(val, dict) else {}

    # --- 写入 ---
    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        if parts[0] not in _SETTABLE_PREFIXES:
            raise ConfigError(
                f"未知配置段 '{parts[0]}'，可用：{', '.join(_SETTABLE_PREFIXES)}"
            )
        cur = self._data
        for part in parts[:-1]:
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
        cur[parts[-1]] = _coerce(value)

    def save(self, path: Path | None = None) -> Path:
        target = path or CONFIG_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_dump_toml(self._data), encoding="utf-8")
        return target


def _coerce(value: Any) -> Any:
    """把 CLI/GUI 传进来的字符串按字面量解释。"""
    if not isinstance(value, str):
        return value
    low = value.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null", ""):
        return ""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_coerce(x.strip().strip("'\"")) for x in inner.split(",")]
    return value


def load(overrides: dict[str, Any] | None = None, config_path: Path | None = None) -> Settings:
    """按优先级合并出最终配置。"""
    data = _deep_merge(DEFAULTS, load_toml(config_path or CONFIG_FILE))
    settings = Settings(data)
    for k, v in (overrides or {}).items():
        settings.set(k, v)
    return settings


def write_default_config(path: Path | None = None) -> Path:
    """把默认配置落盘（首次初始化 / GUI 的“恢复默认”）。"""
    target = path or CONFIG_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_dump_toml(DEFAULTS), encoding="utf-8")
    return target
