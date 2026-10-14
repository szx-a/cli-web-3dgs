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
    "frames": {
        # 抽帧模式：fps（定帧率）或 count（按目标张数均匀抽）
        "mode": "fps",
        "fps": 2.0,
        "count": 120,
        # 缩放长边；0 表示不缩放
        "long_side": 1920,
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
        # 清理模式：black（背景填黑，COLMAP 最稳）| inpaint（纹理补全）
        "mode": "black",
        "dilate_px": 4,
        "feather_px": 2,
        "min_area_ratio": 0.002,  # 小于此面积占比的零散区域丢弃
        "keep_largest": True,     # 只保留最大连通前景
        "batch_size": 4,
        "mask_black_threshold": 8,
        # 分割失败（未检出前景）时：keep=原图直通（下游仍可用）| skip=丢弃该帧
        "on_failure": "keep",
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
        "densify_grad_threshold": 2e-4,
        "densify_grad_max": 5e-3,
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
        # 掩码内损失生效的迭代占比：1.0 = 全程约束在掩码内。
        # （同样是曾与 trainer 不一致的一项：这里 0.25、trainer 1.0。）
        "mask_loss_until": 1.0,
        # 正则项（参考 Faster-GS / 3DGS-MCMC 的做法）。
        # 开启后能压制"半透明浮点" —— 即点云/网格周围那圈雾状杂点。
        # 论文里配合 MCMC 密度控制使用 0.01；默认 0 表示不启用（保持既有行为不变，
        # 待实测确认收益后再考虑改默认）。
        "opacity_reg": 0.0,
        "scale_reg": 0.0,
        # 收尾清理：训练结束时删掉不透明度低于此值的高斯。
        # Faster-GS 用 MIN_OPACITY_AFTER_TRAINING = 1/255 ≈ 0.0039。
        "min_opacity_final": 0.0,
        # 不透明度学习率。官方 3DGS 用 0.05，但那**配合每 3000 步的重置**；
        # 我们默认关闭重置，0.05 会让不透明度一路被压向 0、
        # 退化成"一大团极低不透明度的薄雾"（轮廓指标正常但深度失效 → 网格成碎片）。
        # Faster-GS / 3DGS-MCMC 用 0.025，这里对齐。
        "lr_opacities": 0.025,
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
        "formats": ["glb_points", "mesh", "viewer_html"],
        "glb_max_points": 800000,
        "glb_shrink": 1.0,
        "viewer_cdn": "",         # 留空则用本地静态查看器

        # ---- 网格提取（深度反投影 -> 泊松 -> 平滑 -> QEM 简化）----
        "mesh_method": "poisson",       # poisson（默认，社区标准流程）| 2dgs | tsdf
        "mesh_width": 640,              # 渲深度用的分辨率
        "mesh_max_views": 80,           # 参与融合的视角上限
        "mesh_poisson_depth": 9,        # 泊松八叉树深度：8 快 / 9 均衡 / 10 更细更慢
        "mesh_crop_quantile": 0.95,     # 自动聚焦到主体（0.90 偏紧、0.95 实测更好）
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
        # 按"到输入点距离"裁掉泊松桥接出来的面（阈值 = 输入点间距 × 该值；0=关闭）
        "mesh_trim_sigma": 2.5,

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
