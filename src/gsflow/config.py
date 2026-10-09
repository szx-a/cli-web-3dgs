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
        "densify_quantile": 0.75,      # 取梯度范数的 (1-q) 分位作为自适应阈值
        "densify_grad_threshold": 1.5e-4,
        "densify_grad_max": 5e-3,
        # 以下三个比例都相对 scene_scale（相机中心平均距离），取自官方 gsplat DefaultStrategy。
        # prune_scale3d 是**防发散的关键约束**：漏掉它会让尺度无界增长（实测 log尺度到 739）。
        "grow_scale3d": 0.01,          # 尺度 <= 0.01*scene_scale 走克隆，大于走分裂
        "prune_scale3d": 0.1,          # 尺度 > 0.1*scene_scale 直接裁剪
        "prune_scale_after_reset": True,  # 尺度裁剪只在 step > opacity_reset_interval 后启用
        "revised_opacity": 0.1,        # 分裂出的高斯不透明度（官方 revised_opacity）
        "prune_opacity": 0.005,
        "opacity_reset_interval": 3000,
        "mask_loss_until": 0.25,       # 前 25% 迭代只在掩码内算损失
        "extra_args": "",
    },
    "export": {
        # glb_points（纯 Python，离线可用）| spz | splat | viewer_html
        "formats": ["glb_points", "viewer_html"],
        "glb_max_points": 800000,
        "glb_shrink": 1.0,
        "viewer_cdn": "",         # 留空则用本地静态查看器
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
