"""路径与目录约定。

设计原则：本项目所有宿主路径集中在这里解析，其它模块只拿 Path，
不自己拼字符串；这样 Windows 上换盘或改工作区只需要改一处。

真实目录都在 F:\\3dgs-and-ai 下（DSH 沙箱授权根），
D:\\3dgs-workspace 是指向这些真实目录的目录链接（junction），
方便用户在 D 盘直接访问。
"""

from __future__ import annotations

import os
from pathlib import Path

# --- 项目根（src/gsflow/paths.py -> 上溯三级） ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- 顶层目录 ---
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIG_DIR = PROJECT_ROOT / "config"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

ENV_DIR = PROJECT_ROOT / "env"          # venv / 缓存 / 临时目录
TOOLS_DIR = PROJECT_ROOT / "tools"      # 自带的 colmap / ffmpeg 等
MODELS_DIR = PROJECT_ROOT / "models"    # 模型权重
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"  # 每个作业一个子目录
JOBS_DIR = WORKSPACE_DIR
LOGS_DIR = PROJECT_ROOT / "logs"

# --- 环境变量（子进程继承；沙箱下把临时目录与缓存都关进工作区） ---
VENV_DIR = ENV_DIR / "venv"
TMP_DIR = ENV_DIR / "tmp"
UV_CACHE_DIR = ENV_DIR / "uv-cache"
PIP_CACHE_DIR = ENV_DIR / "pip-cache"
HF_HOME = MODELS_DIR / "hf"
# 第三方库的配置/缓存目录（必须关进工作区，见 D1 踩坑）
YOLO_CONFIG_DIR = ENV_DIR / "config" / "ultralytics"
MPL_CONFIG_DIR = ENV_DIR / "config" / "matplotlib"
# rembg 默认用 Path.home()/".rembg" 存 ONNX 权重；沙箱下 C:\Users\<user> 不可写
REMBG_HOME = MODELS_DIR / "rembg"
# torch 的 JIT 扩展编译目录（默认在 %LOCALAPPDATA%\torch\Extensions）
TORCH_EXTENSIONS_DIR = ENV_DIR / "torch_extensions"
# 免安装（便携）CUDA：tools/cuda 下解开 cuda_nvcc + cudart + cccl，不需要管理员安装
PORTABLE_CUDA_DIR = TOOLS_DIR / "cuda"

ALL_DIRS = [
    DOCS_DIR, CONFIG_DIR, SCRIPTS_DIR, TESTS_DIR,
    ENV_DIR, TOOLS_DIR, MODELS_DIR, DOWNLOADS_DIR,
    WORKSPACE_DIR, LOGS_DIR, TMP_DIR, UV_CACHE_DIR, PIP_CACHE_DIR, HF_HOME,
    YOLO_CONFIG_DIR, MPL_CONFIG_DIR, REMBG_HOME, TORCH_EXTENSIONS_DIR,
]


def _patch_rembg_home() -> None:
    """把 rembg 的模型目录指到工作区。

    rembg 在模块导入时执行 `os.environ["HOME"] = str(Path.home())` 之类逻辑，
    并在 `Path.home()/".rembg"` 下建缓存；沙箱环境下 C:\\Users\\<user> 不可写，
    会直接 `[WinError 5] 拒绝访问`。这里做三重保险：
      1. 直接设置 rembg 模块的 home 属性（若已导入）；
      2. 设置 REMBG_HOME / U2NET_HOME 环境变量（部分版本支持）；
      3. 兜底把 HOME/USERPROFILE 指到工作区（pooch/platformdirs 会跟随）。
    """
    import os as _os

    target = str(REMBG_HOME)
    _os.environ.setdefault("REMBG_HOME", target)
    _os.environ.setdefault("U2NET_HOME", target)
    try:  # 已导入的情况
        import sys as _sys

        mod = _sys.modules.get("rembg")
        if mod is not None and hasattr(mod, "home"):
            from pathlib import Path as _P

            mod.home = _P(target)
    except Exception:
        pass


def ensure_dirs() -> None:
    """幂等地创建全部标准目录。"""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def runtime_env() -> dict[str, str]:
    """会影响**当前进程**第三方库行为的变量（在 CLI/GUI 入口尽早调用 apply_runtime_env）。"""
    env = {
        "TEMP": str(TMP_DIR),
        "TMP": str(TMP_DIR),
        "UV_CACHE_DIR": str(UV_CACHE_DIR),
        "PIP_CACHE_DIR": str(PIP_CACHE_DIR),
        "HF_HOME": str(HF_HOME),
        "YOLO_CONFIG_DIR": str(YOLO_CONFIG_DIR),   # ultralytics 默认想写 AppData\Roaming
        "MPLCONFIGDIR": str(MPL_CONFIG_DIR),       # matplotlib 字体缓存
        "TORCH_EXTENSIONS_DIR": str(TORCH_EXTENSIONS_DIR),  # torch JIT 编译输出
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    if find_portable_cuda() is not None:
        env["CUDA_HOME"] = str(PORTABLE_CUDA_DIR)
        env["CUDA_PATH"] = str(PORTABLE_CUDA_DIR)
        env["TORCH_CUDA_ARCH_LIST"] = "12.0"
    return env


def find_portable_cuda() -> Path | None:
    """查找可用的 CUDA（优先免安装的便携版，其次系统安装版）。

    便携版来自官方安装包里的子包（cuda_nvcc + cuda_cudart + cuda_cccl），
    用 Windows 自带的 bsdtar 解开即可，**不需要管理员权限**。
    """
    if (PORTABLE_CUDA_DIR / "bin" / "nvcc.exe").exists():
        return PORTABLE_CUDA_DIR
    for root in (
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    ):
        base = Path(root)
        if base.exists():
            for d in sorted(base.iterdir(), reverse=True):
                if (d / "bin" / "nvcc.exe").exists():
                    return d
    return None


def _patch_torch_subprocess_decode() -> None:
    """修正 torch 解析编译器输出时的解码方式。

    实测踩坑：中文 Windows 上 cl.exe 输出 UTF-8（代码页 65001），
    而 `torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS` 默认是 `('oem',)`
    （英文系统对应 cp437），于是 torch 解析编译器版本时直接
    `UnicodeDecodeError: 'cp1' codec can't decode ...`，
    表现为 gsplat 编译内核失败、`_C = None`、最后报一个莫名其妙的
    `'NoneType' object has no attribute 'CameraModelType'`。
    """
    try:
        import torch.utils.cpp_extension as ce

        if ce.SUBPROCESS_DECODE_ARGS != ("utf-8",):
            ce.SUBPROCESS_DECODE_ARGS = ("utf-8",)
    except Exception:
        pass


def apply_runtime_env() -> None:
    """把 runtime_env 应用到当前进程（必须在 import ultralytics / rembg / matplotlib 之前调用）。"""
    import os as _os

    ensure_dirs()
    for k, v in runtime_env().items():
        _os.environ.setdefault(k, v)
    _patch_rembg_home()
    _patch_torch_subprocess_decode()


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """为外部子进程（ffmpeg / colmap / python -m ...）准备环境变量。

    关键点（踩坑记录见 docs/03）：
      * TEMP/TMP 必须指到工作区内，否则沙箱环境下 tempfile/ensurepip 会因为
        无法 chmod 沙箱临时目录而 PermissionError；
      * ultralytics 默认写 AppData\\Roaming（被拒），要指到 YOLO_CONFIG_DIR；
      * 各缓存目录一并关进工作区，避免写到 C 盘被沙箱拒绝。
    """
    ensure_dirs()
    env = dict(os.environ)
    env.update(runtime_env())
    if extra:
        env.update(extra)
    return env


def job_dir(job_id: str) -> Path:
    return WORKSPACE_DIR / job_id


def job_stage_dir(job_id: str, stage: str) -> Path:
    return job_dir(job_id) / stage
