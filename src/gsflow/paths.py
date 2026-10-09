"""路径与目录约定。

设计原则：本项目所有宿主路径集中在这里解析，其它模块只拿 Path，
不自己拼字符串；这样 Windows 上换盘或改工作区只需要改一处。

真实目录都在 F:\\3dgs-and-ai 下（DSH 沙箱授权根），
D:\\3dgs-workspace 是指向这些真实目录的目录链接（junction），
方便用户在 D 盘直接访问。
"""

from __future__ import annotations

import os
import sys
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


def _ninja_unusable() -> bool:
    """检测 ninja 在当前环境是否能正常结束。

    实测踩坑：本机沙箱下 ninja（官方版与 MinGW 版都试过）**能完成任务但不退出进程**，
    而 torch 的 cpp_extension 是同步等待 ninja 结束的，于是 JIT 编译永远卡住
    （表现为 ninja 进程 CPU=0、构建目录只有 build.ninja、日志为空）。
    """
    import shutil as _shutil
    import subprocess as _subprocess
    import tempfile as _tempfile

    exe = _shutil.which("ninja")
    if not exe:
        return True  # 没有 ninja，反正用不了
    try:
        with _tempfile.TemporaryDirectory(dir=str(TMP_DIR)) as d:
            nj = Path(d) / "build.ninja"
            nj.write_text("rule r\n  command = cmd /c echo ok > $out\n\nbuild o.txt: r\n", encoding="ascii")
            p = _subprocess.Popen([exe, "-C", d], stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL,
                                  env=child_env())
            try:
                p.wait(timeout=20)
                return False  # 正常退出 -> ninja 可用
            except _subprocess.TimeoutExpired:
                try:
                    p.kill()
                except Exception:
                    pass
                return True
    except Exception:
        return True


def _patch_torch_ninja_bypass() -> None:
    """把 torch 的 ninja 构建路径替换成"直接调编译器"的构建器。

    只在检测到 ninja 不可用时启用（结果会缓存），避免影响正常环境。
    """
    import os as _os

    if _os.environ.get("GSFLOW_FORCE_NINJA", "") == "1":
        return
    if _os.environ.get("GSFLOW_NINJA_BYPASS") == "1":
        return
    # 只在真的要用 torch 编译扩展时才做这个检测：它要启动一次 ninja 并等最多 20 秒
    if "torch.utils.cpp_extension" not in sys.modules:
        return
    if not _ninja_unusable():
        return
    try:
        from . import build as _build

        if _build.install_bypass():
            _os.environ["GSFLOW_NINJA_BYPASS"] = "1"
    except Exception:
        pass


def find_vcvars64() -> Path | None:
    """定位 MSVC 的 `vcvars64.bat`（cl.exe 必需）。

    gsplat 的 CUDA 内核是 JIT 编译的：**光有 nvcc 不够，还要 cl.exe**。
    没有 cl.exe 时 torch 会报 "Cannot find compiler 'cl.exe' in PATH"。
    """
    for p in (
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    ):
        f = Path(p)
        if f.exists():
            return f
    return None


def _msvc_env_done() -> bool:
    """本进程是否已经装载过 MSVC 环境（避免重复跑 vcvars64.bat）。

    为什么要这个标记：`load_msvc_env()` 要起一个 cmd 子进程跑 vcvars64.bat
    （实测单次 ~7 秒）。而 `apply_runtime_env()` 会被各入口反复调用
    （测试套件里每个用例都会走一遍），没有这个短路的话
    **测试从 3 秒变成 103 秒**（实测踩到）。
    """
    return bool(os.environ.get("VSCMD_ARG_TGT_ARCH") or os.environ.get("GSFLOW_MSVC_READY"))


_MSVC_ENV_CACHE: dict[str, str] | None = None


def load_msvc_env() -> dict[str, str]:
    """解析 `vcvars64.bat` 导出的环境变量（在 Python 内完成，不依赖 cmd 包装）。

    为什么要这样做：CLI 侧一直靠 `scripts\\gsflow_gpu.cmd` 先 `call vcvars64.bat`
    再启动 Python，所以训练/导出都能编译内核；但 **Web GUI 是从 `scripts\\web.cmd`
    启动的，没有这层包装**，于是同一份流程在 GUI 里跑就报
    `'NoneType' object has no attribute 'CameraModelType'`（gsplat 内核没编译成）。

    "能不能用"不该取决于用户从哪个脚本启动。所以在 Python 内直接解析
    vcvars64.bat 的输出（`set` 列表），把变量装进当前进程，
    CLI / GUI / 测试三条入口就都对等了。
    """
    global _MSVC_ENV_CACHE
    if _MSVC_ENV_CACHE is not None:
        return _MSVC_ENV_CACHE
    vcvars = find_vcvars64()
    if vcvars is None:
        _MSVC_ENV_CACHE = {}
        return _MSVC_ENV_CACHE
    import subprocess

    # 引号坑（踩了好几轮，这里记清楚）：
    #   `subprocess.run(["cmd","/c", f'call "{p}" && set'])` 在本机拿不到输出 ——
    #   含空格的绝对路径 + `call` + 重定向 + `&&` 叠在一起时，
    #   引号会被各层（PowerShell / cmd / subprocess 的 list2cmdline）反复转义，
    #   表现为 cmd 报 `'\"C:\\Program...' is not recognized` 或直接静默 rc=1。
    #
    # 最稳的做法：**生成一个临时 .bat**，把 vcvars 与 `set` 都写进文件里再执行。
    # 文件里不需要任何转义，也没有多层引号问题。
    try:
        tmp = TMP_DIR / "_vcvars_dump.bat"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            "@echo off\r\n"
            f'call "{vcvars}" >nul 2>&1\r\n'
            "set\r\n",
            encoding="ascii",
        )
        out = subprocess.run(
            [str(tmp)], capture_output=True, text=True, errors="ignore", timeout=180,
        ).stdout or ""
    except Exception:  # noqa: BLE001
        _MSVC_ENV_CACHE = {}
        return _MSVC_ENV_CACHE

    env: dict[str, str] = {}
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if not k:
            continue
        # ⚠️ Windows 坑（踩过）：`set` 输出里路径变量名是 **`Path=`**（首字母大写），
        # 而**环境变量名在 Windows 上不区分大小写**。
        # 我一开始用 `k.upper() == k`（要求全大写）过滤，
        # 结果把 `Path` 与 `VCToolsInstallDir` 这类含小写的名字全丢了 ——
        # 表现为 INCLUDE/LIB 都对、唯独 cl.exe 不在 PATH，很难看出原因。
        # 正确做法：统一**转大写**当作键（os.environ 在 Windows 上大小写不敏感，
        # 赋值给 PATH 与给 Path 是等价的）。
        if not (k[0].isalpha() or k[0] == "_"):
            continue
        env[k.upper()] = v
    _MSVC_ENV_CACHE = env
    return env


def _preload_gsplat_kernel() -> bool:
    """把已编译的 gsplat CUDA 内核加载成模块 `gsplat_cuda`，返回是否成功。

    这是本环境能"秒级用上 gsplat"的关键，原因链条（都是实测踩出来的）：

      1. gsplat 的 `cuda/_backend.py` 里 `load_extension()` **总是**调用
         `_jit_compile()`，它指望 ninja 做增量构建来省时间。
      2. 但本环境 **ninja 不退出**，我们换成了自己的构建器
         （`build.py`，解析 build.ninja 直接调 cl/nvcc/link）。
      3. 于是每次 `import gsplat` 都会走一次"重新链接"：
         实测 80~140 秒，链接日志就是 `link.exe ... /out:gsplat_cuda.pyd`。
      4. 更糟的是：如果我们**已经把这个 .pyd 加载进内存**（比如为了预加载），
         链接器就无法覆盖它，报
         `LINK : fatal error LNK1104: 无法打开文件"gsplat_cuda.pyd"`。
         —— 这正是"先加载再让 gsplat 自己加载"这条错误路线的死法。

    正确做法：**在 gsplat 之前**把 `.pyd` 加载好并注册成 `sys.modules["gsplat_cuda"]`，
    这样 `_backend.py` 里 `from gsplat import csrc as _C` 直接命中缓存，
    `except ImportError` 分支**根本不会被执行**，也就没有编译/链接。

    两个必须注意的点：
      * 模块名必须是 **`gsplat_cuda`** —— `.pyd` 导出的是 `PyInit_gsplat_cuda`，
        换成别的名字会报
        `dynamic module does not define module export function (PyInit_xxx)`。
      * 必须在 `import torch` 之后加载 —— 它依赖 torch 的 DLL，
        否则报 `DLL load failed ... 找不到指定的模块`。
      * 还要**同时注册成 `gsplat.csrc`**。因为 `_backend.py` 的第一条路径是
        `from gsplat import csrc as _C`，成功才会跳过 JIT；
        只注册 `gsplat_cuda` 时这条导入仍然失败，于是照样进编译分支
        （表现为"import gsplat 很快，但第一次调用 rasterization 就开始重编"）。
    """
    try:
        import importlib.util
        import sys as _sys

        ext_root = TORCH_EXTENSIONS_DIR / "gsplat_cuda"
        pyd = next((p for p in ext_root.glob("gsplat_cuda*.pyd") if p.stat().st_size > 0), None)
        if pyd is None:
            return False
        mod = _sys.modules.get("gsplat_cuda")
        if mod is None:
            spec = importlib.util.spec_from_file_location("gsplat_cuda", str(pyd))
            if spec is None or spec.loader is None:
                return False
            mod = importlib.util.module_from_spec(spec)
            _sys.modules["gsplat_cuda"] = mod
            spec.loader.exec_module(mod)
        # 关键：让 `from gsplat import csrc as _C` 命中，从而跳过 JIT 分支
        _sys.modules.setdefault("gsplat.csrc", mod)
        return True
    except Exception:  # noqa: BLE001
        # 失败就退回"让 gsplat 自己处理"，不影响其它功能
        return False


def _patch_gsplat_win_cache() -> None:
    """清掉 gsplat 构建目录里的残留锁，并尝试**预加载**已编译的内核。

    顺序很重要：必须在 `_patch_torch_ninja_bypass()` **之前**预加载。
    否则 torch 已经决定要重编，而库又被我们加载进内存 ->
    链接器覆盖失败（`LNK1104`）。详见 `_preload_gsplat_kernel()` 的说明。
    """
    try:
        ext_root = TORCH_EXTENSIONS_DIR / "gsplat_cuda"
        if not ext_root.exists():
            return
        for lock in ("lock", ".ninja_lock"):
            lp = ext_root / lock
            if lp.exists():
                try:
                    lp.unlink()
                except OSError:
                    pass
        _preload_gsplat_kernel()
    except Exception:  # noqa: BLE001
        pass


def apply_runtime_env() -> None:
    """把 runtime_env 应用到当前进程（必须在 import ultralytics / rembg / matplotlib 之前调用）。

    同时把**便携 CUDA 与 MSVC 环境**装进当前进程，这样无论从 CLI、GUI 还是测试进入，
    需要 nvcc/cl.exe 的功能（gsplat JIT 内核编译）都能正常工作 ——
    早期只有 `scripts\\gsflow_gpu.cmd` 做了这件事，导致 GUI 侧同一份流程失败。
    """
    import os as _os

    ensure_dirs()
    for k, v in runtime_env().items():
        _os.environ.setdefault(k, v)

    # 便携 CUDA：把 bin 加进 PATH（只在还没加过时加一次）
    cuda = find_portable_cuda()
    if cuda is not None:
        _os.environ["CUDA_HOME"] = str(cuda)
        _os.environ["CUDA_PATH"] = str(cuda)
        _os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        cuda_bin = str(cuda / "bin")
        if cuda_bin.lower() not in _os.environ.get("PATH", "").lower():
            _os.environ["PATH"] = cuda_bin + _os.pathsep + _os.environ.get("PATH", "")

    # MSVC：补齐 cl.exe / link.exe 需要的 include、lib、PATH。
    # 用了短路标记：跑一次 vcvars64.bat 约 7 秒，反复调用会把测试拖慢几十倍。
    if not _msvc_env_done():
        msvc = load_msvc_env()
        for k, v in msvc.items():
            _os.environ[k] = v
        _os.environ["GSFLOW_MSVC_READY"] = "1"

    _patch_rembg_home()
    _patch_torch_subprocess_decode()
    # ⚠️ 顺序不能变：预加载内核必须在 `_patch_torch_ninja_bypass()` 之前。
    # 先预加载 => gsplat 走 `from gsplat import csrc` 命中缓存，不进编译分支；
    # 若反过来，torch 已决定重编，链接器又无法覆盖已加载的 .pyd，报 LNK1104。
    _patch_gsplat_win_cache()
    _patch_torch_ninja_bypass()


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
