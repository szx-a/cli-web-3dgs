"""外部工具定位：配置指定 -> 项目自带 tools/ -> PATH。

设计意图：不假设用户装了什么，也不要求管理员权限；找不到时给出
“可以直接照做的”安装建议，而不是抛一堆栈。
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from ..errors import ToolMissingError

# 逻辑名 -> 候选可执行文件名（含自带目录里的常见相对路径）
CANDIDATES: dict[str, list[str]] = {
    "ffmpeg": [
        "ffmpeg.exe",
        "ffmpeg/bin/ffmpeg.exe",
        "ffmpeg-*/bin/ffmpeg.exe",
    ],
    "ffprobe": [
        "ffprobe.exe",
        "ffmpeg/bin/ffprobe.exe",
        "ffmpeg-*/bin/ffprobe.exe",
    ],
    "colmap": [
        "colmap.exe",
        "bin/colmap.exe",
        "COLMAP/colmap.exe",
        "colmap/COLMAP.bat",
        "colmap-*/bin/colmap.exe",
        "colmap-*/COLMAP.bat",
        "COLMAP.bat",
    ],
    "magick": ["magick.exe", "ImageMagick*/magick.exe"],
}

ADVICE: dict[str, str] = {
    "ffmpeg": (
        "跑 `scripts\\download.cmd ffmpeg` 自动下载解压到 F:\\3dgs-and-ai\\tools\\ffmpeg\\，"
        "也可以自己放好，然后在 config/config.toml 的 tools.ffmpeg 填绝对路径。"
    ),
    "ffprobe": "与 ffmpeg 同包，解压后一并出现在 tools\\ffmpeg\\bin\\ 下。",
    "colmap": (
        "跑 `scripts\\download.cmd colmap` 自动下载 COLMAP 3.12 Windows CUDA 版到 tools\\colmap\\，"
        "也可以在 config/config.toml 的 tools.colmap 填 colmap.exe / COLMAP.bat 的绝对路径。"
    ),
    "magick": "可选：无需可跳过，仅在需要 ImageMagick 批量转码时才要。",
}


@dataclass
class ToolInfo:
    name: str
    path: str
    version: str = ""
    source: str = ""   # config | bundled | path


def _iter_bundled(name: str) -> list[Path]:
    found: list[Path] = []
    for pattern in CANDIDATES.get(name, []):
        if "*" in pattern:
            found.extend(sorted(paths.TOOLS_DIR.glob(pattern)))
        else:
            p = paths.TOOLS_DIR / pattern
            if p.exists():
                found.append(p)
    return found


def find_tool(name: str, configured: str = "", required: bool = True) -> ToolInfo | None:
    """按 配置 -> 自带 -> PATH 的顺序定位工具。"""
    if configured:
        p = Path(configured)
        if p.exists():
            return ToolInfo(name, str(p), source="config")
        if required:
            raise ToolMissingError(name, f"配置里指定的路径不存在：{configured}", [configured])
        return None

    for p in _iter_bundled(name):
        return ToolInfo(name, str(p), source="bundled")

    exe = CANDIDATES.get(name, [f"{name}.exe"])[0].split("/")[-1]
    which = shutil.which(exe) or shutil.which(name)
    if which:
        return ToolInfo(name, which, source="path")

    if required:
        raise ToolMissingError(name, ADVICE.get(name, f"请安装 {name} 并加入 PATH。"), [str(paths.TOOLS_DIR)])
    return None


def resolve_all(settings) -> dict[str, ToolInfo | None]:
    """一次性解析本流程需要的全部外部工具。"""
    out: dict[str, ToolInfo | None] = {}
    for name in ("ffmpeg", "ffprobe", "colmap"):
        configured = str(settings.get(f"tools.{name}", "") or "")
        out[name] = find_tool(name, configured, required=False)
    return out


# 各工具的“版本探测”参数不同：COLMAP 不认 -version，只认 help 子命令
VERSION_ARGS: dict[str, list[str]] = {
    "ffmpeg": ["-version"],
    "ffprobe": ["-version"],
    "colmap": ["help"],
    "magick": ["-version"],
}


def version_args(name: str) -> list[str]:
    return VERSION_ARGS.get(name, ["--version"])


def probe_version(path: str, args: list[str] | None = None) -> str:
    """尽量取版本号；失败返回空字符串（不影响主流程）。"""
    from . import runner

    code, text = runner.probe(path, args or ["-version"])
    if code < 0:
        return ""
    for line in (text or "").splitlines():
        line = line.strip()
        if line and not line.lower().startswith(("usage", "colmap", "===")):
            return line[:160]
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line[:160]
    return ""


def run_tool(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    log_file: Path | None = None,
    on_line=None,
    timeout: int | None = None,
    env_extra: dict[str, str] | None = None,
) -> int:
    """运行外部命令（保持旧签名，内部转给 tools.runner）。

    注意：Windows 下必须显式 encoding='utf-8' + errors='ignore'，
    否则 COLMAP 的非 UTF-8 输出会让 pipe 读线程崩掉。
    """
    from . import runner

    if not cmd:
        raise ValueError("run_tool 需要至少一个参数")
    tool_path, args = cmd[0], list(cmd[1:])
    return runner.run(tool_path, args, cwd=cwd, log_file=log_file, on_line=on_line,
                      timeout=timeout, env_extra=env_extra)


def which_any(names: list[str]) -> str | None:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")
