"""外部命令执行包装。

为什么需要它：COLMAP 的 Windows 官方包给的是 `COLMAP.bat`（内部依赖同目录的 DLL/插件，
必须在自身目录下启动）。直接把 .bat 当可执行文件调用会因为工作目录不对而失败，
所以这里统一做一层适配：

    * .bat / .cmd  -> 用 cmd.exe /c 调用，并把 cwd 固定到该脚本所在目录；
    * 其余情况     -> 直接调用，cwd 用调用方指定值（默认项目根）。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .. import paths


def build_cmd(tool_path: str, args: list[str]) -> tuple[list[str], Path | None]:
    """返回 (完整命令, 强制工作目录或 None)。"""
    p = Path(tool_path)
    if p.suffix.lower() in (".bat", ".cmd"):
        return ["cmd.exe", "/c", str(p)] + args, p.parent
    return [str(p)] + args, None


def run(
    tool_path: str,
    args: list[str],
    *,
    cwd: Path | None = None,
    log_file: Path | None = None,
    on_line=None,
    timeout: int | None = None,
    env_extra: dict[str, str] | None = None,
) -> int:
    """执行外部命令，逐行回调 + 可选落盘；返回退出码。"""
    cmd, forced_cwd = build_cmd(tool_path, args)
    workdir = forced_cwd or cwd or paths.PROJECT_ROOT
    env = paths.child_env(env_extra)

    with subprocess.Popen(
        cmd,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        bufsize=1,
        env=env,
    ) as proc:
        fh = log_file.open("a", encoding="utf-8") if log_file else None
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                if fh:
                    fh.write(line + "\n")
                if on_line:
                    on_line(line)
            proc.wait(timeout=timeout)
        finally:
            if fh:
                fh.close()
        return proc.returncode


def probe(tool_path: str, args: list[str], timeout: int = 60) -> tuple[int, str]:
    """短探测：返回 (退出码, 合并输出)。用于版本检查与能力探测。"""
    cmd, forced_cwd = build_cmd(tool_path, args)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(forced_cwd or paths.PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=paths.child_env(),
            encoding="utf-8",
            errors="ignore",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, str(exc)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


# ----------------------------------------------------------- 选项能力探测
_HELP_CACHE: dict[tuple[str, str], set[str]] = {}


def help_tokens(tool_path: str, subcommand: str, timeout: int = 90) -> set[str]:
    """抓取某个子命令的 --help，返回它认识的全部选项名（小写，含前导 --）。

    为什么需要：COLMAP 各版本选项名会变（例如 3.12 把
    SequentialMatching.guided_matching 收归到 SiftMatching.guided_matching，
    某些版本还会少掉 loop_detection）。写死参数会直接报
    "unrecognised option" 并让整个阶段失败，所以这里做能力探测。
    """
    key = (str(tool_path), subcommand)
    if key in _HELP_CACHE:
        return _HELP_CACHE[key]
    _, out = probe(tool_path, [subcommand, "--help"], timeout=timeout)
    tokens: set[str] = set()
    for line in (out or "").splitlines():
        line = line.strip()
        if not line.startswith("--"):
            continue
        token = line.split()[0].split("=")[0].strip().lower()
        tokens.add(token)
    _HELP_CACHE[key] = tokens
    return tokens


def filter_options(
    tool_path: str,
    subcommand: str,
    args: list[str],
    on_drop=None,
) -> list[str]:
    """丢掉当前 COLMAP 版本不认识的选项，返回可安全执行的参数列表。

    选项与其取值成对出现（`--K` `V`）；只丢掉“选项+它的值”，
    位置参数（如子命令名、路径）不动。
    """
    supported = help_tokens(tool_path, subcommand)
    if not supported:
        return args
    out: list[str] = []
    i = 0
    while i < len(args):
        item = args[i]
        if item.startswith("--"):
            name = item.split("=")[0].strip().lower()
            if name not in supported:
                dropped_value = None
                if "=" not in item and i + 1 < len(args) and not args[i + 1].startswith("--"):
                    dropped_value = args[i + 1]
                    i += 1
                if on_drop:
                    on_drop(item, dropped_value)
                i += 1
                continue
        out.append(item)
        i += 1
    return out
