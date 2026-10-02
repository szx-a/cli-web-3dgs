"""不依赖 ninja 的 CUDA 扩展构建器（torch cpp_extension 的替代实现）。

为什么需要：本机沙箱下 ninja（官方版与 MinGW 版都试过）**能完成任务却不退出进程**，
而 torch 的 `_run_ninja_build` 是同步等待 ninja 结束的，于是 gsplat 的 JIT 编译永远卡住
（现象：ninja 进程 CPU=0、构建目录里只有 build.ninja、日志为空）。

做法：直接解析 torch 生成的 `build.ninja`，按顺序自己调用 cl.exe / nvcc.exe / link.exe。
它不是通用 ninja 实现，但足以覆盖 torch 生成的构建文件结构。

启用方式：`paths.apply_runtime_env()` 检测到 ninja 不可用时会自动把
`torch.utils.cpp_extension._run_ninja_build` 替换成一个仅解析参数的包装函数；
真正的构建逻辑在本文件。可用环境变量 `GSFLOW_FORCE_NINJA=1` 强制回到 ninja。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# --------------------------------------------------------------------- 解析
def _unescape(text: str) -> str:
    """ninja 转义：`$ ` 转义空格、`$$` 转义美元、`$:` 转义冒号。"""
    out = text.replace("$ ", " ").replace("$$", "$").replace("$:", ":")
    return out.strip()


def parse_ninja(path: Path) -> dict:
    """解析 torch 生成的 build.ninja，返回 {vars, rules, builds}。"""
    text = path.read_text(encoding="utf-8", errors="ignore")
    # 续行：行尾的 $ 表示下一行继续（torch 的 flags 会这样折行）
    text = re.sub(r"\$\s*\n", " ", text)
    vars_: dict[str, str] = {}
    rules: dict[str, dict[str, str]] = {}
    builds: list[dict[str, object]] = []
    cur_rule: str | None = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("rule "):
            cur_rule = line[5:].strip()
            rules[cur_rule] = {}
            continue
        if line.startswith("build "):
            cur_rule = None
            m = re.match(r"build\s+(.*?):\s*(\S+)\s*(.*)$", line)
            if not m:
                continue
            outs = [_unescape(o) for o in m.group(1).split()]
            ins = [_unescape(i) for i in m.group(3).split()] if m.group(3).strip() else []
            builds.append({"outs": outs, "rule": m.group(2).strip(), "ins": ins, "vars": {}})
            continue
        if line.startswith((" ", "\t")):
            body = line.strip()
            if "=" not in body:
                continue
            key, _, val = body.partition("=")
            key, val = key.strip(), val.strip()
            if cur_rule:
                rules[cur_rule][key] = val
            elif builds:
                builds[-1]["vars"][key] = val  # type: ignore[union-attr]
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            vars_[key.strip()] = val.strip()

    return {"vars": vars_, "rules": rules, "builds": builds}


def _expand(text: str, env: dict[str, str], depth: int = 0) -> str:
    """展开 ninja 变量（支持规则级覆盖、嵌套引用）。"""
    if depth > 12 or "$" not in text:
        return text

    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name == "$":
            return "$"
        return env.get(name, "")

    expanded = re.sub(r"\$(\w+)", repl, text)
    return expanded if expanded == text else _expand(expanded, env, depth + 1)


def _build_env(parsed: dict, build: dict, rule_name: str) -> dict[str, str]:
    env = dict(parsed["vars"])
    env.update(parsed["rules"].get(rule_name, {}))
    # build 语句自带的变量（$in / $out 等）优先级最高
    env["in"] = " ".join(build["ins"])          # type: ignore[arg-type]
    env["out"] = " ".join(build["outs"])        # type: ignore[arg-type]
    env.update(build.get("vars", {}))           # type: ignore[arg-type]
    return env


# --------------------------------------------------------------------- 执行
# nvcc 需要的额外参数与"头文件遮蔽"。
#
# 实测踩坑：`cub/cub.cuh` 会间接引入 Windows 的 rpcndr.h，其中 `#define small char`
# 把 torch 头 `StreamSegmentSize(cudaStream_t s, bool small, size_t sz)` 的参数名
# 替换成 `char`，报 "invalid combination of type specifiers"；只有包含 cub 的 .cu 会踩到。
# 试过但**无效**的做法：`-Xcompiler /Usmall`（生效太晚）、
# `-Dsmall=...` 与 `-include` 注入（rpcndr.h 仍然会把 small 覆盖成 char）。
#
# 有效做法：在 nvcc 的 include 搜索路径**前面**放一个目录，里面是与 torch 同路径的
# `c10/cuda/CUDACachingAllocator.h`，内容先 undef small 再 include 真正的头文件。
# 这样 torch/cub 的其他部分完全不受影响。
_SHADOW_NAME = "_gsflow_shadow"
_REAL_INC_MARKER = _SHADOW_NAME + "/c10/cuda/CUDACachingAllocator.h"
_SHADOW_HEADER = """\
/* gsflow: 遮蔽 torch 的 c10/cuda/CUDACachingAllocator.h。
   原因：Windows 的 rpcndr.h 会 `#define small char`（被 cub 间接引入），
   而该头文件里有个参数就叫 small，被宏替换成 char 后编译失败。
   这里在包含真正的头文件之前把 small 取消定义。 */
#pragma push_macro("small")
#undef small
#include "{real}"
#pragma pop_macro("small")
"""


def _ensure_shadow(build_dir: Path, real_header: str) -> Path:
    """创建遮蔽目录并写入修正版头文件，返回该目录。"""
    d = build_dir / _SHADOW_NAME / "c10" / "cuda"
    d.mkdir(parents=True, exist_ok=True)
    target = d / "CUDACachingAllocator.h"
    content = _SHADOW_HEADER.format(real=real_header.replace("\\", "/"))
    try:
        if not target.exists() or target.read_text(encoding="utf-8") != content:
            target.write_text(content, encoding="utf-8")
    except OSError:
        pass
    return build_dir / _SHADOW_NAME


def _normalize_for_nvcc(cmd: str, build_dir: Path) -> str:
    """给 nvcc 命令补充遮蔽目录（幂等）。"""
    if "nvcc" not in cmd.lower():
        return cmd
    # 找到 torch 的 include 目录，用于 #include 真正的头文件
    real = ""
    for token in cmd.split():
        t = token.strip('"')
        if t.startswith("-I") and "torch" in t and t.endswith("include"):
            cand = Path(t[2:]) / "c10" / "cuda" / "CUDACachingAllocator.h"
            if cand.exists():
                real = str(cand)
                break
    if not real:
        return cmd
    shadow = _ensure_shadow(build_dir, real)
    if _SHADOW_NAME in cmd:
        return cmd
    # 插到最前面，保证优先于 torch 的 include 被搜索到
    return cmd.replace("nvcc ", f'nvcc -I"{shadow}" ', 1)


def _run(cmd: str, cwd: Path, log) -> tuple[int, str]:
    proc = subprocess.run(
        cmd, cwd=str(cwd), shell=True, capture_output=True, text=True,
        encoding="utf-8", errors="ignore", env=os.environ.copy(),
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if log:
        log.write(f"\n$ {cmd}\n{out}\n")
        log.flush()
    return proc.returncode, out


# cl.exe 不认识的 GCC 风格参数：torch 在 Windows 上也会带上它们（-O3 / -Wno-* 等），
# 直接原样传会报 "命令行 error D8021 : 无效的数值参数" 而整批 .cpp 编译失败。
_MSVC_DROP_PREFIXES = ("-Wno-", "-W", "-f", "-m", "-g", "-O0", "-O1", "-O2", "-O3", "-Os", "-std=c++")
_MSVC_MAP = {
    "-O3": "/O2",
    "-O2": "/O2",
    "-O1": "/O1",
    "-O0": "/Od",
}


def _normalize_for_msvc(cmd: str) -> tuple[str, list[str]]:
    """把命令里的 GCC 风格参数替换/剔除成 MSVC 认识的写法。只处理 cl.exe 命令。"""
    if not re.match(r'^\s*"?[^"]*cl(\.exe)?"?\s', cmd, re.IGNORECASE) and " cl " not in f" {cmd} ":
        return cmd, []
    parts = cmd.split()
    out: list[str] = []
    dropped: list[str] = []
    for p in parts:
        if p in _MSVC_MAP:
            out.append(_MSVC_MAP[p])
            continue
        if p.startswith("-"):
            if p.startswith(_MSVC_DROP_PREFIXES) or p == "-use_fast_math":
                dropped.append(p)
                continue
        out.append(p)
    return " ".join(out), dropped


def run_build(build_directory, verbose, error_prefix, **kwargs):
    """替代 `torch.utils.cpp_extension._run_ninja_build`。

    torch 的补充参数（ninja_required_version 等）会被忽略——
    我们的调用只依赖 build_directory / verbose / error_prefix。
    """
    build_dir = Path(build_directory)
    ninja_file = build_dir / "build.ninja"
    if not ninja_file.exists():
        raise RuntimeError(f"{error_prefix}\n找不到 {ninja_file}")

    parsed = parse_ninja(ninja_file)
    builds = [b for b in parsed["builds"] if b["rule"] in ("compile", "cuda_compile")]
    links = [b for b in parsed["builds"] if b["rule"] == "link"]
    if not builds and not links:
        raise RuntimeError(f"{error_prefix}\nbuild.ninja 里没有可执行的构建步骤")

    max_jobs = int(os.environ.get("MAX_JOBS", "4") or 4)
    print(f"[gsflow-build] ninja 不可用，改用内置构建器：{len(builds)} 个编译 + {len(links)} 个链接，"
          f"并行度 {max_jobs}", flush=True)

    log_path = build_dir / "gsflow_build.log"
    failure: dict | None = None
    t0 = time.time()
    done = 0
    dropped_all: set[str] = set()

    def prepare(b: dict) -> str:
        env = _build_env(parsed, b, b["rule"])                       # type: ignore[arg-type]
        cmd = _expand(parsed["rules"][b["rule"]]["command"], env)    # type: ignore[index]
        # 展开后必须再解一次转义：ninja 把链接器的 "C:\..." 写成 "C$:\..."，
        # 不解开就会得到 `C$:\Program Files\...`，cmd 报
        # "The filename, directory name, or volume label syntax is incorrect."
        cmd = _unescape(cmd)
        if b["rule"] == "compile":
            # /showIncludes 会把上万行 "注意: 包含文件" 灌进日志，我们并不解析它
            cmd = cmd.replace(" /showIncludes", "").replace(" /showincludes", "")
            cmd, dropped = _normalize_for_msvc(cmd)
            dropped_all.update(dropped)
        elif b["rule"] == "cuda_compile":
            cmd = _normalize_for_nvcc(cmd, build_dir)
        return cmd

    with log_path.open("w", encoding="utf-8") as log:
        with ThreadPoolExecutor(max_workers=max_jobs) as pool:
            futures = {}
            for b in builds:
                cmd = prepare(b)
                futures[pool.submit(_run, cmd, build_dir, log)] = (b, cmd)
            for fut in futures:
                b, cmd = futures[fut]
                try:
                    rc, out = fut.result()
                except Exception as exc:  # 子进程启动失败
                    rc, out = -1, f"{type(exc).__name__}: {exc}"
                done += 1
                target = b["outs"][0]                            # type: ignore[index]
                if rc != 0 and failure is None:
                    failure = {"cmd": cmd, "out": out[-4000:], "target": target}
                print(f"[gsflow-build] ({done}/{len(builds)}) {target} {'OK' if rc == 0 else 'FAIL'}", flush=True)

        if dropped_all:
            print(f"[gsflow-build] 已为 cl.exe 剔除 GCC 风格参数：{', '.join(sorted(dropped_all))}", flush=True)

        if failure is None:
            for b in links:
                env = _build_env(parsed, b, b["rule"])           # type: ignore[arg-type]
                cmd = _unescape(_expand(parsed["rules"][b["rule"]]["command"], env))  # type: ignore[index]
                print(f"[gsflow-build] linking {b['outs'][0]}", flush=True)  # type: ignore[index]
                rc, out = _run(cmd, build_dir, log)
                if rc != 0:
                    failure = {"cmd": cmd, "out": out[-4000:], "target": str(b["outs"][0])}  # type: ignore[index]
                    break

    print(f"[gsflow-build] 用时 {time.time()-t0:.1f}s，日志 {log_path}", flush=True)
    if failure:
        raise RuntimeError(
            f"{error_prefix}\n\n命令：{failure['cmd']}\n\n输出：\n{failure['out']}\n\n"
            f"完整日志：{log_path}\n（本机构建由 gsflow 内置构建器完成，因为 ninja 在本环境不退出）"
        )
    return None


def install_bypass() -> bool:
    """把 torch 的 ninja 构建替换成本模块的构建器；返回是否替换成功。"""
    try:
        import torch.utils.cpp_extension as ce

        def wrapper(build_directory, verbose, error_prefix, *a, **kw):
            return run_build(build_directory, verbose, error_prefix, *a, **kw)

        ce._run_ninja_build = wrapper  # type: ignore[assignment]
        return True
    except Exception:
        return False
