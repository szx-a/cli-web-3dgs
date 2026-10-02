"""诊断：本机沙箱是否允许子进程使用管道（管道被禁会让 ninja 永久挂起）。

现象：gsplat 的内核编译里 ninja 进程 CPU=0、永不结束、日志为空。
怀疑：沙箱禁止命名管道，而 torch 用 stdout=PIPE 启动 ninja，
      ninja 再用管道与编译子进程通信。

用法：env\\venv\\Scripts\\python.exe scripts\\probe_pipe.py
"""

from __future__ import annotations

import subprocess
import sys
import time


def try_case(name: str, cmd: list[str], **kw) -> None:
    t0 = time.time()
    try:
        r = subprocess.run(cmd, timeout=30, **kw)
        out = (r.stdout or "") if isinstance(r.stdout, str) else (r.stdout or b"")[:120]
        print(f"  [OK]   {name:<34} exit={r.returncode}  用时 {time.time()-t0:.1f}s  out={str(out)[:60]!r}")
    except subprocess.TimeoutExpired:
        print(f"  [HANG] {name:<34} 超时 30s（疑似管道被禁）")
    except Exception as exc:
        print(f"  [FAIL] {name:<34} {type(exc).__name__}: {exc}")


def main() -> int:
    print("=== 子进程 stdio 组合测试 ===")
    py = sys.executable
    try_case("stdout=PIPE (text)", [py, "-c", "print('hi')"], capture_output=True, text=True)
    try_case("stdout=DEVNULL", [py, "-c", "print('hi')"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try_case("stdout=PIPE (bytes)", [py, "-c", "print('hi')"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("\n=== 真实工具（走管道）===")
    import shutil

    ninja = shutil.which("ninja")
    if ninja:
        try_case("ninja --version (PIPE)", [ninja, "--version"], capture_output=True, text=True)
        try_case("ninja --version (DEVNULL)", [ninja, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print("  未找到 ninja")

    print("\n=== torch 启动 ninja 的方式 ===")
    try:
        import torch.utils.cpp_extension as ce

        src = ""
        try:
            import inspect

            src = inspect.getsource(ce._run_ninja_build)
        except Exception as exc:
            src = f"(取源码失败: {exc})"
        for line in src.splitlines():
            if "Popen" in line or "stdout" in line or "stderr" in line or "subprocess" in line:
                print("   ", line.strip()[:110])
    except Exception as exc:
        print(f"  torch 不可用: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
