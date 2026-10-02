"""验证便携 CUDA（无需管理员）能否真正编译 CUDA 内核。

背景：官方 CUDA 安装包本质是 7z 自解压包，Windows 自带的 bsdtar（tar.exe）能直接解开，
所以可以只取 cuda_nvcc + cuda_cudart + cuda_cccl 三个子包，拼成 tools\cuda 下的
免安装 CUDA（实测 358 MB，含 nvcc 12.8.93）。这条路线不需要管理员权限。

用法：env\\venv\\Scripts\\python.exe scripts\\verify_portable_cuda.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow import paths  # noqa: E402

paths.apply_runtime_env()

CUDA = paths.find_portable_cuda()


def run(cmd, **kw):
    env = paths.child_env()
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="ignore", env=env, timeout=1800, **kw)


def main() -> int:
    print("=" * 68)
    print("便携 CUDA 验证")
    print("=" * 68)
    if CUDA is None:
        print("未找到可用 CUDA（既没有 tools\\cuda，也没有系统安装版）")
        print("构建：Windows 自带 tar 直接解官方安装包的子包，例如：")
        print("  tar -xf downloads\\cuda_12.8.1_572.61_windows.exe -C env\\cuda_pkg cuda_nvcc cuda_cudart cuda_cccl")
        return 1

    nvcc = CUDA / "bin" / "nvcc.exe"
    print(f"CUDA 目录 : {CUDA}")
    print(f"nvcc      : {nvcc}")
    print(f"是否便携版: {'是（tools\\cuda，免安装）' if CUDA == paths.PORTABLE_CUDA_DIR else '否（系统安装版）'}")

    r = run([str(nvcc), "--version"])
    ver_line = [l for l in (r.stdout or "").splitlines() if "release" in l]
    print(f"版本      : {ver_line[-1].strip() if ver_line else '未知'}")

    # ---- 1) 用 nvcc 直接编译一个最小内核（exe）----
    work = paths.ENV_DIR / "verify-cuda"
    work.mkdir(parents=True, exist_ok=True)
    src = work / "hello.cu"
    src.write_text(textwrap.dedent("""
        #include <cstdio>
        __global__ void scale(float* a, float k, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) a[i] *= k;
        }
        int main() {
            const int n = 1024;
            float* a; cudaMalloc(&a, n * sizeof(float));
            scale<<<(n + 127) / 128, 128>>>(a, 2.0f, n);
            cudaError_t e = cudaDeviceSynchronize();
            printf("kernel launch: %s\\n", cudaGetErrorString(e));
            cudaFree(a);
            return (e == cudaSuccess) ? 0 : 1;
        }
    """), encoding="utf-8")
    exe = work / "hello.exe"
    print("\n[1/3] nvcc 编译并运行最小内核 ...")
    r = run([str(nvcc), "-arch=sm_120", "-o", str(exe), str(src)])
    if r.returncode != 0:
        print("  编译失败：")
        print("  " + "\n  ".join((r.stderr or r.stdout or "").splitlines()[-12:]))
        return 1
    print("  编译成功")
    r = run([str(exe)])
    out = (r.stdout or "").strip()
    print(f"  运行结果：{out}")
    kernel_ok = "success" in out

    # ---- 2) torch 能否看到这条 CUDA ----
    print("\n[2/3] 检查 torch 侧的 CUDA 探测 ...")
    code = (
        "import torch, os;"
        "print('torch', torch.__version__);"
        "print('cuda_available', torch.cuda.is_available());"
        "print('CUDA_HOME', os.environ.get('CUDA_HOME'));"
        "from torch.utils.cpp_extension import CUDA_HOME as TCH;"
        "print('torch.CUDA_HOME', TCH)"
    )
    r = run([sys.executable, "-c", code])
    print("  " + "\n  ".join((r.stdout or r.stderr or "").strip().splitlines()))

    # ---- 3) gsplat 的 CUDA 扩展能否 JIT 编译 ----
    print("\n[3/3] 触发 gsplat 的 CUDA 扩展 JIT 编译（第一次约 3~10 分钟，请耐心）...")
    code = (
        "import gsplat, torch;"
        "from gsplat.cuda._backend import _C;"
        "print('gsplat _C loaded:', _C is not None)"
    )
    r = run([sys.executable, "-c", code])
    tail = (r.stdout or "") + (r.stderr or "")
    ok = "_C loaded: True" in tail
    print("  " + "\n  ".join(tail.strip().splitlines()[-15:]))

    print("\n" + "=" * 68)
    print("结论")
    print("=" * 68)
    print(f"  nvcc 可编译 sm_120 内核 : {'是' if kernel_ok else '否'}")
    print(f"  gsplat CUDA 扩展可用    : {'是' if ok else '否'}")
    print(f"  是否用了管理员安装      : {'否（便携版）' if CUDA == paths.PORTABLE_CUDA_DIR else '是（系统安装版）'}")
    return 0 if (kernel_ok and ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
