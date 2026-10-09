"""打印解析出的 MSVC 环境里与编译相关的关键变量，定位 cl.exe 为何不在 PATH。

用法：env\\venv\\Scripts\\python.exe scripts\\diag_msvc_env.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow import paths  # noqa: E402


def main() -> int:
    v = paths.find_vcvars64()
    print("vcvars64:", v)
    env = paths.load_msvc_env()
    print(f"解析出变量数: {len(env)}")
    for k in ("VCToolsInstallDir", "VCINSTALLDIR", "VSCMD_ARG_TGT_ARCH",
              "WindowsSdkDir", "UniversalCRTSdkDir"):
        print(f"  {k} = {env.get(k)}")
    path = env.get("PATH", "")
    parts = [p for p in path.split(os.pathsep) if p]
    print(f"PATH 段数: {len(parts)}")
    hit_msvc = [p for p in parts if "MSVC" in p or "VC\\Tools" in p]
    print("含 MSVC 的 PATH 段:")
    for p in hit_msvc:
        print("   ", p)
    print("当前进程 PATH 是否已含 vcvars 的段:",
          all(p in os.environ.get("PATH", "") for p in hit_msvc) if hit_msvc else "无")
    # 直接拿候选 cl.exe 验证存在性
    for p in hit_msvc:
        cl = Path(p) / "cl.exe"
        print(f"   {cl} 存在={cl.exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
