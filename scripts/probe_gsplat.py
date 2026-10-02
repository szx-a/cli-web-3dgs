"""诊断 gsplat 的 CUDA 扩展加载失败原因（会打印内部加载异常）。"""

from __future__ import annotations

import traceback


def main() -> int:
    import gsplat.cuda as gc

    print("gsplat cuda module:", gc.__file__)
    try:
        from gsplat.cuda import _wrapper

        print("wrapper:", _wrapper.__file__)
        print("_C before:", getattr(_wrapper, "_C", "n/a"))
        # 手动触发加载，把真实异常打出来
        try:
            import importlib

            build = importlib.import_module("gsplat.cuda._backend") if False else None
        except Exception:
            build = None
        if hasattr(_wrapper, "_load_lib"):
            try:
                _wrapper._load_lib()
            except Exception:
                print("=== _load_lib 抛出的真实异常 ===")
                traceback.print_exc()
        print("_C after:", getattr(_wrapper, "_C", "n/a"))
    except Exception:
        traceback.print_exc()

    # 是否有 ninja / cl / nvcc
    import shutil

    for exe in ("ninja", "cl", "nvcc", "cuda.dll"):
        print(f"{exe}: {shutil.which(exe)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
