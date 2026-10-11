"""用**子进程 + 超时**扫描 xatlas 的耗水面数（子进程能真正杀掉）。

为什么必须是子进程：我先前用"daemon 线程 + join(timeout)"，超时后主流程确实继续了，
但**线程杀不掉**，实测在 107,460 面的网格上白烧了 **3115 秒 CPU**（约 55 分钟）。
`xatlas.parametrize` 是单个 C 调用，只有独立进程才能被真正终止。

用法：
  env\\venv\\Scripts\\python.exe scripts\\diag_xatlas_scale.py
  env\\venv\\Scripts\\python.exe scripts\\diag_xatlas_scale.py --child --faces 4000 --ply <p>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_PLY = ROOT / "workspace" / "img2-01" / "05_export" / "model_mesh.ply"


def child(faces: int, ply: Path) -> int:
    import numpy as np
    import open3d as o3d
    import xatlas

    t_all = time.time()
    m = o3d.io.read_triangle_mesh(str(ply))
    t_load = time.time() - t_all
    t0 = time.time()
    if faces and len(m.triangles) > faces:
        m = m.simplify_quadric_decimation(faces)
    t_dec = time.time() - t0
    v = np.asarray(m.vertices, dtype="float32")
    t = np.asarray(m.triangles, dtype="uint32")
    t0 = time.time()
    vm, nt, uv = xatlas.parametrize(v, t)
    t_uv = time.time() - t0
    # 分项计时：load / QEM 简化 / UV 展开 —— 不分开看会误判瓶颈
    print(f"RESULT faces={len(t)} verts={len(v)} seconds={t_uv:.2f} "
          f"newverts={len(vm)} load={t_load:.2f} decimate={t_dec:.2f} total={time.time()-t_all:.2f}",
          flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--faces", type=int, default=0)
    ap.add_argument("--ply", default=str(DEFAULT_PLY))
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    if args.child:
        return child(args.faces, Path(args.ply))

    ply = Path(args.ply)
    if not ply.exists():
        print(f"缺少 {ply}；先用 export 生成网格")
        return 1
    import open3d as o3d

    total = len(o3d.io.read_triangle_mesh(str(ply)).triangles)
    print(f"源网格 {ply.name}：{total} 面")
    print(f"{'面数':>8}{'UV展开(s)':>12}{'简化(s)':>10}{'读取(s)':>10}{'新顶点':>10}")
    print("-" * 54)
    for f in (500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 0):
        cmd = [sys.executable, str(Path(__file__)), "--child",
               "--faces", str(f), "--ply", str(ply)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=args.timeout, errors="ignore")
        except subprocess.TimeoutExpired:
            print(f"{f if f else total:>8}{'>'+str(int(args.timeout)):>12}"
                  f"{'-':>10}{'-':>10}{'-':>10}  **超时被杀**")
            continue
        out = (r.stdout or "").strip().splitlines()
        line = next((l for l in reversed(out) if l.startswith("RESULT")), "")
        if line:
            d = dict(kv.split("=") for kv in line.split()[1:])
            print(f"{int(d['faces']):>8}{float(d['seconds']):>12.2f}"
                  f"{float(d['decimate']):>10.2f}{float(d['load']):>10.2f}"
                  f"{int(d['newverts']):>10}")
        else:
            err = (r.stderr or "").strip().splitlines()
            print(f"{f if f else total:>8}{'-':>12}{'-':>10}{'-':>10}{'-':>10}  "
                  f"失败: {err[-1][:44] if err else '无输出'}")
    print("\n判读：看是『UV展开』还是『简化』占了大头 —— 分项计时才知道真瓶颈。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
