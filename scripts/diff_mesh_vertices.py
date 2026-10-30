"""比较精修前后网格的顶点位移（量化"精修到底改了多少"）。"""
from __future__ import annotations

import numpy as np
from pathlib import Path

ROOT = Path(r"F:\3dgs-and-ai")


def load_vertices(job: str) -> np.ndarray:
    d = ROOT / "workspace" / job / "05_export"
    cands = sorted(d.glob("*mesh*.ply")) or sorted(d.glob("*mesh*.obj"))
    if not cands:
        cands = sorted(d.glob("model*.ply"))
    p = cands[0]
    if p.suffix == ".obj":
        vs = []
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
        return np.asarray(vs)
    # 二进制/文本 PLY 的顶点
    from plyfile import PlyData

    ply = PlyData.read(str(p))
    v = ply["vertex"]
    return np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)


a = load_vertices("img5-refine-r0")
b = load_vertices("img5-refine-r100")
print(f"  顶点数：r0={len(a)}  r100={len(b)}")
if len(a) == len(b):
    d = np.linalg.norm(b - a, axis=1)
    print(f"  位移：中位 {np.median(d):.5f}  p90 {np.percentile(d, 90):.5f}  最大 {d.max():.5f}")
    print(f"  相对场景尺度：中位 {np.median(d)/3.678*100:.2f}%  p90 {np.percentile(d, 90)/3.678*100:.2f}%")
    print(f"  有位移的顶点比例：{(d > 1e-6).mean()*100:.1f}%")
else:
    print("  顶点数不同，无法逐点比较（拓扑应一致，需检查）")
