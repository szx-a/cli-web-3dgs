"""拆解 PLY 体积构成，并对比 glb / splat / 贴图网格的体积（回答"为什么 PLY 90 MB 而 glb 只有几 MB"）。

PLY 里每个高斯存的是**完整 3DGS 参数**：位置 3 + 法线 3（本项目有）+ 不透明度 1 + 尺度 3 + 旋转 4 + SH 48
= 62 个 float32 = 248 B/高斯。40 万高斯就是 ~96 MB。

而 glb / splat / 贴图网格存的是**渲染用的精简表示**：
* points glb：只存位置 + 顶点色（约 15 B/点）；
* splat：位置/尺度/颜色/旋转量化成 16 B/高斯；
* 贴图网格：顶点 + 索引 + 一张 2K 贴图，与高斯数量无关。

这个脚本把每部分字节数算出来，用于文档与汇报。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData

JOB = Path(r"F:\3dgs-and-ai\workspace\img5-solid-v3")


def main() -> int:
    ply = sorted((JOB / "04_train" / "point_cloud").glob("*/point_cloud.ply"),
                 key=lambda p: p.stat().st_size)[-1]
    size = ply.stat().st_size
    v = PlyData.read(str(ply))["vertex"]
    n = len(v["x"])

    groups = {"xyz": 0.0, "normal": 0.0, "f_dc": 0.0, "f_rest": 0.0,
              "opacity": 0.0, "scale": 0.0, "rot": 0.0, "other": 0.0}
    for p in v.properties:
        width = np.dtype(p.val_dtype).itemsize * (getattr(p, "count", 1) or 1)
        nm = p.name
        if nm in ("x", "y", "z"):
            groups["xyz"] += width
        elif nm in ("nx", "ny", "nz"):
            groups["normal"] += width
        elif nm.startswith("f_dc"):
            groups["f_dc"] += width
        elif nm.startswith("f_rest"):
            groups["f_rest"] += width
        elif nm == "opacity":
            groups["opacity"] += width
        elif nm.startswith("scale"):
            groups["scale"] += width
        elif nm.startswith("rot"):
            groups["rot"] += width
        else:
            groups["other"] += width

    total = sum(groups.values())
    print(f"PLY: {ply.relative_to(JOB)}")
    print(f"  文件 {size / 2**20:.1f} MB，高斯 {n:,}，{len(v.properties)} 个属性，"
          f"{size / n:.1f} B/高斯（理论 {total:.0f}）")
    for k, b in groups.items():
        if b:
            print(f"  {k:8s} {b:6.1f} B/高斯  {100 * b / total:5.1f}%"
                  f"   合计 {b * n / 2**20:6.2f} MB")

    print("\n其它产物：")
    for f in sorted((JOB / "05_export").iterdir()):
        if f.suffix.lower() in (".glb", ".ply", ".obj", ".splat", ".png", ".mtl"):
            print(f"  {f.name:34s} {f.stat().st_size / 2**20:7.2f} MB")

    # 估算 glb points 的理论体积
    print("\n理论对比（40 万高斯）：")
    print(f"  PLY 全参数（62 float32）：{n * 248 / 2**20:.1f} MB")
    print(f"  若只存 DC 颜色（14 float32 = 56 B）：{n * 56 / 2**20:.1f} MB")
    print(f"  glb points（pos3 + color3 float32 = 24 B）：{n * 24 / 2**20:.1f} MB")
    print(f"  splat（位置 3×4B + 尺度 3×4B + 颜色 4B + 旋转 4B = 20 B）：{n * 20 / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
