"""导出验证 `parsePly` 用的 3DGS PLY 样本（供 `probe_gauss_ply.mjs` 比对）。

## 为什么不用 `read_gaussians_full` 取基准

它为了渲染效率**按不透明度降序重排**（`np.argsort(-op)`），
而查看器的 `parsePly` 读的是**文件原序**。第一版我用它当基准，
于是"所有字段都不一致"（连位置都差 0.87）—— 那是**比对基准错**，不是解析错。

⇒ 这里用 numpy **直接按 PLY 文件原序**解析，作为独立基准。
这样既对齐了顺序，也真正做到"两套独立实现互校"。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SIZE = {"char": 1, "uchar": 1, "int8": 1, "uint8": 1, "short": 2, "ushort": 2,
        "int16": 2, "uint16": 2, "int": 4, "uint": 4, "int32": 4, "uint32": 4,
        "float": 4, "float32": 4, "double": 8, "float64": 8}
NP_DT = {"char": "<i1", "uchar": "<u1", "int8": "<i1", "uint8": "<u1",
         "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
         "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
         "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}


def parse_ply_raw(ply: Path) -> dict:
    """按文件原序解析 binary little-endian PLY 的 vertex 属性。"""
    raw = ply.read_bytes()
    i = raw.find(b"end_header")
    if i < 0:
        raise RuntimeError("找不到 end_header")
    j = raw.find(b"\n", i) + 1
    head = raw[:j].decode("ascii", "replace")
    props = []
    vcount = 0
    in_v = False
    for line in (l.strip() for l in head.splitlines()):
        if not line:
            continue
        if line.startswith("element "):
            parts = line.split()
            in_v = parts[1] == "vertex"
            if in_v:
                vcount = int(parts[2])
            continue
        if line.startswith("property ") and in_v:
            p = line.split()
            props.append((p[2], NP_DT.get(p[1], "<f4")))
    if not props or vcount == 0:
        raise RuntimeError("没有 vertex 属性")
    dt = np.dtype([(n, t) for n, t in props])
    arr = np.frombuffer(raw, dtype=dt, count=vcount, offset=j)
    return {"props": [n for n, _ in props], "vcount": vcount, "arr": arr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="img3-crop")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    job = ROOT / "workspace" / args.job
    plys = sorted((job / "04_train" / "point_cloud").glob("*/point_cloud.ply"), key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0))
    if not plys:
        print("找不到点云 PLY")
        return 1
    ply = plys[-1]
    d = parse_ply_raw(ply)
    arr, names = d["arr"], d["props"]
    print(f"PLY {ply.name}: {d['vcount']} 顶点 / {len(names)} 个属性")

    rest = sorted((k for k in names if k.startswith("f_rest_")),
                  key=lambda s: int(s.rsplit("_", 1)[1]))
    per_ch = len(rest) // 3 if rest else 0
    bases = per_ch + 1
    side = int(round(bases ** 0.5))
    sh_degree = side - 1 if side * side == bases else 0
    print(f"  f_rest 数量 {len(rest)}，每通道 {per_ch}，SH 阶 {sh_degree}")

    cases = []
    for i in range(min(args.n, d["vcount"])):
        sh_dc = [float(arr["f_dc_%d" % c][i]) for c in range(3)] if "f_dc_0" in names \
            else [0.0, 0.0, 0.0]
        sh_rest = []
        if rest:
            # 通道主序：先 R 的 per_ch 个基，再 G，再 B
            for ch in range(3):
                for k in range(per_ch):
                    sh_rest.append(float(arr["f_rest_%d" % (ch * per_ch + k)][i]))
        cases.append({
            "idx": i,
            "pos": [float(arr[k][i]) for k in ("x", "y", "z")],
            "scale": [float(arr["scale_%d" % k][i]) for k in range(3)],
            "quat": [float(arr["rot_%d" % k][i]) for k in range(4)],
            "opacity": float(arr["opacity"][i]),
            "sh_dc": sh_dc,
            "sh_rest": sh_rest,
        })

    out = ROOT / "env" / "tmp" / "gauss_probe.json"
    out.write_text(json.dumps({
        "ply_path": str(ply),
        "count": int(d["vcount"]),
        "n_rest": int(len(rest)),
        "sh_degree": int(sh_degree),
        "cases": cases,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"已写出 {out}（{len(cases)} 个样本，按**文件原序**）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
