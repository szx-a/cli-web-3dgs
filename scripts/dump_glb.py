"""诊断 GLB 内部结构：accessor / bufferView / 索引编码是否自洽。

用在"网格 glb 校验报索引越界"这类问题上 —— 直接把真实字节打出来，
比反复猜快得多。

用法：env\\venv\\Scripts\\python.exe scripts\\dump_glb.py --path <file.glb>
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    args = ap.parse_args()

    raw = Path(args.path).read_bytes()
    magic, ver, total = struct.unpack("<III", raw[:12])
    print(f"magic={raw[:4]!r} version={ver} total={total} actual={len(raw)}")
    clen, ctype = struct.unpack("<II", raw[12:20])
    g = json.loads(raw[20:20 + clen].decode("utf-8"))
    boff = 20 + clen + 8
    print("顶层键:", sorted(g.keys()))
    for key in ("images", "textures", "samplers", "materials"):
        if key in g:
            print(f"{key}: {json.dumps(g[key], ensure_ascii=False)[:200]}")
    prim = g["meshes"][0]["primitives"][0]
    print("attributes:", prim.get("attributes"), "indices:", prim.get("indices"),
          "mode:", prim.get("mode"))
    for i, a in enumerate(g.get("accessors", [])):
        print(f"  accessor[{i}] type={a['type']} ct={a['componentType']} "
              f"count={a['count']} bv={a.get('bufferView')}")
    for i, bv in enumerate(g.get("bufferViews", [])):
        print(f"  bufferView[{i}] off={bv['byteOffset']} len={bv['byteLength']}")
    idx_id = prim.get("indices")
    if idx_id is not None:
        ia = g["accessors"][idx_id]
        bv = g["bufferViews"][ia["bufferView"]]
        seg = raw[boff + bv["byteOffset"]: boff + bv["byteOffset"] + 24]
        print("索引前 24 字节:", seg.hex())
        fmt = "<H" if ia["componentType"] == 5123 else "<I"
        step = 2 if ia["componentType"] == 5123 else 4
        n = len(seg) // step
        vals = struct.unpack(f"<{n}{fmt[1]}", seg[:n * step])
        print(f"按 {('uint16' if step==2 else 'uint32')} 解读:", vals)
        nv = g["accessors"][prim["attributes"]["POSITION"]]["count"]
        print(f"顶点数 {nv}；声明索引数 {ia['count']}")
        print("byteLength 与 count*step 是否一致:",
              bv["byteLength"] == ia["count"] * step,
              f"({bv['byteLength']} vs {ia['count']*step})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
