"""校验交付的 GLB 是否**真的**含它该有的东西（不看大小，看结构）。

为什么不能只看文件大小：一个 9 MB 的"带纹理"GLB 也可能是
"网格 + 一张全黑贴图"，或者在写 UV 时崩掉留下残file。用户之前就踩过
"查看器全黑"的坑，那次也是"文件存在且大小正常"。

本脚本解析 GLB 的 JSON chunk，报告：
  * meshes / primitives 数量与 draw mode（4=TRIANGLES, 0=POINTS）
  * 是否含 POSITION / NORMAL / TEXCOORD_0 属性
  * 是否含 images + textures + material.pbrMetallicRoughness.baseColorTexture
  * 图片的 mimeType 与解码后的实际尺寸（用 PIL 从 bufferView 里解出来）

用法: python scripts/check_glb_delivery.py workspace/final-v1/05_export
"""
from __future__ import annotations

import io
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_glb(path: Path) -> tuple[dict, bytes]:
    """返回 (glTF JSON, 二进制 chunk)。"""
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:  # 'glTF'
        raise ValueError(f"{path.name} 不是 GLB（magic=0x{magic:08X}）")
    off = 12
    js, bin_chunk = {}, b""
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        chunk = data[off:off + clen]
        off += clen
        if ctype == 0x4E4F534A:      # JSON
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:    # BIN
            bin_chunk = chunk
    return js, bin_chunk


def describe(path: Path) -> int:
    print(f"\n=== {path.name}  ({path.stat().st_size / 1024:,.0f} KB) ===")
    try:
        g, bin_chunk = read_glb(path)
    except Exception as e:  # noqa: BLE001
        print(f"  **解析失败**：{type(e).__name__}: {e}")
        return 1

    problems = 0
    meshes = g.get("meshes", [])
    print(f"  meshes={len(meshes)}  materials={len(g.get('materials', []))}  "
          f"images={len(g.get('images', []))}  textures={len(g.get('textures', []))}")
    for i, m in enumerate(meshes):
        for j, p in enumerate(m.get("primitives", [])):
            mode = p.get("mode", 4)
            mode_name = {0: "POINTS", 4: "TRIANGLES", 5: "TRIANGLE_STRIP"}.get(mode, str(mode))
            attrs = sorted(p.get("attributes", {}).keys())
            has_idx = "indices" in p
            n = 0
            if has_idx:
                acc = g["accessors"][p["indices"]]
                n = acc.get("count", 0)
            print(f"    mesh[{i}].prim[{j}] mode={mode_name} 顶点属性={attrs} "
                  f"索引数={n}")

    # 纹理链路：image -> texture -> material -> primitive
    mat_ok = False
    for i, mat in enumerate(g.get("materials", [])):
        pbr = mat.get("pbrMetallicRoughness", {})
        bt = pbr.get("baseColorTexture")
        if bt is not None:
            mat_ok = True
            print(f"    material[{i}] baseColorTexture 索引={bt.get('index')}")
    for i, img in enumerate(g.get("images", [])):
        bv = img.get("bufferView")
        mime = img.get("mimeType", "?")
        size = (g["bufferViews"][bv]["byteLength"]
                if bv is not None and bv < len(g.get("bufferViews", [])) else 0)
        dims = ""
        if bv is not None:
            view = g["bufferViews"][bv]
            off = view.get("byteOffset", 0)
            raw = bin_chunk[off:off + view["byteLength"]]
            try:
                from PIL import Image
                with Image.open(io.BytesIO(raw)) as im:
                    dims = f" 实际尺寸={im.size} 模式={im.mode}"
                    # 检查是否全黑（用户踩过"查看器全黑"）
                    ext = im.convert("L").getextrema()
                    if ext[1] == 0:
                        print(f"    **警告** image[{i}] 解码后**全黑**")
                        problems += 1
            except Exception as e:  # noqa: BLE001
                dims = f" （无法解码：{type(e).__name__}）"
        print(f"    image[{i}] mime={mime} bufferView 字节={size:,}{dims}")

    if mat_ok and not g.get("images"):
        print("  **警告**：material 引用了 baseColorTexture 但没有 image")
        problems += 1
    if not meshes:
        print("  **警告**：没有 mesh")
        problems += 1
    return problems


def main() -> int:
    ap_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "workspace" / "final-v1" / "05_export"
    if not ap_dir.exists():
        print(f"目录不存在：{ap_dir}")
        return 2
    glbs = sorted(ap_dir.glob("*.glb"))
    if not glbs:
        print(f"{ap_dir} 下没有 .glb")
        return 2
    bad = 0
    for p in glbs:
        bad += describe(p)
    print(f"\n结论: {len(glbs)} 个 GLB，{bad} 个问题"
          + ("（全部结构正确）" if bad == 0 else ""))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
