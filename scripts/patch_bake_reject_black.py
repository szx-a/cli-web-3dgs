"""给 `bake_texture_from_views` 加上 `bake_reject_black` 参数与统计（幂等）。

为什么用脚本改而不是手动编辑：该函数有 4 处需要同步改动
（签名、计数器初始化、采样逻辑、返回统计），手动改容易漏；
脚本可以断言"每处都恰改到 1 次"，改不到就报错而不是静默跳过。

用法: python scripts/patch_bake_reject_black.py [--apply]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "gsflow" / "gsdata" / "mesh_export.py"

EDITS = [
    # (说明, 原文, 新文)
    ("签名加参数",
     "def bake_texture_from_views(mesh, uv_tris, uv, images, c2ws, ks, src_whs,\n"
     "                            tex_size: int = 2048, batch: int = 200,\n"
     "                            margin_px: int = 2, max_views: int = 60):",
     "def bake_texture_from_views(mesh, uv_tris, uv, images, c2ws, ks, src_whs,\n"
     "                            tex_size: int = 2048, batch: int = 200,\n"
     "                            margin_px: int = 2, max_views: int = 60,\n"
     "                            bake_reject_black: bool = True):"),
    ("计数器初始化",
     "    tex = np.zeros((tex_size, tex_size, 3), dtype=np.float32)\n"
     "    filled = np.zeros((tex_size, tex_size), dtype=bool)",
     "    tex = np.zeros((tex_size, tex_size, 3), dtype=np.float32)\n"
     "    filled = np.zeros((tex_size, tex_size), dtype=bool)\n"
     "    # 因采到黑背景而被拒绝的采样数（写进统计，避免行为变化被静默吞掉）\n"
     "    n_rejected = 0"),
    ("返回统计",
     '{"tex_size": tex_size, "coverage": float(filled.mean()),\n'
     '             "views_used": len(cams)})',
     '{"tex_size": tex_size, "coverage": float(filled.mean()),\n'
     '             "views_used": len(cams),\n'
     '             "samples_rejected_black": int(n_rejected)})'),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    text = TARGET.read_text(encoding="utf-8")
    if "bake_reject_black: bool = True" in text and "samples_rejected_black" in text:
        print("已打过该补丁（幂等）")
        return 0

    new = text
    failed = []
    for label, old, rep in EDITS:
        n = new.count(old)
        if n != 1:
            failed.append(f"{label}：命中 {n} 次（应为 1）")
            continue
        new = new.replace(old, rep, 1)
        print(f"  [OK] {label}")
    if failed:
        print("**以下改动未生效**：")
        for f in failed:
            print("  -", f)
        return 1
    if args.apply:
        TARGET.write_text(new, encoding="utf-8")
        print(f"\n已写入 {TARGET}")
    else:
        print(f"\n（预览，未写入。加 --apply 生效）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
