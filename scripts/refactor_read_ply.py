"""把 `ExportStage._read_ply` 的方法体替换为**委托给** `gsflow.gsdata.shio`。

为什么要做（避免再次分叉）：`_read_ply` 与 `shio.read_gaussians_full` 是同一件事。
本会话已经因为"同一个决定在 4 处各写一遍"而把高阶 SH 全丢了（低估 1.91 dB）。
留两份实现 = 留两次分叉的机会。

用法: python scripts/refactor_read_ply.py [--apply]
  不带 --apply 时只打印将要替换的行区间，便于先核对。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "gsflow" / "stages" / "export.py"

NEW_BODY = '''    def _read_ply(ply: Path, max_points: int = 0) -> dict:
        """读取 3DGS PLY（**委托给 `gsflow.gsdata.shio.read_gaussians_full`**）。

        **重要修正（用户指出的"实现被简化"）**：
        旧版只解 `f_dc_*` 成 RGB 并**丢掉全部高阶 SH**。但训练用 `sh_degree=3`
        ⇒ PLY 里有 45 个 `f_rest_*`。丢掉它们意味着：
          * 预览图 / 网格顶点色**不是模型的真实外观**（缺视角相关的色彩变化）；
          * 用 `sh_degree=0` 渲染来评估，实测**低估 1.91 dB**
            （同模型：全 SH 22.10 dB vs 仅 DC 20.20 dB），
            而且对"改善视角相关细节"的优化手段（感知损失等）**天然不敏感**
            —— 这会直接制造**假阴性**（把有效的优化判成无效）。

        返回值：
          * `rgb`        —— DC 解出的视角无关基色（GLB/OBJ 顶点色用）
          * `shs`        —— **完整** SH [N, (deg+1)², 3]，供 gsplat 真实渲染
          * `sh_degree`  —— 由 `f_rest_*` 数量反推（45 ⇒ 3）

        实现只在 `gsdata/shio.py` 一处 —— 本会话已因"同一决定写 4 遍"而丢过高阶 SH，
        所以这里不再自己解析，直接委托，避免再次分叉。
        """
        from ..gsdata.shio import read_gaussians_full

        return read_gaussians_full(ply, max_points=max_points)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    lines = TARGET.read_text(encoding="utf-8").splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines)
                  if l.strip().startswith("def _read_ply(")), None)
    if start is None:
        print("找不到 def _read_ply")
        return 1
    # 方法体结束：下一个**同级缩进**的 def（4 个空格）
    end = None
    for i in range(start + 1, len(lines)):
        if re.match(r"^    def ", lines[i]) or re.match(r"^    # ---", lines[i]):
            end = i
            break
    if end is None:
        print("找不到 _read_ply 的结束位置")
        return 1

    print(f"将替换 export.py 的第 {start + 1}..{end} 行（共 {end - start} 行）")
    print(f"  首行: {lines[start].rstrip()[:70]}")
    print(f"  末行: {lines[end - 1].rstrip()[:70]}")
    if not args.apply:
        print("\n（未加 --apply，仅预览）")
        return 0

    new_lines = lines[:start] + [NEW_BODY] + lines[end:]
    TARGET.write_text("".join(new_lines), encoding="utf-8")
    print(f"\n已写入。{len(lines)} 行 -> {len(new_lines)} 行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
