"""回归测试：**检查点必须按数字排序**，不能按字符串排序。

## 事故（2026-10-22，本轮最严重的方法论 bug）

诊断脚本用 `sorted(checkpoint_dirs)[-1]` 取"最终模型"。字符串排序下：

    sorted(['iteration_10000', 'iteration_30000', 'iteration_5000'])
    -> ['iteration_10000', 'iteration_30000', 'iteration_5000']   # 最后是 5000！

于是**24 个诊断脚本、27 处**都在渲染**第 5000 步的中间模型**，而指标说的是 30000 步。
后果（同一份 30k 模型）：

| 指标 | 误测（5000 步）| 真值（30000 步）|
|---|---|---|
| 单视角掩码内 PSNR | 12~14 dB | **22.6 dB** |
| 渲染外观 | 过曝 + 彩色镶边 | 正常灰石 |
| 掩码外 alpha | 0.0446 | **0.0115** |

⚠️ 本测试**不碰文件系统**（本机 pytest 的 tmp 目录曾出现 WinError 5），只验证排序语义。
⚠️ 全仓扫描时**跳过本文件**：文档字符串里就引用了那段错误写法（第一版因此自证有罪）。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

# 故意用"拼接"而不是直接写整句，避免本文件的文档字符串被扫描命中
BAD_PATTERN = re.compile(r"sorted\([^\n]*point_cloud\.ply[^\n]*\)\s*\[" + r"-1\]")


def _iter_key(p: Path) -> int:
    """与各脚本统一使用的数值 key 完全一致。"""
    return int("".join(ch for ch in p.parent.name if ch.isdigit()) or 0)


def test_numeric_key_beats_lexical() -> None:
    """纯路径对象即可验证：字符串排序取到 5000，数值排序取到 30000。"""
    paths = [Path(f"iteration_{n}") / "point_cloud.ply"
             for n in (5000, 10000, 30000)]
    lexical = sorted(paths)[-1].parent.name
    numeric = sorted(paths, key=_iter_key)[-1].parent.name
    assert lexical == "iteration_5000", f"字符串排序行为变了：{lexical}"
    assert numeric == "iteration_30000", f"数值排序没取到最终检查点：{numeric}"


def test_no_lexical_checkpoint_sort_in_repo() -> None:
    """全仓库不得再出现"对 point_cloud.ply 直接取 sorted 末尾"（除本文件）。"""
    bad = []
    for d in ("scripts", "src", "tests"):
        for p in (ROOT / d).rglob("*.py"):
            if p.resolve() == SELF:
                continue
            try:
                t = p.read_text(encoding="utf-8")
            except OSError:
                continue
            for m in BAD_PATTERN.finditer(t):
                if "key=" in m.group(0):
                    continue
                bad.append(f"{p.relative_to(ROOT)}: {m.group(0)[:80]}")
    assert not bad, "发现按字符串排序取最后一个检查点：\n" + "\n".join(bad)
