"""把 `mesh_poisson_depth` 的同模型对照结果写进文档（幂等追加）。

数据来自 4 组同模型对照（2 个模型 × 2 个训练时长），结论是一个**稳定的权衡**：
depth=8 恒定把相邻面法线翻转率降低约 10 倍，代价是轮廓 IoU 让出 ~0.01、
深度误差中位数让出 ~0.5pp。此前文档只写了"depth=8 更好"，不够精确。

用法: python scripts/record_depth_tradeoff.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "03-踩坑记录.md"
MARKER = "### G86."

BODY = """### G86. `mesh_poisson_depth=8` 的**精确代价**：4 组同模型对照
G83 只报了"depth=8 把翻转率降 10 倍"，容易被读成"无条件更好"。补做同模型对照后，
它是一个**稳定的权衡**（每行都是同一模型、同一份点云，唯一变量是 depth）：

| 模型 | 迭代 | depth | 翻转率 | 轮廓 IoU | 深度中位数 |
|---|---|---|---|---|---|
| final-v1 | 8000 | **8** | **2.5%** | 0.739 | 5.75% |
| final-v1 | 8000 | 9 | 24.5% | **0.753** | **5.39%** |
| dd-15 | 20000 | **8** | **2.2%** | 0.780 | 7.64% |
| dd-15 | 20000 | 9 | 22.4% | **0.793** | **7.04%** |

⇒ **规律稳定，与训练时长无关**：
* 翻转率：depth=8 恒定降低约 **10 倍**（24.5%→2.5%、22.4%→2.2%）；
* 轮廓 IoU：depth=8 恒定让出 **~0.01**（0.753→0.739、0.793→0.780）；
* 深度中位数：depth=8 恒定让出 **~0.5pp**。

**所以 depth=8 不是"更好"，而是"用 ~0.01 IoU + 0.5pp 深度换 10 倍法线质量"。**
这个交换在当前目标下是**值得的**（用户最初的抱怨是"破碎"，而破碎=法线/自交问题，
不是轮廓覆盖问题），但如果某个下游任务更看重轮廓精度，应当把它调回 9。

**方法学教训**：我一开始只在 `dd-15` 上比过一次就写进文档，
后来在 `final-v1` 上量到 IoU 0.739（明显低于 dd-15 的 0.780）时差点误判成
"depth=8 在新模型上退化"—— 实际那是**不同模型**的差异。
**跨模型比较会把"模型差异"错算成"参数效应"**，这是本会话反复出现的同一类错误（见 G74）。"""


def main() -> int:
    text = DOC.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"已存在 {MARKER!r}，跳过（幂等）")
        return 0
    DOC.write_text(text.rstrip("\n") + "\n\n" + BODY.strip("\n") + "\n", encoding="utf-8")
    print(f"已追加 G86 到 {DOC}（{BODY.count(chr(10))} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
