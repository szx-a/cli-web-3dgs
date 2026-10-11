"""对比 TrainConfig 的字段、config.DEFAULTS['train'] 的键、以及后端实际注入的键。

动机（本轮找到的根因）：`GsplatBackend.run` 逐个参数手写 TrainConfig(...)，
且用的是 `ctx.cfg("k", d) or d` 这种写法 —— **`or` 会把合法的 0/False 吞掉**。
实测：配置里 `opacity_reset_interval = 0`（关闭重置），但
`int(0 or 3000)` = 3000，于是"关闭重置"这个修复从未生效。
（这是本项目**第三处**默认值来源，前两处是 config.DEFAULTS 与 TrainConfig dataclass。）

本脚本把三份清单摆出来，用于改成"按 dataclass 字段循环注入"而不丢字段。

用法：env\\venv\\Scripts\\python.exe scripts\\check_trainconfig_wiring.py
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gsflow.config import DEFAULTS  # noqa: E402
from gsflow.trainers.gsplat_trainer import TrainConfig  # noqa: E402

fields = {f.name: f for f in dataclasses.fields(TrainConfig)}
cfg_keys = set(DEFAULTS["train"].keys())

src = (ROOT / "src" / "gsflow" / "trainers" / "gsplat_backend.py").read_text(encoding="utf-8")
injected = set(re.findall(r"(\w+)\s*=\s*(?:int|float|bool|str)?\(?ctx\.cfg\(", src))
# 上面的正则只抓关键字名：改成抓 "name=...ctx.cfg" 形式
injected = set(re.findall(r"(\w+)\s*=\s*[^,\n]*ctx\.cfg\(", src))

print(f"TrainConfig 字段 {len(fields)} 个")
print(f"DEFAULTS['train'] 键 {len(cfg_keys)} 个")
print(f"后端显式注入 {len(injected)} 个\n")

missing = sorted(set(fields) - injected - {"data_dir", "out_dir", "iterations"})
print("**后端未显式注入字段**（只能用 dataclass 默认值，改配置无效）：")
for k in missing:
    print(f"    {k:32s} dataclass 默认 = {fields[k].default!r}"
          + (f"   config 默认 = {DEFAULTS['train'][k]!r}" if k in cfg_keys else "   (config 无此键)"))

print("\n**config 有键但 dataclass 没有**（配置会被静默忽略）：")
for k in sorted(cfg_keys - set(fields)):
    print(f"    {k:32s} config = {DEFAULTS['train'][k]!r}")

print("\n**两边都有但默认值不同**（很可能就是漂移/drift）：")
for k in sorted(cfg_keys & set(fields)):
    a, b = DEFAULTS["train"][k], fields[k].default
    if a != b:
        print(f"    {k:32s} config={a!r}  dataclass={b!r}")

print("\n**后端用 `or` 会吞掉合法 0/False 的位置**：")
for i, line in enumerate(src.splitlines(), 1):
    if "ctx.cfg(" in line and " or " in line:
        print(f"    行 {i}: {line.strip()}")
