"""清理 workspace 里的过时测试产物。

规则（2026-10-30）：
* 删除：`workspace/` 下、创建时间早于 2026-10-24 的作业；以及 10-25 的 `ab-*` 实验臂
  （结论已写进 README / docs，作业本身是中间产物）；
* 保留：README/文档引用的证据作业（img3/img4/img5 系列）、当前正在跑的作业。

安全：只删 `F:\\3dgs-and-ai\\workspace` 的**直接子目录**，逐个校验绝对路径前缀。
"""
from __future__ import annotations

import datetime
import os
import shutil
import sys
from pathlib import Path

WS = Path(r"F:\3dgs-and-ai\workspace").resolve()
assert str(WS) == r"F:\3dgs-and-ai\workspace", f"工作区路径异常：{WS}"

KEEP = {
    "img5-v1", "img5-quality", "img5-quality2", "img5-quality3", "img5-quality4",
    "img4-v1", "img3-v1", "img3-v2", "img5-2dgs",
}
CUT = datetime.datetime(2026, 10, 24).timestamp()
RUNNING_PREFIX = "ab-dv2"

dry = "--dry-run" in sys.argv


def size_gb(d: Path) -> float:
    s = 0
    for dp, _, fns in os.walk(d):
        for f in fns:
            try:
                s += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return s / 2 ** 30


targets: list[Path] = []
for d in sorted(WS.iterdir()):
    if not d.is_dir() or d.name in KEEP or d.name.startswith(RUNNING_PREFIX):
        continue
    mt = d.stat().st_mtime
    is_old = mt < CUT
    is_ab_arm = d.name.startswith("ab-")
    if is_old or is_ab_arm:
        targets.append(d)

total = sum(size_gb(d) for d in targets)
print(f"待删 {len(targets)} 个作业，共 {total:.1f} GB" + ("（dry-run）" if dry else ""))

freed = 0.0
failed = 0
for d in targets:
    p = d.resolve()
    if not str(p).startswith(str(WS) + os.sep):
        print(f"  跳过（路径越界）：{p}")
        continue
    sz = size_gb(p)
    if dry:
        continue
    try:
        shutil.rmtree(p)
        freed += sz
    except Exception as exc:  # noqa: BLE001
        failed += 1
        print(f"  [失败] {d.name}：{type(exc).__name__}: {exc}")

if not dry:
    print(f"已删除，回收 {freed:.1f} GB；失败 {failed} 个")
    rest = [x for x in WS.iterdir() if x.is_dir()]
    print(f"workspace 剩余 {len(rest)} 个作业，共 {sum(size_gb(x) for x in rest):.1f} GB")
