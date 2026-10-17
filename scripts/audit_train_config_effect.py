"""审计全 workspace 的训练作业：多少作业的**实际**致密化判据与配置不符。

## 为什么要做这次审计

G109 的 bug（孤儿 `elif` 让 grad2d 每步被 3D 梯度覆盖）不是某个作业的特例 ——
它从打完那个补丁起，对**每一个**把 `densify_criterion` 设成 `grad2d` 的作业都生效。
而我在那段时间里跑了几十组"同数据对照"，并据此写下了结论。

所以必须回答：**哪些历史结论还站得住，哪些必须作废？**

判据是"组内各臂的有效判据是否一致"：
  * 若一组 A/B 的**所有**臂都退回了 grad3d ⇒ 组内仍然一致，
    该组的**相对**结论（如"antialiased 比 classic 差"）依然成立，只是标签写错了；
  * 若一组里有的臂被退回、有的没有 ⇒ 该组对比**无效**；
  * 专门比较 grad2d vs grad3d 的组 ⇒ **一律无效**。

用法: python scripts/audit_train_config_effect.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workspace/train_config_audit.json")
    args = ap.parse_args()

    import check_train_used_config as C

    ws = ROOT / "workspace"
    jobs = sorted(p.name for p in ws.iterdir()
                  if p.is_dir() and (p / "04_train" / "metrics.json").exists())
    rows = []
    for j in jobs:
        r = C.analyze(j)
        rows.append({"job": j, "requested": r.get("requested"),
                     "effective": r.get("effective"),
                     "fallback_warn": r.get("fallback_warn"),
                     "psnr": r.get("psnr"), "gaussians": r.get("gaussians")})

    affected = [r for r in rows if (r["fallback_warn"] or 0) > 0]
    clean = [r for r in rows if (r["fallback_warn"] or 0) == 0]
    print("=== 训练作业审计 ===")
    print(f"  有训练产物的作业      : {len(rows)}")
    print(f"  **受 G109 影响**（退回 grad3d）: {len(affected)}")
    print(f"  未受影响              : {len(clean)}")

    # 按作业名前缀分组（我的 A/B 命名约定是 `<组名>-<臂>` 或 `<组名><臂>`
    # 同组共享前缀），看哪些组是"组内混合"（=对比无效）
    def group_of(name: str) -> str:
        for sep in ("-", "_"):
            if sep in name:
                return name.split(sep)[0]
        # 无分隔符时按末尾字母/数字剥离（如 b2/a2）
        return name.rstrip("0123456789").rsplit("-", 1)[0] or name

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[group_of(r["job"])].append(r)

    mixed, uniform_bad, all_clean = [], [], []
    for g, items in groups.items():
        states = {bool((it["fallback_warn"] or 0) > 0) for it in items}
        if len(items) < 2:
            continue
        if states == {True}:
            uniform_bad.append(g)
        elif states == {False}:
            all_clean.append(g)
        else:
            mixed.append(g)

    print(f"\n  多臂分组（>=2 个作业）: {len(mixed) + len(uniform_bad) + len(all_clean)}")
    print(f"    组内**全部**退回 grad3d（相对结论仍成立，标签需更正）: "
          f"{len(uniform_bad)}")
    if uniform_bad:
        print("      " + ", ".join(sorted(uniform_bad)[:14]))
    print(f"    组内**混合**（对比无效，必须重跑）: {len(mixed)}")
    if mixed:
        for g in sorted(mixed)[:10]:
            print(f"      {g}: " + ", ".join(sorted(i["job"] for i in groups[g])[:6]))
    print(f"    组内全部干净: {len(all_clean)}")

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({
        "total": len(rows), "affected": len(affected), "clean": len(clean),
        "groups_mixed": sorted(mixed), "groups_uniformly_affected": sorted(uniform_bad),
        "groups_clean": sorted(all_clean), "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
