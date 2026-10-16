"""核对 `README.md` 的参数表是否与**代码里的实际默认值**一致。

为什么需要（用户明确要求）："每次改进成功先去检查文档过时和不符的更新。"
本会话已经出现过 README 写着 `mesh_method=poisson` / `density_quantile=0.0` /
`crop_quantile=0.90`，而代码里分别是 `2dgs` / `0.001` / `0.95`。
**过时的默认值比没有文档更糟** —— 用户会照着它去调参，然后困惑为什么没效果。

⚠️ 本脚本自己也曾出错（第一版正则要求单元格恰好是 `9` 这种裸值，
遇到 `**8**` 这种加粗写法就匹配不到，于是报出 9 项**假**不符）。
所以下面先跑**自检**：对一段构造好的已知表格，检查器必须给出预期的判定。

用法: python scripts/check_readme_params.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 需要核对的参数（README 表格里的行）
KEYS = [
    ("export", "mesh_method"), ("export", "mesh_width"), ("export", "mesh_max_views"),
    ("export", "mesh_poisson_depth"), ("export", "mesh_density_quantile"),
    ("export", "mesh_crop_quantile"), ("export", "mesh_trim_sigma"),
    ("export", "mesh_consistency_min_views"), ("export", "mesh_presimplify_ratio"),
    ("export", "mesh_target_triangles"), ("export", "mesh_smooth_iters"),
    ("export", "mesh_texture"),
    ("train", "iterations"), ("train", "resolution"), ("train", "cap_max"),
    ("train", "densify_until"), ("train", "lambda_lpips"),
    ("train", "lambda_depth_distortion"),
]


def cell(text: str, section: str, key: str) -> str | None:
    """取表格里 `| \\`<section>.<key>\\` | <值> |` 的第 2 列（去掉加粗与反引号）。"""
    pat = re.compile(r"^\|\s*`" + re.escape(f"{section}.{key}") +
                     r"`\s*\|\s*(.+?)\s*\|", re.M)
    m = pat.search(text)
    if not m:
        return None
    v = m.group(1)
    v = v.replace("**", "").replace("`", "").strip()
    return v


def norm(v) -> str:
    """把值归一成可比较的字符串。

    ⚠️ 两个必须处理的等价性（否则会报假不符）：
      * Python 的 `True/False` vs README/TOML 的 `true/false`；
      * **数值等价**：`10.0` 与 `10` 是同一个值（第一版按字符串比，报了假不符）。
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        # 数值统一成最简形式：10.0 -> 10，0.001 -> 0.001
        return f"{float(v):g}"
    return str(v)


def norm_readme(s: str) -> str:
    """README 单元格里的值按同一规则归一（可能写 10 或 10.0）。"""
    s = s.strip()
    try:
        return f"{float(s):g}"
    except (TypeError, ValueError):
        return s.lower()


def selftest() -> int:
    """**已知答案自检**：构造一段表格，检查器必须给出预期结果。"""
    sample = (
        "| 参数 | 默认 | 说明 |\n|---|---|---|\n"
        "| `export.mesh_poisson_depth` | **8** | x |\n"
        "| `export.mesh_texture` | false | x |\n"
        "| `train.lambda_lpips` | 0.0 | x |\n"
    )
    checks = [
        (cell(sample, "export", "mesh_poisson_depth"), "8", "加粗值 **8** 必须解析为 8"),
        (cell(sample, "export", "mesh_texture"), "false", "布尔值"),
        (cell(sample, "train", "lambda_lpips"), "0.0", "浮点 0.0 必须解析出来（不能被当成缺失）"),
        (cell(sample, "export", "not_there"), None, "不存在的键必须返回 None"),
    ]
    bad = 0
    for got, want, why in checks:
        ok = got == want
        print(f"  [{'OK' if ok else '**FAIL**'}] 自检：{why}（got={got!r} want={want!r}）")
        if not ok:
            bad += 1
    return bad


def main() -> int:
    from gsflow.config import DEFAULTS

    print("=== 检查器自检（已知答案）===")
    bad_self = selftest()
    if bad_self:
        print(f"\n**检查器自身有问题（{bad_self} 项）⇒ 下面的结论不可信**")
        return 2

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    print("\n=== README 参数表 vs 代码默认值 ===")
    bad = missing = 0
    for sec, key in KEYS:
        actual = DEFAULTS.get(sec, {}).get(key, "(无此默认值)")
        got = cell(readme, sec, key)
        if got is None:
            print(f"  {sec}.{key:<26} README 里没有这一行        实际={norm(actual)}")
            missing += 1
            continue
        ok = norm_readme(got) == norm(actual)
        if not ok:
            bad += 1
        print(f"  {sec}.{key:<26} README={got:<14} 实际={norm(actual):<12} "
              f"{'OK' if ok else '**不符**'}")
    print(f"\n结论: {bad} 项不符、{missing} 项缺失"
          + ("  ⇒ 需要更新 README" if (bad or missing) else "  ⇒ 一致"))
    return 0 if (bad == 0 and missing == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
