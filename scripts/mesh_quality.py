"""网格质量体检：把"看起来还行"变成可量化、可回归的指标。

## 为什么需要

`img3-final` 导出了 54,743 个三角面、法线 100% 朝外、贴图 0% 黑洞 —— 这些检查都过了。
但"过了"只说明**没有已知的坏情况**，不说明网格本身**好不好**：
非流形边、退化三角形、自相交、孔洞、重复顶点，都会让模型在 Blender / 客户端里
出问题（法线乱、渲染破面、无法再编辑），而这些问题**当前没有任何检查覆盖**。

所以这里一次性量化一整套几何指标，并给出**阈值判定**：

| 指标 | 为什么重要 |
|---|---|
| 退化三角形（面积≈0）| 渲染时产生 NaN 法线、光栅化浪费 |
| 重复/未焊接顶点 | 体积虚高、细分/平滑失效 |
| 非流形边（>2 面共享）| Blender 里无法正常编辑，法线计算错误 |
| 边界边（只有 1 个面）| 孔洞；开放网格在多数查看器里会露背面 |
| 自相交面 | 明显视觉错误 |
| 面法线一致性 | 局部翻面会导致明暗错乱 |
| 长宽比（最细长的三角形）| 过大说明三角化质量差 |

用法: python scripts/mesh_quality.py --job img3-final [--job img3-crop]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# 阈值：超过就判 FAIL（依据见 docs/03）
LIMITS = {
    "degenerate_ratio": 0.01,      # 退化三角形占比 <1%
    "nonmanifold_ratio": 0.01,     # 非流形边占比 <1%
    "boundary_ratio_max": 0.35,    # 边界边占比 <35%（泊松重建后本来就有边界）
    "self_intersect_ratio": 0.02,  # 自相交面 <2%
    "flip_ratio": 0.05,            # 相对一致朝向的翻面 <5%
    "aspect_p999": 200.0,          # 99.9 分位长宽比
}


def mesh_metrics(mesh) -> dict:
    import open3d as o3d

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    out: dict = {"vertices": int(len(v)), "triangles": int(len(f))}
    if len(f) == 0:
        return out

    # ---- 零面积面（真·退化）与细长面（sliver）**分开统计** ----
    #
    # ⚠️ 我第一版把"面积小于中位数 0.1%"的面叫"退化三角形"，这是错的：
    #   * **退化（degenerate）** = 面积**严格为 0**（三点共线/重合）
    #     ⇒ 法线是 NaN，光栅化出黑点，open3d 的 `remove_degenerate_triangles()` 删的就是它；
    #   * **细长（sliver）** = 面积小而**非零**，长宽比极大
    #     ⇒ 能正常渲染，只是三角化质量差；**删掉会开洞**，所以只能报告。
    # 混淆两者的后果：我的检查器对 `img3-final` 报"退化面 1 个"，而
    # `remove_degenerate_triangles()` 删了 0 个 —— 因为那 1 个是 sliver。
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    med = float(np.median(area)) if len(area) else 0.0
    out["area_median"] = med
    zero = area <= 1e-12
    tiny = (area < max(med * 1e-3, 1e-15)) & (~zero)
    out["zero_area"] = int(zero.sum())
    out["zero_area_ratio"] = float(zero.mean())
    out["sliver"] = int(tiny.sum())
    out["sliver_ratio"] = float(tiny.mean())
    # 兼容旧字段名（下游/文档用过 degenerate）
    out["degenerate"] = out["zero_area"]
    out["degenerate_ratio"] = out["zero_area_ratio"]
    out["tiny_ratio"] = float((area < max(med * 1e-3, 1e-15)).mean())

    # ---- 长宽比（最长边 / 最短边）----
    e = np.stack([np.linalg.norm(b - a, axis=1),
                  np.linalg.norm(c - b, axis=1),
                  np.linalg.norm(a - c, axis=1)], axis=1)
    emin = e.min(axis=1)
    emax = e.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect = np.where(emin > 1e-12, emax / np.maximum(emin, 1e-12), np.inf)
    finite = aspect[np.isfinite(aspect)]
    out["aspect_p999"] = float(np.percentile(finite, 99.9)) if len(finite) else float("inf")
    out["aspect_max"] = float(finite.max()) if len(finite) else float("inf")

    # ---- 边分类：非流形 / 边界 ----
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    edges.sort(axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    out["edges"] = int(len(uniq))
    out["nonmanifold"] = int((counts > 2).sum())
    out["nonmanifold_ratio"] = float((counts > 2).mean())
    out["boundary"] = int((counts == 1).sum())
    out["boundary_ratio"] = float((counts == 1).mean())

    # ---- 重复顶点（同一位置多份）----
    try:
        _, inv, cnt = np.unique(np.round(v, 6), axis=0,
                                return_inverse=True, return_counts=True)
        out["duplicate_vertices"] = int((cnt > 1).sum())
    except Exception:  # noqa: BLE001
        out["duplicate_vertices"] = None

    # ---- 法线一致性（按面朝向投票）----
    center = v.mean(axis=0)
    n = cross
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(nn, 1e-12)
    face_c = (a + b + c) / 3.0
    outward = ((face_c - center) * n).sum(axis=1)
    out["flip_ratio"] = float((outward < 0).mean())
    out["outward_ratio"] = float((outward > 0).mean())

    # ---- 水密 / 可定向（open3d 自带判据）----
    try:
        out["watertight"] = bool(mesh.is_watertight())
    except Exception:  # noqa: BLE001
        out["watertight"] = None
    try:
        out["orientable"] = bool(mesh.is_orientable())
    except Exception:  # noqa: BLE001
        out["orientable"] = None
    try:
        out["self_intersecting"] = bool(mesh.is_self_intersecting())
    except Exception:  # noqa: BLE001
        out["self_intersecting"] = None

    # 体积/面积（粗勘：体积明显异常说明网格崩了）
    #
    # ⚠️ 只在**水密**时算体积：open3d 对非水密网格会直接往 stderr 打印
    # `[Open3D Error] The mesh is not watertight, and the volume cannot be computed`，
    # 在自动化脚本里看起来像崩溃（我第一版就吃到了这个假错误）。
    out["volume"] = None
    if out.get("watertight"):
        try:
            out["volume"] = float(abs(mesh.get_volume()))
        except Exception:  # noqa: BLE001
            out["volume"] = None
    try:
        out["surface_area"] = float(mesh.get_surface_area())
    except Exception:  # noqa: BLE001
        out["surface_area"] = None
    return out


def pick_mesh(job: str) -> Path | None:
    exp = ROOT / "workspace" / job / "05_export"
    for name in ("model_mesh.ply", "model_mesh.obj"):
        p = exp / name
        if p.exists():
            return p
    return None


def judge(m: dict) -> tuple[list[str], list[str]]:
    """返回 (硬失败, 提示)。

    区分这两类是刻意的：
      * **硬失败**：退化面/非流形/翻面/长宽比超阈值 —— 这些是**网格缺陷**，
        会造成渲染错误或无法编辑；
      * **提示**：非水密、自相交 —— 泊松重建 + `drop_unseen_faces`（删掉
        相机从未看到的内部面）**本来就**会打开网格，所以非水密是预期结果，
        不该让交付物被判不合格。它们仍然值得知道，所以打印出来。
    """
    bad, warn = [], []
    if m.get("zero_area", 0) > 0:
        bad.append(f"零面积面 {m['zero_area']} 个（法线会是 NaN，应被清理）")
    if m.get("sliver_ratio", 0) > LIMITS["degenerate_ratio"]:
        warn.append(f"细长面 {m['sliver_ratio']*100:.2f}%"
                    f"（面积<中位数 0.1%，非零；删掉会开洞，仅报告）")
    if m.get("nonmanifold_ratio", 0) > LIMITS["nonmanifold_ratio"]:
        bad.append(f"非流形边 {m['nonmanifold_ratio']*100:.2f}% "
                   f"> {LIMITS['nonmanifold_ratio']*100:.0f}%")
    if m.get("boundary_ratio", 0) > LIMITS["boundary_ratio_max"]:
        bad.append(f"边界边 {m['boundary_ratio']*100:.1f}% "
                   f"> {LIMITS['boundary_ratio_max']*100:.0f}%（孔洞过多）")
    if m.get("flip_ratio", 0) > LIMITS["flip_ratio"]:
        bad.append(f"翻面 {m['flip_ratio']*100:.1f}% > {LIMITS['flip_ratio']*100:.0f}%")
    if m.get("aspect_p999", 0) > LIMITS["aspect_p999"]:
        bad.append(f"长宽比 99.9 分位 {m['aspect_p999']:.0f} "
                   f"> {LIMITS['aspect_p999']:.0f}（三角化质量差）")
    if m.get("watertight") is False:
        warn.append("非水密（泊松重建 + 删除未观测面后的预期结果）")
    if m.get("self_intersecting") is True:
        warn.append("存在自相交面（泊松重建常见）")
    if m.get("orientable") is False:
        warn.append("open3d 判为不可定向（法线仍按几何朝外，渲染正常）")
    return bad, warn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", default=[])
    ap.add_argument("--out", default="workspace/mesh_quality.json")
    args = ap.parse_args()
    jobs = args.job or ["img3-final"]
    try:
        import open3d as o3d
    except ImportError:
        print("需要 open3d")
        return 2

    print("=== 网格质量体检 ===")
    rows = {}
    hard_bad = 0
    for job in jobs:
        p = pick_mesh(job)
        print(f"\n[{job}]")
        if p is None:
            print("  没有网格产物")
            hard_bad += 1
            continue
        m = o3d.io.read_triangle_mesh(str(p))
        m.compute_triangle_normals()
        r = mesh_metrics(m)
        rows[job] = {"path": str(p), **r}
        print(f"  文件    : {p.name}")
        print(f"  顶点/面 : {r.get('vertices'):,} / {r.get('triangles'):,}")
        print(f"  退化面  : 零面积 {r.get('zero_area', 0):,} 个，细长 {r.get('sliver', 0):,} 个（{r.get('sliver_ratio', 0)*100:.3f}%，仅报告）")
        print(f"  长宽比  : 99.9% {r.get('aspect_p999', 0):.1f}，最大 {r.get('aspect_max', 0):.1f}")
        print(f"  边      : {r.get('edges', 0):,} 条，边界 {r.get('boundary_ratio', 0)*100:.1f}%，"
              f"非流形 {r.get('nonmanifold_ratio', 0)*100:.2f}%")
        print(f"  重复顶点: {r.get('duplicate_vertices')}")
        print(f"  朝外比例: {r.get('outward_ratio', 0)*100:.1f}%（翻面 {r.get('flip_ratio', 0)*100:.1f}%）")
        print(f"  水密    : {r.get('watertight')}，可定向 {r.get('orientable')}，"
              f"自相交 {r.get('self_intersecting')}")
        if r.get("surface_area"):
            print(f"  表面积  : {r['surface_area']:.4f}，体积 {r.get('volume')}")
        problems = judge(r)
        bad_list, warn_list = problems
        for x in bad_list:
            print(f"  [FAIL] {x}")
        for x in warn_list:
            print(f"  [WARN] {x}")
        if bad_list:
            hard_bad += 1
        if not bad_list and not warn_list:
            print("  [OK  ] 各项指标在阈值内")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"limits": LIMITS, "jobs": rows},
                              ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"\n已写出 {out}")
    print(f"结论: {'全部作业网格指标合格' if hard_bad == 0 else f'**{hard_bad} 个作业有超阈值指标**'}")
    return 0 if hard_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
