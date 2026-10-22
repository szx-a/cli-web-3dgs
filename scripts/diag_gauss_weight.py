"""高斯容量分布与"视觉权重"体检。

## 这个指标是怎么来的

用户反复说"网络依然破碎"。查到最后发现：网格本身已经修好（表面积 1.72 → 18.10），
但**网格只能覆盖高斯真正密集的地方**，而高斯分布极度失衡：

    img3-final（94,598 个高斯）
      贡献 80% 视觉权重的高斯：**4,301 个（4.5%）**
      不透明度 > 0.3 的 24,629 个占视觉权重 73.1%
      高斯尺度：中位最大轴 0.015，p95 0.064，**max 1.31**

也就是说：**约 5% 的高斯在"画"这个模型**，其余 95% 挤在中央雕花区贡献 20% 画面。
于是"按位置"建网格必然只得到那团核心 —— 这不是网格算法的 bug，
而是**容量分配的失衡**。

## 视觉权重怎么定义

单个高斯覆盖的屏幕面积 ∝ 尺度²，乘不透明度 ≈ 它对像素的贡献：

    weight_i = opacity_i × max(scale_i)²

（用最大轴尺度而不只用世界距离，因为 3DGS 的可见贡献主要随投影面积走。）

## 输出什么

* `n80`：贡献 80% 视觉权重所需的高斯数占比 —— **越小说明容量越集中**；
* 位置分位 vs 视觉分位：两者的差距说明"在画面边缘撑场面的其实是大高斯"；
* **网格覆盖的视觉权重比例**：把交付的网格 bbox 与视觉权重对照，
  直接回答"这个网格代表了多少外观"。

用法:
  python scripts/diag_gauss_weight.py --job img3-final
  python scripts/diag_gauss_weight.py --job img3-final --job img3-crop
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def visual_weight(g: dict) -> np.ndarray:
    op = np.asarray(g["opacity"], dtype=np.float64).reshape(-1)
    sc = np.asarray(g["scales"], dtype=np.float64)
    w = op * (sc.max(axis=1) ** 2)
    s = float(w.sum())
    return w / s if s > 0 else w


def weighted_percentiles(v: np.ndarray, w: np.ndarray, ps) -> list[float]:
    idx = np.argsort(v)
    cw = np.cumsum(w[idx])
    return [float(v[idx][min(len(v) - 1, int(np.searchsorted(cw, p)))]) for p in ps]


def analyze(job: str, mesh: Path | None = None) -> dict:
    from gsflow.gsdata.shio import read_gaussians_full

    jd = ROOT / "workspace" / job
    plys = sorted((jd / "04_train" / "point_cloud").glob("*/point_cloud.ply"), key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0))
    out: dict = {"job": job, "ok": False}
    if not plys:
        out["note"] = "没有训练产物"
        return out
    g = read_gaussians_full(plys[-1])
    xyz = np.asarray(g["xyz"], dtype=np.float64)
    op = np.asarray(g["opacity"], dtype=np.float64).reshape(-1)
    sc = np.asarray(g["scales"], dtype=np.float64)
    w = visual_weight(g)
    n = len(xyz)

    idx = np.argsort(-w)
    cw = np.cumsum(w[idx])
    out.update({
        "ok": True, "gaussians": int(n),
        "n80": int(np.searchsorted(cw, 0.8)) + 1,
        "n95": int(np.searchsorted(cw, 0.95)) + 1,
        "scale_median": float(np.median(sc.max(1))),
        "scale_p95": float(np.percentile(sc.max(1), 95)),
        "scale_max": float(sc.max()),
        "opaque_frac": float((op > 0.3).mean()),
        "opaque_weight": float(w[op > 0.3].sum()),
    })
    out["n80_frac"] = out["n80"] / n
    # 位置分位 vs 视觉分位
    out["pos_pct"] = np.percentile(xyz, [1, 50, 99], axis=0).round(3).tolist()
    out["vis_pct"] = [weighted_percentiles(xyz[:, k], w, (0.01, 0.5, 0.99))
                      for k in range(3)]
    out["vis_pct"] = [[round(x, 3) for x in col] for col in out["vis_pct"]]

    # ---- 网格覆盖了多少视觉权重 ----
    if mesh is None:
        mesh = jd / "05_export" / "model_mesh.ply"
    if mesh.exists():
        try:
            import open3d as o3d

            m = o3d.io.read_triangle_mesh(str(mesh))
            v = np.asarray(m.vertices)
            lo, hi = v.min(0), v.max(0)
            inside = np.all((xyz >= lo) & (xyz <= hi), axis=1)
            out["mesh"] = {
                "path": str(mesh.name),
                "surface_area": float(m.get_surface_area()),
                "triangles": int(len(m.triangles)),
                "weight_covered": float(w[inside].sum()),
                "gaussians_inside": int(inside.sum()),
                "bbox": [lo.round(3).tolist(), hi.round(3).tolist()],
            }
        except Exception as exc:  # noqa: BLE001
            out["mesh"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def verdict(r: dict) -> list[str]:
    notes = []
    if not r.get("ok"):
        return ["无法判断（没有训练产物）"]
    f = r["n80_frac"]
    if f < 0.10:
        notes.append(f"**容量高度集中**：只有 {f*100:.1f}% 的高斯贡献 80% 视觉权重 "
                     f"（{r['n80']:,}/{r['gaussians']:,}）。"
                     f"后果：按位置建网格只能覆盖核心，外部结构靠少数大尺度高斯撑着")
    elif f > 0.35:
        notes.append(f"容量较平均（{f*100:.1f}% 的高斯贡献 80% 权重）")
    if r.get("scale_max", 0) > 20 * max(r.get("scale_median", 1e-9), 1e-9):
        notes.append(f"存在**极大尺度**高斯（最大 {r['scale_max']:.3f} vs 中位 "
                     f"{r['scale_median']:.4f}）—— 少量大高斯承担了画面覆盖")
    m = r.get("mesh")
    if isinstance(m, dict) and "weight_covered" in m:
        cov = m["weight_covered"]
        notes.append(f"导出的网格 bbox 覆盖 **{cov*100:.1f}%** 的视觉权重"
                     + ("（偏低 ⇒ 网格只代表模型的一部分）" if cov < 0.85 else "（覆盖良好）"))
    return notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", default=[])
    ap.add_argument("--out", default="workspace/gauss_weight.json")
    args = ap.parse_args()
    jobs = args.job or ["img3-final"]

    print("=== 高斯容量分布 / 视觉权重体检 ===")
    rows = {}
    bad = 0
    for job in jobs:
        r = analyze(job)
        rows[job] = r
        print(f"\n[{job}]")
        if not r.get("ok"):
            print(f"  {r.get('note')}")
            bad += 1
            continue
        print(f"  高斯总数          : {r['gaussians']:,}")
        print(f"  贡献 80% 视觉权重 : {r['n80']:,} 个（{r['n80_frac']*100:.1f}%）")
        print(f"  贡献 95% 视觉权重 : {r['n95']:,} 个（{r['n95']/r['gaussians']*100:.1f}%）")
        print(f"  尺度 中位/p95/max : {r['scale_median']:.4f} / "
              f"{r['scale_p95']:.4f} / {r['scale_max']:.4f}")
        print(f"  不透明度>0.3 占比 : {r['opaque_frac']*100:.1f}%"
              f"（占视觉权重 {r['opaque_weight']*100:.1f}%）")
        m = r.get("mesh")
        if isinstance(m, dict) and "surface_area" in m:
            print(f"  网格              : {m['triangles']:,} 面，表面积 "
                  f"{m['surface_area']:.2f}，覆盖视觉权重 {m['weight_covered']*100:.1f}%，"
                  f"包住 {m['gaussians_inside']:,} 个高斯")
        for n in verdict(r):
            print(f"  ⇒ {n}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"\n已写出 {out}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
