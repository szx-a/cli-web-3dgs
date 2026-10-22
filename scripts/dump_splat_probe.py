"""把 `.splat` 的解析/协方差计算与 NumPy 交叉验证 —— 防"椭球朝向全错"。

## 为什么必须交叉验证

泼溅渲染的质量全在 **3×3 世界协方差** M = R·diag(s²)·Rᵀ 上：
  * M 算错 ⇒ 椭球朝向/拉伸全错，画面会像"炸开的碎片"，而不是连续表面；
  * 四元数顺序错（w,x,y,z 写成 x,y,z,w）⇒ 同样是朝向全错；
  * 尺度没平方 ⇒ 椭球尺寸差一个量级。

而这些**在无浏览器环境下看不出来**。所以用 NumPy 独立算一遍，
与 JS 实现逐元素比对。任何一处偏大就说明实现有问题。

用法: python scripts/dump_splat_probe.py --job img3-crop
      node scripts/probe_splat_cov.mjs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """四元数 (w,x,y,z) → 旋转矩阵。"""
    w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="img3-crop")
    ap.add_argument("--n", type=int, default=8, help="取前多少个高斯做比对")
    args = ap.parse_args()

    from gsflow.gsdata.shio import read_gaussians_full
    from gsflow.gsdata.splat_io import gaussians_to_splat

    job = ROOT / "workspace" / args.job
    plys = sorted((job / "04_train" / "point_cloud").glob("*/point_cloud.ply"), key=lambda p: int(''.join(ch for ch in p.parent.name if ch.isdigit()) or 0))
    if not plys:
        print("找不到点云 ply")
        return 1
    g = read_gaussians_full(plys[-1])
    data = gaussians_to_splat(g)
    out = ROOT / "env" / "tmp" / "test.splat"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"已写出 {out}（{g['n']} 个高斯，{len(data)} 字节）")

    # 计算期望的协方差（NumPy 版），连同原始属性一起交给 JS 比对
    #
    # ⚠️ 必须用**量化后**的四元数算期望值：
    # `.splat` 把四元数每分量压成 1 字节（`q*128+128`），量化步长约 1/128。
    # 我第一版用**未量化**的四元数算期望，于是 8 个样本全部"不符"
    # （相对误差 2%~26%）。那不是 JS 的 bug，是**比对基准错了** ——
    # 四元数量化会让旋转差几度，近各向同性的协方差上相对误差会被放大。
    def quantize_quat(q: np.ndarray) -> np.ndarray:
        qn = q / max(np.linalg.norm(q), 1e-12)
        b = np.clip(qn * 128.0 + 128.0, 0, 255).astype(np.uint8)
        dq = (b.astype(np.float64) - 128.0) / 128.0
        return dq / max(np.linalg.norm(dq), 1e-12)

    cases = []
    for i in range(min(args.n, g["n"])):
        s = np.asarray(g["scales"][i], dtype=np.float64)
        q = np.asarray(g["quats"][i], dtype=np.float64)
        qq = quantize_quat(q)          # 与文件里一致
        R = quat_to_R(qq)
        M = R @ np.diag(s * s) @ R.T
        cases.append({
            "idx": int(i),
            "scale": [float(v) for v in s],
            "quat_raw": [float(v) for v in q],
            "quat_quantized": [float(v) for v in qq],
            # 与 JS 同序：[xx, xy, xz, yy, yz, zz]
            "expect_cov": [float(M[0, 0]), float(M[0, 1]), float(M[0, 2]),
                           float(M[1, 1]), float(M[1, 2]), float(M[2, 2])],
            "expect_center": [float(v) for v in np.asarray(g["xyz"][i])],
            # 量化本身的相对步长，用于判断"允许多大误差"
            "quant_rel": float(np.max(np.abs(qq - q / max(np.linalg.norm(q), 1e-12)))),
        })
    probe = ROOT / "env" / "tmp" / "splat_probe.json"
    probe.write_text(json.dumps({
        "splat_path": str(out),
        "count": int(g["n"]),
        "bytes": len(data),
        "cases": cases,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {probe}（{len(cases)} 个比对样本）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
