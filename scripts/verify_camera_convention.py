"""验证相机朝向约定：相机到底看向哪边？—— 用三种独立证据交叉确认。

起因：用户说"我拍的确实是一圈"，而我的工具报"只覆盖 247.8°、缺 112.2°"。
然后我用"相机中心均值"当物体中心，算出"相机看向反方向"（点积 −0.813），
但那可能只是**中心取错了**（相机轨迹若是弧，中心均值会偏到弦上）。

本脚本用三种互相独立的证据判断朝向/覆盖，任何单一证据都可能是错的：

  证据 1【正交性与行列式】R 必须正交、det=+1。若 det=−1 说明有反射，坐标系被镜像。
  证据 2【看向稀疏点云】用 `points3D` 的质心当物体中心 —— 这是**真实几何位置**，
          不是相机位置的统计量。dot(相机→点云质心, 相机前向) 应接近 +1。
  证据 3【渲染自检】用真实位姿渲染，与真值比 PSNR。若位姿约定错了，
          渲染必然极差（这是**端到端**的验证，不依赖任何坐标约定假设）。

再加一项：把"方位角覆盖"分别按 **XZ 平面**（Y 上）与 **其他两轴** 计算，
确认没有选错竖直轴。

用法: scripts\\gpu_python.cmd scripts\\verify_camera_convention.py --job img3-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def angular_coverage(cs: np.ndarray, axis: int) -> tuple[float, float, float]:
    """按"绕 axis 轴的方位角"计算覆盖。返回 (覆盖度, 最大缺口, 冗余)。"""
    a = [i for i in range(3) if i != axis]
    c = cs.mean(0)
    v = cs - c
    az = (np.degrees(np.arctan2(v[:, a[1]], v[:, a[0]])) + 360) % 360
    azs = np.sort(az)
    gaps = np.diff(np.concatenate([azs, [azs[0] + 360]]))
    gap = float(gaps.max())
    occ = int((np.bincount(np.floor(az / 10).astype(int)) > 0).sum())
    return 360 - gap, gap, len(cs) / max(occ, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--views", type=int, default=8)
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    from gsflow.gsdata.colmap_io import read_model

    job = ROOT / "workspace" / args.job
    # 取注册数最多的模型（COLMAP 会产生碎片模型，取第一个会误判）
    cands = sorted({p.parent for p in (job / "03_colmap").rglob("cameras.bin")})
    best = None
    for c in cands:
        try:
            r = read_model(c)
        except Exception:  # noqa: BLE001
            continue
        if best is None or len(r.images) > len(best[1].images):
            best = (c, r)
    if best is None:
        print("找不到 COLMAP 模型")
        return 1
    md, rec = best
    print(f"模型 {md.relative_to(job)}：{len(rec.images)} 帧，{len(rec.points3D)} 稀疏点\n")

    Rs, cs, fwds = [], [], []
    for im in rec.images:
        R, t = im.cam_to_world()
        Rs.append(R)
        cs.append(t)
        fwds.append(-R[:, 2])       # 相机看向 -Z
    Rs = np.asarray(Rs)
    cs = np.asarray(cs)
    fwds = np.asarray(fwds)

    # ---- 证据 1：正交性与 det ----
    print("=== 证据 1：旋转矩阵正交性 ===")
    orth = np.abs(np.einsum("nij,nik->njk", Rs, Rs) - np.eye(3)).max()
    dets = np.linalg.det(Rs)
    print(f"  |RᵀR − I| 最大偏差 = {orth:.2e}（应 < 1e-6）")
    print(f"  det(R) 范围 = [{dets.min():.4f}, {dets.max():.4f}]（应全为 +1）")
    if orth > 1e-4 or dets.min() < 0.9:
        print("  ⇒ **旋转矩阵有问题**，后续结论不可信")
    else:
        print("  ⇒ 旋转矩阵正常（无镜像、无退化）")

    # ---- 证据 2：看向稀疏点云 ----
    print("\n=== 证据 2：相机是否看向稀疏点云（真实几何位置）===")
    cen = rec.points3D.mean(0) if len(rec.points3D) else cs.mean(0)
    v = cen - cs
    vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    d1 = (vn * fwds).sum(1)
    print(f"  稀疏点云质心 = {np.round(cen, 3)}")
    print(f"  dot(相机→点云质心, 相机前向)：均值 {d1.mean():+.3f}  "
          f"p10 {np.quantile(d1,0.1):+.3f}  p90 {np.quantile(d1,0.9):+.3f}")
    if d1.mean() > 0.7:
        print("  ⇒ 相机确实看向物体（约定正确）")
    elif d1.mean() < -0.7:
        print("  ⇒ **相机'前向'符号取反了** —— 应改为 +Z")
        fwds = -fwds
    else:
        print("  ⇒ 方向不一致（部分帧看向别处？）")

    # 用相机位置均值再算一遍（对比，说明为什么这个中心不可靠）
    v2 = cs.mean(0) - cs
    v2n = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9)
    d2 = (v2n * fwds).sum(1)
    print(f"  （若用'相机位置均值'当中心：{d2.mean():+.3f}"
          f" ← 轨迹是弧时该值会偏，不是可靠的判据）")

    # ---- 证据 3：三个轴的方位角覆盖 ----
    print("\n=== 证据 3：方位角覆盖（分别按三个轴，确认竖直轴没选错）===")
    for axis, label in ((1, "Y（假设竖直）"), (0, "X"), (2, "Z")):
        cov, gap, red = angular_coverage(cs, axis)
        print(f"  绕 {label:<12} 覆盖 {cov:6.1f}°/360°  缺口 {gap:6.1f}°  冗余 {red:5.1f}×")
    print("  （对'绕物体水平转一圈'的拍摄，**竖直轴**上的缺口应当最小）")

    # ---- 证据 4：渲染自检（端到端，不依赖坐标约定）----
    print("\n=== 证据 4：用真实位姿渲染并与真值比 PSNR（端到端验证）===")
    try:
        import torch
        from gsplat import rasterization

        from gsflow.gsdata.shio import read_gaussians_full
        from gsflow.trainers.gsplat_trainer import _load_images, load_colmap_data
        from gsflow.stages.export import ExportStage

        # 需要去畸变数据 + 训练好的模型；若还没有就跳过
        train_dir = job / "04_train"
        if not (train_dir / "data").exists() or not (train_dir / "metrics.json").exists():
            print("  （还没有训练产物，跳过 —— 这一项要训练后才可用）")
        else:
            metrics = __import__("json").loads(
                (train_dir / "metrics.json").read_text(encoding="utf-8"))
            ply = Path(metrics["best_ply"])
            g = read_gaussians_full(ply)
            d = load_colmap_data(train_dir / "data", downscale=1,
                                 white_background=False, flip_camera=False)
            imgs, _ = _load_images(train_dir / "data", d["names"], None)
            ks = torch.from_numpy(d["ks"]).float()
            vms = torch.linalg.inv(torch.from_numpy(d["c2ws"]).float())
            dev = "cuda"
            psnrs = []
            step = max(1, len(imgs) // args.views)
            for i in range(0, len(imgs), step):
                gt = imgs[i].to(dev)
                H, W = gt.shape[:2]
                with torch.no_grad():
                    out = rasterization(
                        means=torch.as_tensor(g["xyz"], device=dev),
                        quats=torch.as_tensor(g["quats"], device=dev),
                        scales=torch.as_tensor(g["scales"], device=dev),
                        opacities=torch.as_tensor(g["opacity"], device=dev).reshape(-1).clamp(0, 1),
                        colors=torch.as_tensor(g["shs"], device=dev),
                        viewmats=vms[i][None].to(dev), Ks=ks[i][None].to(dev),
                        width=W, height=H, sh_degree=int(g["sh_degree"]),
                        near_plane=0.01, far_plane=1e10, packed=False, render_mode="RGB")
                r = out[0][0].clamp(0, 1)
                mse = float(torch.mean((r - gt) ** 2).clamp_min(1e-10))
                psnrs.append(10 * np.log10(1 / mse))
            print(f"  渲染 PSNR：均值 {np.mean(psnrs):.2f} dB "
                  f"（范围 {min(psnrs):.2f}~{max(psnrs):.2f}）")
            if np.mean(psnrs) > 18:
                print("  ⇒ 位姿约定正确（若错，渲染会是噪声级 < 10 dB）")
            else:
                print("  ⇒ **渲染很差，位姿或模型有问题**")
    except Exception as e:  # noqa: BLE001
        print(f"  （渲染自检失败：{type(e).__name__}: {e}）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
