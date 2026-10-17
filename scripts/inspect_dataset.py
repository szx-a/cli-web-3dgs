"""评估一个新素材是否值得训练：角度覆盖、冗余、掩码质量、稀疏点。

在花 20~60 分钟训练之前先做这个判断（用户要求"先看数据"）。

与 `estimate_quality_ceiling.py` 的区别：那个是**训练后**审计既有模型；
本脚本是**训练前**的前置检查，只看数据本身。

判据（每条都能给出可操作结论）：
  * **注册率**  < 90% ⇒ 素材本身难重建
  * **重投影误差** > 1.5px ⇒ 位姿不可靠，网格必碎
  * **角度覆盖**  缺口 > 30° ⇒ 那部分是外推，网格/纹理必然糊
  * **角度冗余**  > 3× ⇒ 大量视角拍同一面，有效信息少
  * **前景占比**  < 5% ⇒ 物体太小，像素信息不足；> 60% ⇒ 掩码可能没抠干净
  * **掩码一致性**：前景占比的变异系数过大 ⇒ 分割不稳定
  * **稀疏点/视图** < 20 ⇒ 纹理太弱或视角太相似，初始化会很差

用法: scripts\\gpu_python.cmd scripts\\inspect_dataset.py --job img3-01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--compare", default="", help="逗号分隔的对照作业")
    args = ap.parse_args()

    from gsflow.paths import apply_runtime_env
    apply_runtime_env()

    from PIL import Image

    from gsflow.trainers.gsplat_trainer import find_model_dir, load_colmap_data
    from gsflow.gsdata.colmap_io import read_model

    jobs = [args.job] + [j.strip() for j in args.compare.split(",") if j.strip()]
    results = []
    for job in jobs:
        jd = ROOT / "workspace" / job
        # COLMAP 数据（frames/mask/colmap 阶段后就有 03_colmap，但 load_colmap_data 要的是去畸变目录）
        rep = jd / "03_colmap" / "report.json"
        info = {}
        if rep.exists():
            try:
                info = json.loads(rep.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                info = {}
        # 稀疏模型
        n_reg, n_pts, err = None, None, None
        try:
            # ⚠️ 两个都踩过的坑：
            #   1) `find_model_dir` 要的是**去畸变根目录**，传 `03_colmap/sparse` 会再拼一层；
            #   2) COLMAP mapper 常生成**多个模型**（sparse/0、1、2…），其中只有一个是完整的。
            #      实测 img3-01：sparse/0 只有 47.9 KB（几帧碎片）、sparse/2 是 6,997 KB 的主模型。
            #      取"第一个 cameras.bin"会得到注册 2 帧的假象（我因此误报"注册率 0.5%"）。
            # ⇒ 必须遍历所有候选并按**注册图像数**选最优。
            colmap_dir = jd / "03_colmap"
            cands = sorted({p.parent for p in colmap_dir.rglob("cameras.bin")}) \
                if colmap_dir.exists() else []
            best = None
            for c in cands:
                try:
                    r = read_model(c)
                except Exception:  # noqa: BLE001
                    continue
                if best is None or len(r.images) > len(best[1].images):
                    best = (c, r)
            if best:
                md, rec = best
                n_reg = len(rec.images)
                n_pts = len(rec.points3D) if hasattr(rec, "points3D") else None
                errs = [im.error for im in rec.images if getattr(im, "error", None)]
                if errs:
                    err = float(np.mean(errs))
                if len(cands) > 1:
                    print(f"  [提示] {job}: 发现 {len(cands)} 个 COLMAP 模型，"
                          f"按注册数选了 {md.name}（{n_reg} 帧）")
                # 角度覆盖：用相机中心
                cs = []
                for im in rec.images:
                    r, t = im.cam_to_world()
                    cs.append(t)
                cs = np.asarray(cs)
                c = cs.mean(0)
                v = cs - c
                az = (np.degrees(np.arctan2(v[:, 2], v[:, 0])) + 360) % 360
                azs = np.sort(az)
                gaps = np.diff(np.concatenate([azs, [azs[0] + 360]]))
                gap = float(gaps.max())
                bins = np.floor(az / 10).astype(int)
                occ = int((np.bincount(bins) > 0).sum())
                redundancy = n_reg / max(occ, 1)
                radius = float(np.linalg.norm(v, axis=1).mean())
                yspan = float(cs[:, 1].max() - cs[:, 1].min())
            else:
                gap = redundancy = radius = yspan = None
        except Exception as e:  # noqa: BLE001
            print(f"  [警告] {job} 读稀疏模型失败：{type(e).__name__}: {e}")
            gap = redundancy = radius = yspan = None

        # 掩码
        mask_dir = jd / "02_masks"
        fracs = []
        if mask_dir.exists():
            files = sorted(mask_dir.glob("*.png"))
            step = max(1, len(files) // 60)
            for p in files[::step]:
                try:
                    with Image.open(p) as im:
                        fracs.append(float((np.asarray(im.convert("L")) > 127).mean()))
                except Exception:  # noqa: BLE001
                    pass
        fg = float(np.mean(fracs)) if fracs else None
        fg_cv = float(np.std(fracs) / max(np.mean(fracs), 1e-9)) if fracs else None
        size = None
        imgs = sorted((jd / "01_images_raw").glob("*")) if (jd / "01_images_raw").exists() else []
        if imgs:
            try:
                with Image.open(imgs[0]) as im:
                    size = im.size
            except Exception:  # noqa: BLE001
                pass

        results.append(dict(job=job, n_reg=n_reg, n_pts=n_pts, err=err, gap=gap,
                            redundancy=redundancy, radius=radius, yspan=yspan,
                            fg=fg, fg_cv=fg_cv, size=size, n_frames=len(imgs)))

    # ---- 输出 ----
    for r in results:
        print(f"\n=== {r['job']} ===")
        print(f"  图像：{r['n_frames']} 张，尺寸 {r['size']}")
        print(f"  注册：{r['n_reg']} 张" +
              (f"，平均重投影 {r['err']:.3f} px" if r['err'] is not None else ""))
        print(f"  稀疏点：{r['n_pts']}" +
              (f"（每视图 {r['n_pts']/max(r['n_reg'],1):.0f}）" if r['n_pts'] and r['n_reg'] else ""))
        if r["gap"] is not None:
            print(f"  角度覆盖：缺 {r['gap']:.1f}°（覆盖 {360-r['gap']:.1f}°/360°）")
            print(f"  角度冗余：{r['redundancy']:.1f}×（每 10° 扇区平均拍 "
                  f"{r['n_reg']/max(360/10,1):.1f} 张）")
            print(f"  绕行半径 {r['radius']:.2f}，相机高度跨度 {r['yspan']:.2f}")
        if r["fg"] is not None:
            print(f"  前景占比：{r['fg']*100:.1f}%（变异系数 {r['fg_cv']:.2f}）")

        # ---- 判据 ----
        issues, oks = [], []
        if r["n_reg"] and r["n_frames"]:
            rate = r["n_reg"] / r["n_frames"] * 100
            (oks if rate >= 90 else issues).append(f"注册率 {rate:.1f}%")
        if r["err"] is not None:
            (oks if r["err"] <= 1.5 else issues).append(f"重投影 {r['err']:.3f}px")
        if r["gap"] is not None:
            (oks if r["gap"] <= 30 else issues).append(f"角度缺口 {r['gap']:.1f}°")
        if r["redundancy"] is not None:
            (oks if r["redundancy"] <= 3 else issues).append(f"角度冗余 {r['redundancy']:.1f}×")
        if r["fg"] is not None:
            ok = 0.05 <= r["fg"] <= 0.6
            (oks if ok else issues).append(f"前景 {r['fg']*100:.1f}%")
        if r["fg_cv"] is not None:
            (oks if r["fg_cv"] <= 0.5 else issues).append(f"掩码稳定性 CV={r['fg_cv']:.2f}")
        if r["n_pts"] and r["n_reg"]:
            ppv = r["n_pts"] / r["n_reg"]
            (oks if ppv >= 20 else issues).append(f"稀疏点/视图 {ppv:.0f}")

        if oks:
            print(f"  [OK]   {' | '.join(oks)}")
        if issues:
            print(f"  [注意] {' | '.join(issues)}")
        r["_issues"] = issues

    if len(results) == 2:
        a, b = results
        print("\n=== 两素材对比 ===")
        for key, label, better in (("err", "重投影误差(px)", "低"),
                                   ("gap", "角度缺口(°)", "低"),
                                   ("redundancy", "角度冗余", "低"),
                                   ("n_pts", "稀疏点", "高"),
                                   ("fg", "前景占比", "—")):
            va, vb = a.get(key), b.get(key)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                winner = ""
                if better == "低":
                    winner = " ⇒ " + (a["job"] if va < vb else b["job"]) + " 更好"
                elif better == "高":
                    winner = " ⇒ " + (a["job"] if va > vb else b["job"]) + " 更好"
                print(f"  {label:<16} {a['job']}={va:.3f}   {b['job']}={vb:.3f}{winner}")

    n_issue = sum(1 for r in results if r["_issues"])
    print(f"\n结论: {len(results) - n_issue}/{len(results)} 个素材无明显问题")
    return 0


if __name__ == "__main__":
    sys.exit(main())
