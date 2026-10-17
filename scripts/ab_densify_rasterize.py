"""同一份数据上的**多臂 × 多重复**对照实验，带 95% 置信区间。

## 为什么必须重做

上一轮我在 `docs/03` 里记了：
  * `grad2d` vs `grad3d`：`fix-new` 602,166 高斯 vs `fix-old` 142,492 高斯
  * `antialiased`：27.72 vs 28.79 ⇒ **−1.07 dB**

但后来发现训练器里有一个**孤儿 `elif`**，让 `grad2d` 每步算完就被 3D 梯度覆盖
（两个作业各打了 30000 次假告警）。也就是说：

  * "grad2d 更好"这个结论是**在 grad2d 根本没生效的情况下**得出的；
  * `antialiased` 那组对比同样跑在被打坏的致密化上。

⇒ 这两条结论**都不可信，必须重测**。本脚本就是重测工具。

## 方法学（都不是可选项）

1. **同一份 `data`**：所有臂共享同一个 `04_train/data`，否则分辨率/内参不同，
   比较无意义（曾经因此得出过假结论）。
2. **重复实验**：单次运行的噪声 σ≈0.03–0.05 dB。只跑一次就下结论，
   等于把噪声当效应。
3. **报 95% 置信区间**，不报裸差值。区间跨 0 就是"没测出差异"。
4. **自证参数真的生效**：把实际生效的配置打出来，并断言各臂取值不同
   （避免"配置传进去了、功能没执行"这一类静默失败）。

用法:
  python scripts/ab_densify_rasterize.py --reps 3 --iters 15000
  python scripts/ab_densify_rasterize.py --arms g2d,g3d --reps 2
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 每臂的配置覆盖（都是 `train.*` 的键）
ARMS: dict[str, dict[str, str]] = {
    # 官方判据（2D 屏幕梯度累积，阈值 2e-4 是官方默认）
    "g2d": {"train.densify_criterion": "grad2d",
            "train.densify_grad2d_threshold": "2e-4"},
    # 旧路径：3D 梯度 + 同一个数值阈值（量纲并不匹配，这是要检验的点）
    "g3d": {"train.densify_criterion": "grad3d",
            "train.densify_grad2d_threshold": "2e-4"},
    # 阈值标定：`img3-final` 的致密化**早期激活率只有 9%**（见
    # `diag_densify_activation.py`），低于社区典型 20~60% ⇒ 怀疑 2e-4 偏高。
    # 1e-4 是官方默认的一半，若激活率与最终质量都上升，说明该调低。
    "g2d1e4": {"train.densify_criterion": "grad2d",
               "train.densify_grad2d_threshold": "1e-4"},
    "g2d5e5": {"train.densify_criterion": "grad2d",
               "train.densify_grad2d_threshold": "5e-5"},
    # Mip-Splatting 抗锯齿光栅化（上次测出 −1.07 dB，需复核）
    "aa": {"train.densify_criterion": "grad2d",
           "train.densify_grad2d_threshold": "2e-4",
           "train.rasterize_mode": "antialiased"},
    # 背景是否参与训练。
    #
    # 用户："现在我看到的东西就像曝光一样"。实测模型里 **13.5% 的亮高斯占了
    # 62.8% 的视觉权重**（`scripts/diag_bright_background.py`），因为转台 + `keep`
    # 模式把整片亮背景也拟合成高斯了。
    #
    # `mask_loss_mode=inside` 只在物体掩码内算损失 ⇒ 背景不再被拟合，
    # 但 COLMAP 仍然"看得到"背景（掩码只作用于训练损失，不作用于注册）。
    # 注意：全图 PSNR 会下降（背景故意不拟合），要看**物体区域的画质**与
    # **背景视觉权重占比**，不能只看全图 PSNR。
    "inside": {"train.mask_loss_mode": "inside"},
}
BASELINE = "g2d"


def ensure_shared_data(src_job: str, work: str) -> Path:
    """把源作业的 `04_train/data` 复制到共享目录（幂等）。"""
    dst = ROOT / "workspace" / work / "data"
    if (dst / "images").exists() and any((dst / "images").iterdir()):
        n = sum(1 for p in (dst / "images").iterdir() if p.is_file())
        print(f"复用共享数据：{dst}（{n} 张）")
        return dst
    src = ROOT / "workspace" / src_job / "04_train"
    if not (src / "data" / "images").exists():
        raise SystemExit(f"源作业没有 04_train/data：{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src / "data", dst)
    shutil.copy2(ROOT / "workspace" / src_job / "04_train" / "undistort_info.json",
                 dst.parent / "undistort_info.json")
    n = sum(1 for p in (dst / "images").iterdir() if p.is_file())
    print(f"已复制共享数据：{dst}（{n} 张）")
    return dst


def ci95(vals: list[float]) -> tuple[float, float, float]:
    """返回 (均值, 下界, 上界)。单样本时区间退化为均值。"""
    import numpy as np
    from scipy import stats

    a = np.asarray(vals, dtype=np.float64)
    m = float(a.mean())
    if len(a) < 2:
        return m, m, m
    se = float(a.std(ddof=1) / np.sqrt(len(a)))
    h = float(stats.t.ppf(0.975, len(a) - 1) * se)
    return m, m - h, m + h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-from", default="img3-final",
                    help="提供 04_train/data 的源作业")
    ap.add_argument("--work", default="ab-r41", help="共享数据与各臂输出的前缀目录")
    ap.add_argument("--arms", default="g2d,g3d,aa")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--reuse", action="store_true",
                    help="复用已完成的运行结果（断点续跑）")
    args = ap.parse_args()

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arm_names if a not in ARMS]
    if unknown:
        print(f"未知臂：{unknown}（可选 {sorted(ARMS)}）")
        return 2

    from gsflow.config import load as load_settings
    from gsflow.events import EventBus
    from gsflow.paths import apply_runtime_env
    from gsflow.pipeline.context import StageContext
    from gsflow.trainers.gsplat_backend import GsplatBackend

    apply_runtime_env()
    data_dir = ensure_shared_data(args.data_from, args.work)

    results: dict[str, list[dict]] = {a: [] for a in arm_names}
    for arm in arm_names:
        for rep in range(1, args.reps + 1):
            job = f"{args.work}-{arm}-r{rep}"
            job_dir = ROOT / "workspace" / job
            out_dir = job_dir / "04_train"
            res_p = out_dir / "metrics.json"
            if args.reuse and res_p.exists():
                try:
                    d = json.loads(res_p.read_text(encoding="utf-8"))
                    results[arm].append({"job": job, "psnr": d.get("psnr"),
                                         "gaussians": d.get("gaussians")})
                    print(f"[{arm} r{rep}] 复用 {d.get('psnr')} dB / "
                          f"{d.get('gaussians')} 高斯")
                    continue
                except Exception:  # noqa: BLE001
                    pass
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            params = dict(ARMS[arm])
            params["train.iterations"] = str(args.iters)
            params["train.eval_interval"] = str(args.iters)
            settings = load_settings(dict(params))
            ctx = StageContext(job_id=job, job_dir=job_dir, settings=settings,
                               events=EventBus(), params=params)
            # ⚠️ 必须设置 `_stage`：`ctx.cfg("k")` 会先拼成 `train.k`。
            # 不设置时前缀解析路径与生产环境不一致，表现为"覆盖看起来传进去了、实际没生效"。
            ctx._stage = "train"

            # 自证：把实际生效的值打出来（不信任"我以为传进去了"）
            eff = {k.replace("train.", ""): ctx.cfg(k.replace("train.", ""), None)
                   for k in ARMS[arm]}
            print(f"\n===== [{arm} r{rep}] {job} =====")
            print("  生效: " + "  ".join(f"{k}={v}" for k, v in eff.items()), flush=True)

            t0 = time.time()
            res = GsplatBackend().run(ctx, data_dir=data_dir, out_dir=out_dir,
                                      iterations=args.iters)
            res = res if isinstance(res, dict) else {}
            print(f"[{arm} r{rep}] PSNR={res.get('psnr')} dB  "
                  f"高斯={res.get('gaussians')}  {time.time()-t0:.0f}s", flush=True)
            results[arm].append({"job": job, "psnr": res.get("psnr"),
                                 "gaussians": res.get("gaussians")})

    # ---------------- 汇总 ----------------
    print("\n=== 汇总（同一份数据，仅被测项不同）===")
    print(f"{'臂':<6}{'n':>3}{'PSNR 均值':>12}{'95% CI':>22}{'Δ vs 基线':>12}"
          f"{'Δ 95% CI':>22}{'高斯(均值)':>12}")
    stats_out: dict[str, dict] = {}
    base_vals = [r["psnr"] for r in results.get(BASELINE, []) if r["psnr"] is not None]
    for arm in arm_names:
        vals = [r["psnr"] for r in results[arm] if r["psnr"] is not None]
        if not vals:
            print(f"{arm:<6}{0:>3}  （无有效结果）")
            continue
        m, lo, hi = ci95(vals)
        g = [r["gaussians"] for r in results[arm] if r["gaussians"]]
        gm = sum(g) / len(g) if g else 0
        if arm == BASELINE or len(base_vals) < 2 or len(vals) < 2:
            d_s = "—"
            d_ci = "—"
            delta = None
        else:
            import numpy as np
            delta = m - (sum(base_vals) / len(base_vals))
            # 独立样本的差值标准误（Welch）
            se = float(np.sqrt(np.var(vals, ddof=1) / len(vals)
                               + np.var(base_vals, ddof=1) / len(base_vals)))
            from scipy import stats as _st
            # Welch–Satterthwaite 自由度
            v1 = np.var(vals, ddof=1) / len(vals)
            v2 = np.var(base_vals, ddof=1) / len(base_vals)
            df = (v1 + v2) ** 2 / ((v1 ** 2 / (len(vals) - 1))
                                   + (v2 ** 2 / (len(base_vals) - 1)))
            h = float(_st.t.ppf(0.975, df) * se)
            d_s = f"{delta:+.3f}"
            d_ci = f"[{delta-h:+.3f},{delta+h:+.3f}]"
        print(f"{arm:<6}{len(vals):>3}{m:>12.3f}{f'[{lo:.3f},{hi:.3f}]':>22}"
              f"{d_s:>12}{d_ci:>22}{gm:>12,.0f}")
        stats_out[arm] = {"n": len(vals), "mean": m, "ci": [lo, hi],
                          "delta_vs_base": delta,
                          "gaussians": gm,
                          "runs": results[arm]}

    print("\n判读规则：**95% 置信区间跨 0 ⇒ 没测出差异**，不要当成「有效」。")
    print(f"噪声参照：单次运行 σ≈0.03–0.05 dB（此前实测），n={args.reps} 次时"
          f"均值标准误约 {0.04/ (args.reps ** 0.5):.3f} dB。")

    outp = ROOT / "workspace" / args.work / "ab_densify_rasterize.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"iters": args.iters, "reps": args.reps,
                                "baseline": BASELINE, "arms": ARMS,
                                "stats": stats_out}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\n已写出 {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
