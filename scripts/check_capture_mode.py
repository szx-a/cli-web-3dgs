"""检查 `capture_mode` 模块：判定两种拍法并给出训练策略。

用法: python scripts/check_capture_mode.py [--job img3-01] [--job maskloss-v2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", default=[])
    args = ap.parse_args()
    jobs = args.job or ["img3-01", "maskloss-v2", "full361-v2"]

    from gsflow.gsdata.capture_mode import detect_from_frames, training_policy

    bad = 0
    for j in jobs:
        jd = ROOT / "workspace" / j
        if not (jd / "01_images_raw").exists():
            print(f"  [跳过] {j}：没有 01_images_raw")
            continue
        a = detect_from_frames(jd)
        if a is None:
            print(f"  [跳过] {j}：无法判定（帧太少或缺掩码）")
            continue
        pol = training_policy(a.mode)
        print(f"\n=== {j} ===")
        print(f"  判定：**{a.mode}**")
        print(f"  背景差异 {a.bg_diff:.2f} / 前景差异 {a.fg_diff:.2f} "
              f"= {a.ratio:.4f}（{a.n_pairs} 对帧）")
        print(f"  理由：{a.reason}")
        print(f"  训练策略：训练时是否把背景合成黑底 = "
              f"{pol['train_composite_background']}")
        print(f"    为什么：{pol['why']}")
        print(f"    导出：{pol['export_hint']}")

        # 断言：已知答案的素材必须判对
        if j == "img3-01" and a.mode != "turntable":
            print("  **FAIL** img3-01 是转台拍法（背景差异仅 0.48），判定错误")
            bad += 1
        if j in ("maskloss-v2", "full361-v2") and a.mode != "orbit":
            print(f"  **FAIL** {j} 是相机绕行（背景差异 ~20），判定错误")
            bad += 1

    print(f"\n结论: {'判定与已知答案一致' if bad == 0 else f'**{bad} 项不符**'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
