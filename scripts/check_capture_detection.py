"""检查 `capture_mode` 的两条判据（有掩码 / 无掩码）在**已知答案**上的表现。

为什么需要：`detect_without_masks` 是本轮新加的兜底判据，
它必须在已知答案上给出正确结果，否则就是"永远返回 None"的假工具。
已知答案：
  * img3-01 / img3-crop / img3-tt：**转台**（物体转、相机固定）
  * maskloss-v2 / full361-v2 / final-v1：**相机绕行**

用法: python scripts/check_capture_detection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KNOWN = {
    "img3-01": "turntable",
    "img3-crop": "turntable",
    "img3-tt": "turntable",
    "maskloss-v2": "orbit",
    "full361-v2": "orbit",
    # 注：`final-v1` 等"训练产物作业"本身没有 02_masks（掩码来自其数据源作业），
    # 所以不作为判据的已知答案 —— 否则会得到"缺掩码"的假失败。
}


def main() -> int:
    from gsflow.gsdata.capture_mode import detect_from_frames, detect_without_masks

    bad = 0
    checked = 0
    print(f"  {'作业':<14}{'期望':<11}{'有掩码判据':<12}{'无掩码判据':<12}")
    print("  " + "-" * 50)
    for job, want in KNOWN.items():
        # ⚠️ 两个函数都接受**作业根目录**（内部会拼 `01_images_raw`），
        # 不是 `01_images_raw` 本身。我第一次传错，得到的全是 "None"，
        # 还以为是函数 bug，查了好几轮 —— 教训：**工具返回空值时先怀疑自己传错参数**。
        jd = ROOT / "workspace" / job
        if not (jd / "01_images_raw").exists():
            continue
        a1 = detect_from_frames(jd, max_pairs=20)
        a2 = detect_without_masks(jd, max_pairs=20)
        m1 = a1.mode if a1 else "(无)"
        m2 = a2.mode if a2 else "(无)"
        ok1 = "OK" if m1 == want else "**错**"
        ok2 = "OK" if m2 == want else "**错**"
        print(f"  {job:<14}{want:<11}{m1 + ' ' + ok1:<12}{m2 + ' ' + ok2:<12}")
        checked += 1
        if m1 != want:
            bad += 1
            if a1:
                print(f"      有掩码理由：{a1.reason}")
        if m2 != want:
            bad += 1
            if a2:
                print(f"      无掩码理由：{a2.reason}")

    if checked == 0:
        print("\n没有可检查的素材")
        return 1
    total = checked * 2
    print(f"\n结论: {total - bad}/{total} 条判据与已知答案一致"
          + ("" if bad == 0 else f"（**{bad} 条不符**）"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
