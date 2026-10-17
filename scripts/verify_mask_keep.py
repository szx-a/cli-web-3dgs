"""验证 `mask.mode=keep` 是否真的**保留了原始背景**（而不是仍被涂黑）。

`capture.mode=turntable` 的实质是让 mask 阶段改用 `keep`。但如果 `keep`
实际没生效（例如仍走黑底合成），整条"转台修复"就是空的。

判据：
  * 原始帧的背景区域（掩码外）**亮度中位数** 与 清理后图像的同一区域比较；
  * `keep` ⇒ 两者几乎相同（差异 < 5）；
  * `black` ⇒ 清理后该区域应接近 0。

用法: python scripts/verify_mask_keep.py --job tt-auto3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--frames", default="0,5,10", help="逗号分隔帧序号（按排序后的 01_images_raw）")
    args = ap.parse_args()

    from PIL import Image

    job = ROOT / "workspace" / args.job
    raw_dir = job / "01_images_raw"
    clean_dir = job / "02_images_clean"
    mask_dir = job / "02_masks"
    for d in (raw_dir, clean_dir, mask_dir):
        if not d.exists():
            print(f"缺少目录 {d}")
            return 1

    raw = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))

    # 判断该作业用的是哪种清理模式（从 job.json 读；读不到就按未知处理）。
    # ⚠️ 第一版我无条件按 "keep 未生效" 报错，结果把**正确使用 black 模式的作业**
    # （img3-01，背景 232→0）判成失败 —— 判据必须按**实际配置的模式**期望不同的行为。
    mode = ""
    try:
        import json
        # **优先读阶段实际生效值**（`02_masks/capture_mode.json`）。
        # 为什么：`job.json` 的 `settings_snapshot` 记录的是**用户请求**的
        # `mask.mode`（例如 black），而 `capture.mode=turntable` 会在运行时
        # 把它改成 keep ⇒ 快照与实际行为不一致。我曾据此误判"keep 没生效"。
        eff = mask_dir / "capture_mode.json"
        if eff.exists():
            em = json.loads(eff.read_text(encoding="utf-8"))
            mode = str(em.get("mask_mode_effective", "") or "")
        if not mode:
            job_json = job / "job.json"
            if job_json.exists():
                data = json.loads(job_json.read_text(encoding="utf-8"))
                snap = data.get("settings_snapshot") or {}
                mode = str((snap.get("mask") or {}).get("mode", "") or "")
    except Exception:  # noqa: BLE001
        mode = ""
    expect = "keep" if mode == "keep" else ("black" if mode == "black" else "?")
    print(f"  mask.mode = {mode or '(未记录)'} ⇒ 期望背景"
          f"{'保持不变' if expect == 'keep' else ('被涂黑' if expect == 'black' else '未知')}")

    print(f"  {'帧':<6}{'背景亮度(原)':>14}{'背景亮度(清理后)':>18}{'前景亮度(清理后)':>18}")
    print("  " + "-" * 58)
    bad = 0
    for fi in [int(x) for x in args.frames.split(",") if x.strip()]:
        if fi >= len(raw):
            continue
        p = raw[fi]
        cp = clean_dir / p.name
        if not cp.exists():
            cp = clean_dir / (p.stem + ".jpg")
        mp = None
        for c in (mask_dir / (p.stem + ".png"), mask_dir / (p.name + ".png")):
            if c.exists():
                mp = c
                break
        if not cp.exists() or mp is None:
            print(f"  {fi:<6}（缺清理图或掩码）")
            continue
        with Image.open(p) as im:
            a = np.asarray(im.convert("L"), dtype=np.float32)
        with Image.open(cp) as im:
            b = np.asarray(im.convert("L"), dtype=np.float32)
        with Image.open(mp) as im:
            m = np.asarray(im.convert("L")) > 127
        if b.shape != a.shape:
            with Image.open(cp) as im:
                b = np.asarray(im.convert("L").resize(
                    (a.shape[1], a.shape[0]), Image.NEAREST), dtype=np.float32)
        if m.shape != a.shape:
            m = np.asarray(Image.fromarray((m.astype(np.uint8) * 255)).resize(
                (a.shape[1], a.shape[0]), Image.NEAREST)) > 127
        bg = ~m
        bg_raw = float(np.median(a[bg])) if bg.any() else float("nan")
        bg_cl = float(np.median(b[bg])) if bg.any() else float("nan")
        fg_cl = float(np.median(b[m])) if m.any() else float("nan")
        print(f"  {fi:<6}{bg_raw:>14.1f}{bg_cl:>18.1f}{fg_cl:>18.1f}")

        if expect == "keep":
            if abs(bg_cl - bg_raw) > 5.0:
                print(f"        ⇒ **keep 未生效**：背景应保持 {bg_raw:.0f}，实际 {bg_cl:.0f}")
                bad += 1
        elif expect == "black":
            if bg_cl > 5.0 and bg_raw > 20:
                print(f"        ⇒ **black 未生效**：背景应为 0，实际 {bg_cl:.0f}")
                bad += 1

    print()
    if expect == "?":
        print("  结论: 无法确定该作业的模式（job.json 里没有 mask.mode），未做判定")
        return 1
    if bad == 0:
        print(f"  结论: 行为与 mask.mode={expect} 的预期一致")
        return 0
    print(f"  结论: **{bad} 帧行为与 mask.mode={expect} 不符**")
    return 1


if __name__ == "__main__":
    sys.exit(main())
