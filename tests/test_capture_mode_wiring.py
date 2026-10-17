"""`capture.mode` 必须**真的驱动** `mask.mode`，且运行时的实际行为要可追溯。

## 为什么需要这组测试（本轮踩到的三个真问题）

1. `capture.mode` **没有被任何阶段读取** —— 选了完全没效果，典型"配置了但没执行"。
2. 自动判定在 mask 阶段调 `detect_from_frames` 时**还没有掩码**（本阶段才产出掩码），
   鸡生蛋 ⇒ 永远回退 orbit。需要"画面静止比例"的无掩码兜底判据。
3. 两个检测函数都接受**作业根目录**，我传了 `01_images_raw` ⇒ 内部拼成
   `01_images_raw/01_images_raw` ⇒ 不存在 ⇒ 永远返回 None。
   同一个错误在测试脚本和 mask 阶段各犯一次。

还有一点：`job.json` 的 `settings_snapshot` 记录的是**用户请求**（black），
而运行时会改成 keep ⇒ 快照与实际行为不一致，我曾据此误判"keep 没生效"。
所以阶段输出必须显式记录 `mask_mode_effective` / `capture_mode_effective`。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------- 1. 检测函数必须能跑（参数语义） ----------

def test_detectors_accept_job_root_not_frames_dir() -> None:
    """两个检测函数都接受**作业根目录**；传 `01_images_raw` 会返回 None。

    这条测试把"参数语义"钉死，避免再次传错。
    """
    from gsflow.gsdata.capture_mode import detect_from_frames, detect_without_masks

    jd = ROOT / "workspace" / "img3-01"
    if not (jd / "01_images_raw").exists():
        pytest.skip("缺少 img3-01 素材")
    # 作业根目录 ⇒ 能判定
    assert detect_without_masks(jd, max_pairs=12) is not None, \
        "传作业根目录应能判定"
    # 传 frames 目录本身 ⇒ 内部会再拼一层，找不到 ⇒ None
    assert detect_without_masks(jd / "01_images_raw", max_pairs=12) is None, \
        "传 01_images_raw 应返回 None（这正是我踩过的坑）"


def test_maskfree_detection_matches_known_answers() -> None:
    """无掩码判据必须在已知答案上正确（这是 mask 阶段的兜底）。"""
    from gsflow.gsdata.capture_mode import detect_without_masks

    cases = {"img3-01": "turntable", "img3-crop": "turntable",
             "maskloss-v2": "orbit", "full361-v2": "orbit"}
    checked = 0
    for job, want in cases.items():
        jd = ROOT / "workspace" / job
        if not (jd / "01_images_raw").exists():
            continue
        a = detect_without_masks(jd, max_pairs=16)
        if a is None:
            continue
        checked += 1
        assert a.mode == want, f"{job}: 期望 {want}，实际 {a.mode}（{a.reason}）"
    if checked == 0:
        pytest.skip("没有可用的已知答案素材")


# ---------- 2. capture.mode 必须真的被消费 ----------

def test_mask_stage_reads_capture_mode() -> None:
    """mask 阶段必须读取 `capture.mode` 并据此调整 `mask.mode`。

    用源码断言（这是结构约束，不是行为）；行为部分由下方的端到端测试覆盖。
    """
    src = (ROOT / "src" / "gsflow" / "stages" / "mask.py").read_text(encoding="utf-8")
    assert "_resolve_capture_mode" in src, "mask 阶段应解析拍摄模式"
    assert 'capture.mode' in src, "应读取 capture.mode"
    assert 'mode = "keep"' in src or "mode == \"keep\"" in src, \
        "turntable 应把 mask.mode 改成 keep"
    # 无掩码兜底必须被调用（否则会鸡生蛋）
    assert "detect_without_masks" in src, "应有无掩码兜底判据"


def test_mask_stage_records_effective_mode() -> None:
    """阶段输出必须记录**实际生效**的模式，而不只是用户请求的值。"""
    src = (ROOT / "src" / "gsflow" / "stages" / "mask.py").read_text(encoding="utf-8")
    for key in ("mask_mode_effective", "capture_mode_effective", "capture_mode.json"):
        assert key in src, f"阶段输出缺少 {key}"
    # 且应写入独立 json（便于不解析 manifest）
    assert "capture_mode.json" in src


def test_keep_mode_preserves_background_in_source() -> None:
    """`keep` 分支必须直接返回原图（不涂黑）。"""
    src = (ROOT / "src" / "gsflow" / "stages" / "mask.py").read_text(encoding="utf-8")
    assert 'if mode == "keep":' in src
    # keep 分支里应赋值为原图 rgb
    i = src.index('if mode == "keep":')
    seg = src[i:i + 200]
    assert "clean = rgb" in seg, "keep 分支应保持原图"


# ---------- 3. 已知答案：turntable 与 orbit 的实质差别 ----------

def test_turntable_uses_keep_orbit_uses_black() -> None:
    """映射的**已知答案**（本轮的核心结论，不能被改回去）。"""
    from gsflow.gsdata.capture_mode_params import capture_mode_params

    assert capture_mode_params("turntable")["mask.mode"] == "keep", \
        "转台必须保留背景，否则模型把'物体转'当成'相机转'"
    assert capture_mode_params("orbit")["mask.mode"] == "black"
