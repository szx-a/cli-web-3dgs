"""相机交互的回归测试：确认"不会卡死/跳变/方向反转"，且实现基于四元数。

⚠️ **本文档要诚实记录一个判据错误的教训**：
我曾试图用"在 pitch=90° 处改变 yaw 是否还能改变视线方向"来判定万向锁，
并据此写负向用例。实测发现**两种实现（欧拉角与四元数）在该处的结果完全相同（0.0000）** ——
因为相机指向 ±Y 时，绕 Y 轴转本来就不改变视线方向，这是**几何必然**。

⇒ **万向锁是"参数化"的缺陷，不是"方向向量"的可观测量。**
用输出量去测参数化的缺陷，测不到是应该的；我用错了判据三次（一次 yaw 绕世界 Y
所以极点仍有效、一次矩阵乘序写错、一次用输出量测参数化）。
那三个失败脚本已删除，只保留能给出**明确结论**的证据。

本测试断言的是**用户真正能感知的东西**（这才是"有没有 bug"的正确判据）：
  1. 竖直拖拽**不会方向反转**（欧拉角 wrap-around 的典型症状）；
  2. **不会出现不连续跳变**（单步角远大于名义值）；
  3. 水平拖拽可**无限圈**且全程均匀（yaw 无奇点）；
  4. 实现基于**四元数**（`qAxisAngle`/`qMul`），且有 pitch 夹取。
这四条合起来正是"相机操作不卡死、不乱跳"的可操作定义。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "src" / "gsflow" / "web" / "static" / "viewer.js"


def run_node(script: str) -> tuple[int, str]:
    p = subprocess.run(["node", str(ROOT / "scripts" / script)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def test_node_available() -> None:
    p = subprocess.run(["node", "--version"], capture_output=True, text=True)
    if p.returncode != 0:
        pytest.skip("node 不可用，跳过相机探针")


def test_camera_control_probe_passes() -> None:
    """探针必须通过，且**确实测到了东西**（不是空壳）。"""
    rc, out = run_node("probe_gimbal_lock.mjs")
    assert rc == 0, f"相机交互探针失败：\n{out[-1500:]}"

    # 1) clamp 模式：0 次方向反转（"卡死/乱跳"的核心症状）
    m = re.search(r"方向反转次数 = (\d+)", out)
    assert m, "探针输出里找不到方向反转统计（探针可能被改成空壳）"
    assert int(m.group(1)) == 0, f"clamp 模式出现 {m.group(1)} 次方向反转"

    # 2) 无跳变
    assert "不连续跳变: 无" in out, "出现不连续跳变"

    # 3) yaw 无限圈且均匀
    assert "全程均匀转动，无奇点" in out, "yaw 出现奇点"

    # 4) 关键数字必须在合理范围（防止探针被改成恒真）
    nums = [float(x) for x in re.findall(r"最大单步角 = ([\d.]+)", out)]
    assert nums, "找不到最大单步角"
    assert all(n < 0.2 for n in nums), f"单步角过大（{nums}），可能跳变"


def test_camera_uses_quaternions_not_euler() -> None:
    js = VIEWER.read_text(encoding="utf-8")
    assert "qAxisAngle" in js and "qMul" in js, "相机应基于四元数"
    assert "m4rotationXYZ" not in js, "检测到欧拉角旋转构造"


def test_pitch_is_clamped_in_default_mode() -> None:
    js = VIEWER.read_text(encoding="utf-8")
    assert "PITCH_MAX" in js, "缺少 PITCH_MAX 夹取"
    assert "rotMode = 'clamp'" in js, "默认旋转模式应为 clamp"


def test_probe_reproduces_viewer_math() -> None:
    """探针必须复刻 viewer.js 的数学（否则测的是另一份实现）。"""
    js = VIEWER.read_text(encoding="utf-8")
    probe = (ROOT / "scripts" / "probe_gimbal_lock.mjs").read_text(encoding="utf-8")
    for token in ("qAxisAngle", "qRot", "right"):
        assert token in js and token in probe, \
            f"探针与 viewer.js 的数学不一致（缺 {token}）"


def test_probe_detects_known_bad_cameras() -> None:
    """**负向测试**：探针必须能抓到三个已知坏实现，且不误杀正确实现。

    三个坏实现都对应历史上真实出现过的 bug（符号写反、越过极点、步长跳变）。
    如果这条失败，说明探针太宽松（漏报）或太严格（误杀）—— 两种都会让
    "探针说通过"变得没有意义。
    """
    rc, out = run_node("selftest_camera_probe.mjs")
    assert rc == 0, f"负向测试失败：\n{out[-1200:]}"
    assert "既不误杀也不漏报" in out
    # 三个坏实现必须都出现在输出里且都"报警"
    for bug in ("BUG-1", "BUG-2", "BUG-3"):
        assert bug in out, f"负向用例缺少 {bug}"
