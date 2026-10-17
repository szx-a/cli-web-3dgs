"""CLI 与 GUI 的"拍摄模式"必须产出**完全相同**的配置参数。

用户要求："模式的选则要同步到 gui 和 cli"。
本测试从**三个层面**锁住一致性：

  1. Python 映射函数 ↔ 两个前端（GUI 接口 / CLI）用的是同一份；
  2. GUI 的 `<select>` 选项 == `CHOICES`（否则界面能选到后端不认识的模式）；
  3. **已知答案**：turntable 必须带 `mask.mode=keep`、orbit 必须带 `mask.mode=black`
     —— 这是两种拍法唯一的实质差别，不能被改错。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_choices_are_complete_and_documented() -> None:
    from gsflow.gsdata.capture_mode_params import CHOICES

    assert set(CHOICES) == {"auto", "turntable", "orbit"}, \
        "模式集合变化时必须同步更新 GUI 下拉框、CLI 的 choices、以及文档"


def test_turntable_keeps_background_orbit_blackens() -> None:
    """**已知答案**（本轮审计得出的核心结论，不能被改回去）。

    转台拍法里静态背景是"相机没动"的唯一证据 ⇒ 必须保留（mask.mode=keep）；
    相机绕行时抠背景是安全的 ⇒ mask.mode=black。
    """
    from gsflow.gsdata.capture_mode_params import capture_mode_params

    tt = capture_mode_params("turntable")
    assert tt["mask.mode"] == "keep", (
        "turntable 必须保留背景！否则模型会把'物体转'当成'相机转'，"
        "导致光照/遮挡不自洽、细节糊（见 docs/03 的拍摄模式审计）"
    )
    orb = capture_mode_params("orbit")
    assert orb["mask.mode"] == "black"
    auto = capture_mode_params("auto")
    # auto 不预设 mask.mode：判定后由调用方回填，避免"auto 却偷偷改了行为"
    assert "mask.mode" not in auto, "auto 不应预设 mask.mode（会与自动判定冲突）"
    assert auto["capture.mode"] == "auto"


def test_unknown_mode_raises() -> None:
    from gsflow.gsdata.capture_mode_params import capture_mode_params

    with pytest.raises(ValueError):
        capture_mode_params("handheld")


def test_web_api_matches_python_mapping() -> None:
    """GUI 取到的参数必须与 CLI 用的完全相同（两端同构的核心断言）。"""
    from fastapi.testclient import TestClient

    from gsflow.gsdata.capture_mode_params import CHOICES, capture_mode_params
    from gsflow.web.app import create_app

    client = TestClient(create_app())
    r = client.get("/api/capture_modes")
    assert r.status_code == 200, f"接口不可用：{r.status_code}"
    got = {x["value"]: x["params"] for x in r.json()["choices"]}
    assert set(got) == set(CHOICES)
    for c in CHOICES:
        assert got[c] == capture_mode_params(c), \
            f"模式 {c}：GUI={got[c]} 与 CLI={capture_mode_params(c)} 不一致"


def test_gui_select_options_match_choices() -> None:
    from gsflow.gsdata.capture_mode_params import CHOICES

    html = (ROOT / "src" / "gsflow" / "web" / "static" / "index.html").read_text(
        encoding="utf-8")
    m = re.search(r'id="p-capture-mode"(.*?)</select>', html, re.S)
    assert m, "index.html 里找不到拍摄模式下拉框"
    vals = set(re.findall(r'<option value="([^"]+)"', m.group(1)))
    assert vals == set(CHOICES), f"下拉框选项 {vals} 与 CHOICES {set(CHOICES)} 不一致"


def test_cli_accepts_capture_mode_and_has_same_choices() -> None:
    """CLI 的 `--capture-mode` 必须存在，且 choices 与映射一致。"""
    from gsflow.gsdata.capture_mode_params import CHOICES

    text = (ROOT / "src" / "gsflow" / "cli.py").read_text(encoding="utf-8")
    m = re.search(r'"--capture-mode".*?choices=\[([^\]]+)\]', text, re.S)
    assert m, "cli.py 里找不到 --capture-mode 的定义"
    vals = {v.strip().strip('"') for v in m.group(1).split(",")}
    vals.discard("")                      # 允许空串（未指定）
    assert vals == set(CHOICES), f"CLI choices {vals} 与 CHOICES {set(CHOICES)} 不一致"


def test_auto_detection_matches_known_answers() -> None:
    """自动判定必须在**已知答案**的素材上给出正确结果。

    img3-01 是转台（背景帧间差异 0.34），maskloss-v2 是相机绕行（17.88）。
    若这两个素材不在，跳过（不假装通过）。
    """
    from gsflow.gsdata.capture_mode import detect_from_frames

    cases = {"img3-01": "turntable", "maskloss-v2": "orbit"}
    checked = 0
    for job, want in cases.items():
        jd = ROOT / "workspace" / job
        if not (jd / "01_images_raw").exists():
            continue
        a = detect_from_frames(jd, max_pairs=20)
        if a is None:
            continue
        checked += 1
        assert a.mode == want, f"{job} 应判为 {want}，实际 {a.mode}（ratio={a.ratio:.3f}）"
    if checked == 0:
        pytest.skip("没有可用的已知答案素材")
