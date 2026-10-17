"""回归测试：泼溅渲染在 WebGL1 下的**索引类型契约**，以及"黑屏必须自报"。

## 真实故障（用户两次报"还是黑屏"）

`splat.js` 用 `gl.UNSIGNED_INT` 索引 + `drawElementsInstancedANGLE` 绘制，
但**从未启用 `OES_element_index_uint`**。

WebGL1 规范：没有该扩展时传 `UNSIGNED_INT` 索引会返回 `INVALID_OPERATION`
并**什么也不画** —— 没有异常、控制台干净、画面全黑。
用户只能说"还是黑屏"，我也只能猜。

更糟的是：我的无头探针 stub 里 `OES_element_index_uint` **永远可用**，
而且从不校验"索引类型是否与扩展匹配"，所以探针一路绿灯。

## 修复的两部分（各配测试）

1. `splat.js` 先 `getExtension('OES_element_index_uint')`，
   拿到才走 32 位索引快路径；**拿不到就改用无索引的 6 顶点/实例 + 重排顶点缓冲**，
   从而不依赖任何索引扩展。
2. 画完 `readPixels` 抽查 144 个采样点，亮点为 0 就返回 `blank: true`，
   界面把它写进状态栏 —— 把"黑屏"变成可报告的事实。
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPLAT = ROOT / "src" / "gsflow" / "web" / "static" / "splat.js"
PROBE = ROOT / "scripts" / "probe_viewer_runtime.mjs"


def _pick_ply() -> Path | None:
    for job in ("img3-final", "img3-crop", "verify-e2e2"):
        for p in sorted((ROOT / "workspace" / job / "04_train" / "point_cloud")
                        .glob("*/point_cloud.ply")):
            head = p.read_bytes()[:4000]
            if b"f_dc_0" in head:
                return p
    return None


def test_splat_requests_uint_index_extension() -> None:
    """源码必须显式取 `OES_element_index_uint`，并把 32 位索引用它来把关。"""
    src = SPLAT.read_text(encoding="utf-8")
    assert "OES_element_index_uint" in src, "没有请求 32 位索引扩展（WebGL1 会静默不画）"
    assert re.search(r"useIndexed\s*=\s*!!extU32", src), \
        "32 位索引路径必须由扩展是否可用决定"
    # 无索引回退必须存在
    assert "drawArraysInstancedANGLE" in src, "缺少无索引回退路径"


def test_splat_has_blank_frame_probe() -> None:
    src = SPLAT.read_text(encoding="utf-8")
    assert "probePixels" in src and "readPixels" in src, "缺少画后像素自检"
    assert "blank" in src


def test_viewer_and_ui_report_blank() -> None:
    v = (ROOT / "src" / "gsflow" / "web" / "static" / "viewer.js").read_text(encoding="utf-8")
    a = (ROOT / "src" / "gsflow" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert "probePixels" in v, "viewer 应在泼溅绘制后调用像素自检"
    assert v.count("probePixels") >= 2, "两个泼溅入口（.splat / gauss3d）都要自检"
    assert "画面全黑" in a, "界面必须把全黑结果明确写出来"


def test_probe_stub_rejects_uint_without_extension() -> None:
    """stub 必须校验"没启用扩展却用 UNSIGNED_INT" —— 否则探针照样放过 bug。"""
    src = PROBE.read_text(encoding="utf-8")
    assert "0x1405" in src, "stub 应检查 UNSIGNED_INT 索引类型"
    assert "OES_element_index_uint" in src
    assert "静默不画" in src or "INVALID" in src


@pytest.mark.skipif(not _pick_ply(), reason="工作区里没有 3DGS PLY")
def test_probe_fallback_without_uint_index() -> None:
    """`--no-uint-index` 下必须走无索引路径，且**不产生** UNSIGNED_INT 错误。"""
    if shutil.which("node") is None:
        pytest.skip("没有 node")
    ply = _pick_ply()
    r = subprocess.run(["node", str(PROBE), str(ply), "--no-uint-index"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    out = (r.stdout or "") + (r.stderr or "")
    flat = out.replace(" ", "").lower()
    assert '"indexed":false' in flat, f"没有回退到无索引路径：\n{out[-700:]}"
    assert '"mode":"gauss3d"' in flat, f"回退路径没能渲染：\n{out[-700:]}"
    assert "OES_element_index_uint" not in out or "INVALID" not in out, \
        f"回退路径仍触发了索引扩展错误：\n{out[-700:]}"
    assert r.returncode == 0, f"探针以 {r.returncode} 退出：\n{out[-700:]}"


@pytest.mark.skipif(not _pick_ply(), reason="工作区里没有 3DGS PLY")
def test_probe_reports_blank_in_headless() -> None:
    """无头 stub 不做真实合成，像素自检应如实报 `blank: true`（而非撒谎说没事）。"""
    if shutil.which("node") is None:
        pytest.skip("没有 node")
    ply = _pick_ply()
    r = subprocess.run(["node", str(PROBE), str(ply)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    out = (r.stdout or "") + (r.stderr or "")
    flat = out.replace(" ", "").lower()
    assert '"pixels":{"blank":true' in flat, f"像素自检没有如实报告：\n{out[-700:]}"
