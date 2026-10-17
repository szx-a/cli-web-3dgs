"""回归测试：取景半径必须覆盖**稀疏但真实的远处结构**。

## 为什么（用户报"网络都变深色了"）

我一度把取景半径从"完整包围盒"改成"中位数中心 + 95 分位距离"，理由是
"包围盒被 0.61% 的离群点撑大 5.65 倍，相机被拉太远"。

**那个判断是错的**。素材是一座**牌坊**：

    实测 img3-final（94,598 个高斯）
      包围盒半径（真实尺寸）  4.680   ← 正确
      距中位数 95 分位        0.794   ← 当时用了这个
      距中位数 99.5 分位      1.496

质量集中在中央板面/门洞，**屋檐与翼角虽然稀疏，却是真实结构**。按分位数裁，
会先把真实的远处结构当噪点裁掉 ⇒ 半径 0.79 ⇒ 相机距离 2.2，
而物体跨 6.4×4.7×5.0 ⇒ **相机钻进模型内部** ⇒ 网格整片深色/全黑。

"离群点"的正确判据是**几何上不可能属于物体**，不是"数量少"。
本测试用合成数据把这条固定下来：1 个远处点必须被纳入取景。
"""
from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPLAT_JS = ROOT / "src" / "gsflow" / "web" / "static" / "splat.js"
VIEWER_JS = ROOT / "src" / "gsflow" / "web" / "static" / "viewer.js"


def _write_splat(path: Path, far_point: tuple[float, float, float] | None) -> int:
    """写一个 .splat：1000 个点聚在原点附近，可选再加一个远处点。"""
    import random

    random.seed(7)
    raw = bytearray()
    n = 1000
    for _ in range(n):
        pos = (random.gauss(0, 0.05), random.gauss(0, 0.05), random.gauss(0, 0.05))
        raw += struct.pack("<3f", *pos)
        raw += struct.pack("<3f", 0.01, 0.01, 0.01)
        raw += bytes([128, 128, 128, 255])
        raw += bytes([128, 128, 128, 128])
    if far_point is not None:
        raw += struct.pack("<3f", *far_point)
        raw += struct.pack("<3f", 0.01, 0.01, 0.01)
        raw += bytes([128, 128, 128, 255])
        raw += bytes([128, 128, 128, 128])
        n += 1
    path.write_bytes(bytes(raw))
    return n


def _parse_in_node(path: Path) -> dict:
    if shutil.which("node") is None:
        pytest.skip("没有 node")
    script = (
        "const fs=require('fs');"
        f"const src=fs.readFileSync({str(SPLAT_JS)!r},'utf8');"
        "const S=new Function('window', src+'\\nreturn window.GsSplatRenderer;')({});"
        f"const b=fs.readFileSync({str(path)!r});"
        "const p=S.parseSplat(b.buffer.slice(b.byteOffset, b.byteOffset+b.byteLength));"
        "console.log(JSON.stringify(p.bounds));"
    )
    r = subprocess.run(["node", "-e", script], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=str(ROOT))
    assert r.returncode == 0, (r.stdout or "") + (r.stderr or "")
    line = [l for l in (r.stdout or "").splitlines() if l.strip().startswith("{")]
    assert line, (r.stdout or "") + (r.stderr or "")
    return json.loads(line[-1])


def test_far_sparse_structure_is_included(workdir: Path) -> None:
    """稀疏的远处点必须被纳入取景（不能当离群点裁掉）。"""
    p = workdir / "far.splat"
    n = _write_splat(p, far_point=(5.0, 0.0, 0.0))
    assert n == 1001
    b = _parse_in_node(p)
    # 远处点在 x=5 ⇒ 取景半径至少 2.5（半对角线），中心 x 约 2.4~2.5
    assert b["radius"] >= 2.0, f"远处结构被裁掉了：radius={b['radius']:.3f}"
    assert b["center"][0] > 1.5, f"中心没有考虑远处点：{b['center']}"


def test_no_far_point_gives_tight_bounds(workdir: Path) -> None:
    """没有远处点时，取景半径应当很小（说明不是无脑放大）。"""
    p = workdir / "near.splat"
    _write_splat(p, far_point=None)
    b = _parse_in_node(p)
    assert b["radius"] < 0.6, f"取景半径异常地大：{b['radius']}"


def _strip_comments(src: str) -> str:
    out, in_block = [], False
    for line in src.splitlines():
        s = line.strip()
        if in_block:
            if "*/" in s:
                in_block = False
            continue
        if s.startswith("/*"):
            in_block = "*/" not in s
            continue
        if s.startswith("//"):
            continue
        out.append(line)
    return "\n".join(out)


def test_framing_uses_bbox_not_percentile() -> None:
    """代码里不得再出现"按分位数定取景半径"的实现（注释里说明可以）。"""
    for f in (SPLAT_JS, VIEWER_JS):
        code = _strip_comments(f.read_text(encoding="utf-8"))
        assert "q95" not in code, f"{f.name} 仍在用 95 分位距离定取景半径"
        assert "0.95 *" not in code, f"{f.name} 仍在算 95 分位"
    # 必须仍然用包围盒对角线
    s = _strip_comments(SPLAT_JS.read_text(encoding="utf-8"))
    assert "Math.hypot(mxx - mnx" in s, "splat.js 应使用包围盒对角线"
