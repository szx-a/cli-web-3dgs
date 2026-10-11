"""查看器（前端 JS）回归测试：用 Node 跑真实 `viewer.js`，把关键行为钉住。

为什么需要：`viewer.js` 是纯前端代码，前面的 bug 都**不会**被 Python 测试发现，
而且浏览器里只能看到"黑屏"，没有任何可读的错误。所以这里用
`scripts/probe_viewer_runtime.mjs` 在 Node 里加载**真实的 viewer.js**
（stub 掉 WebGL/DOM），把两类曾经真实发生的问题变成自动化断言：

1. **线框模式整帧不画**：原先 `drawElements(gl.LINES, 三角形索引数)`，
   索引数不是偶数时 WebGL 报 INVALID_OPERATION，**什么都不画**。
   现在必须为线框单独生成线段索引（偶数个）。
2. **万向锁**：pitch 被硬夹在 ±1.5 rad，而 m4lookAt 在相机与 up 共线时
   `norm(cross(up, z))` 得到零向量 -> 视图矩阵含 NaN -> 画面消失。
   现在拖过顶/底时矩阵必须始终有限。

Node 不可用时跳过（不阻塞 Python 测试）。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "workspace"

STATIC_DIR = ROOT / "src" / "gsflow" / "web" / "static"


def _node() -> str | None:
    return shutil.which("node")


def _pick_glb() -> Path | None:
    """找一个真实导出过的网格 GLB 用于测试。"""
    if not WORKSPACE.exists():
        return None
    cands: list[Path] = []
    for job in sorted(WORKSPACE.iterdir(), reverse=True):
        out = job / "05_export"
        if not out.is_dir():
            continue
        cands += sorted(out.glob("*mesh*.glb"))
        cands += sorted(out.glob("*.glb"))
        # 优先真正的网格（有索引），其次点云
        meshes = [c for c in cands if "mesh" in c.name]
        if meshes:
            return meshes[0]
        if cands:
            return cands[0]
    return None


needs_node = pytest.mark.skipif(_node() is None, reason="没有 node，跳过前端回归")
needs_glb = pytest.mark.skipif(_pick_glb() is None, reason="workspace 里没有可用的 GLB 产物")


def _run_probe(glb: Path) -> subprocess.CompletedProcess:
    py = shutil.which("node") or "node"
    return subprocess.run(
        [py, str(ROOT / "scripts" / "probe_viewer_runtime.mjs"), str(glb)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=180,
    )


@needs_node
@needs_glb
def test_viewer_loads_a_real_glb() -> None:
    """真实 viewer.js 必须能加载真实产物，并报告一致的解析结果。"""
    glb = _pick_glb()
    assert glb is not None
    r = _run_probe(glb)
    out = (r.stdout or "") + (r.stderr or "")
    assert "加载 viewer.js: OK" in out, out
    assert "init() OK" in out, out
    assert "load() 返回" in out, out
    # 解析出的顶点数必须为正，且不该出现解析异常
    assert "解析失败" not in out, out
    assert "**" not in out.split("load() 返回")[0], out


@needs_node
@needs_glb
def test_viewer_no_gimbal_lock_and_wireframe_index_is_even() -> None:
    """拖过顶/底不得产生 NaN 矩阵；线框的线段索引数必须是偶数。

    这两条分别对应两个真实 bug（见模块 docstring）。探针在检测到退化时以非 0 退出。
    """
    glb = _pick_glb()
    assert glb is not None
    r = _run_probe(glb)
    out = (r.stdout or "") + (r.stderr or "")

    assert "含 NaN/Inf 的次数 = 0" in out, (
        "拖拽越界后视图矩阵出现 NaN（万向锁回归）：\n" + out)
    assert "LINES" not in out or "INVALID_OPERATION" not in out, (
        "线框用了奇数长度的索引缓冲，WebGL 会整帧不画：\n" + out)
    assert r.returncode == 0, f"探针以 {r.returncode} 退出：\n{out}"


@needs_node
def test_viewer_source_has_no_hard_pitch_clamp() -> None:
    """`viewer.js` 不得再出现旧的 ±1.5 硬夹取（它会锁死竖直方向）。

    这条是"静态"检查：即使没有 GLB 产物也能跑，防止改回去。
    """
    src = (STATIC_DIR / "viewer.js").read_text(encoding="utf-8")
    assert "Math.min(1.5, pitch" not in src, "pitch 又被硬夹在 1.5（万向锁会回来）"
    assert "PITCH_LIMIT" in src, "缺少 PITCH_LIMIT"
    # lookAt 必须有共线保护
    assert "共线退化保护" in src or "1e-6" in src, "m4lookAt 缺少共线保护"
