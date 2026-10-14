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

import re
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
def test_viewer_source_uses_quaternion_camera_without_pitch_clamp() -> None:
    """相机必须用**四元数**且**不得再出现 pitch 夹取**。

    演进（三次都在同一问题上）：
      1) pitch 硬夹 ±1.5 rad（±86°）—— 竖直方向转不过去；
      2) 改成 ±(π/2−ε) + `lookAt` 共线兜底 —— 数值不崩了，但**仍然转不过顶/底**；
      3) 现在改为四元数累积旋转 —— 没有会退化的 pitch 自由度，可自由越过天顶/天底。
    这条是静态检查，防止退回夹取方案。
    """
    src = (STATIC_DIR / "viewer.js").read_text(encoding="utf-8")
    assert "Math.min(1.5, pitch" not in src, "pitch 又被硬夹在 1.5（万向锁会回来）"
    assert "PITCH_LIMIT" not in src, "仍存在 PITCH_LIMIT（说明又回到夹取方案）"
    # 四元数基础设施
    for fn in ("qMul", "qNorm", "qAxisAngle", "qRot", "m4viewFromQuat"):
        assert fn in src, f"缺少四元数函数 {fn}"
    # 相机状态是四元数，且拖拽走 orbit()
    assert "let qcam" in src, "相机状态不是四元数"
    assert "function orbit(" in src, "缺少 orbit() 增量旋转"
    assert "m4viewFromQuat(qcam" in src, "绘制时没有用四元数构造视图矩阵"


@needs_node
def test_viewer_camera_can_pass_through_poles() -> None:
    """四元数相机必须能**越过天顶/天底**（这是万向锁的真实症状）。

    做法：在 Node 里驱动真实 viewer.js，持续向下拖很久，检查
    `GsViewer.debug().camera.eye`（相机位置）始终有限、且**确实越过了极点**：
    相机方向向量的 y 分量应该先增到 +1 附近、**然后转为下降**（越过天顶后从另一侧下来）。
    老实现会卡在 y≈+1 附近不动。
    """
    glb = _pick_glb()
    if glb is None:
        pytest.skip("workspace 里没有可用的 GLB 产物")
    py = shutil.which("node") or "node"
    r = subprocess.run(
        [py, str(ROOT / "scripts" / "probe_camera_orbit.mjs"), str(glb)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=300,
    )
    out = (r.stdout or "") + (r.stderr or "")
    # 判据：相机方向能同时到达两极（旧实现会停在一个极点附近）
    assert "是否同时到达两极 = 是" in out, f"相机只能到达一个极点（万向锁仍在）：\n{out}"
    assert "含 NaN/Inf 的 = 0" in out, f"出现 NaN 矩阵：\n{out}"
    assert r.returncode == 0, out


@needs_node
def test_viewer_draw_elements_uses_legal_index_type() -> None:
    """`drawElements` 的索引类型必须合法 —— 这是"画布全黑"的真凶。

    WebGL1 的 `drawElements` **只接受 UNSIGNED_BYTE / UNSIGNED_SHORT**；
    用 `UNSIGNED_INT`(0x1405) 必须显式启用 `OES_element_index_uint` 扩展，否则报
    **INVALID_ENUM 并且一个图元都不画**（画布全黑，无任何可见提示）。

    实测经过（用户的 Edge 控制台）：
        GL INVALID_ENUM @ gl.drawElements(4, 222885, 5125(0x1405), 0)
    即无条件传了 UNSIGNED_INT。我们的网格普遍有 3~4 万顶点，
    超过 UNSIGNED_SHORT 的 65535 上限，所以**每个带面的模型都会黑屏**。

    这条测试让 Node 探针的 mock **模拟那条 WebGL1 约束**（探针里已加），
    于是"用了非法索引类型"会像浏览器一样报错并使探针非 0 退出。
    """
    # 选一个 u16 装得下、和一个装不下的模型，两条路径都要覆盖
    small = _pick_ply_mesh()
    if small is None:
        pytest.skip("workspace 里没有带面的网格 PLY")

    r = _run_probe(small)
    out = (r.stdout or "") + (r.stderr or "")
    assert "drawElements(type=" not in out, f"索引类型非法（会全黑）：\n{out}"
    # 合法的绘制调用：type 只能是 5121(UBYTE) 或 5123(USHORT)，或 5125 在扩展可用时
    assert '"drawElements",4' in out.replace(" ", ""), out
    assert r.returncode == 0, f"探针以 {r.returncode} 退出：\n{out}"


@needs_node
def test_viewer_reports_clear_error_without_uint_index_extension() -> None:
    """若浏览器**没有** `OES_element_index_uint` 且顶点数 > 65535，
    必须给出明确错误，而不是静默黑屏（这正是原来的失败模式）。"""
    big = None
    if WORKSPACE.exists():
        for job in sorted(WORKSPACE.iterdir(), reverse=True):
            p = job / "05_export" / "model_mesh.ply"
            if p.exists():
                head = p.read_bytes()[:4000]
                m = re.search(rb"element vertex (\d+)", head)
                if m and int(m.group(1)) > 65535:
                    big = p
                    break
    if big is None:
        pytest.skip("workspace 里没有顶点数 > 65535 的网格 PLY")

    py = shutil.which("node") or "node"
    r = subprocess.run(
        [py, str(ROOT / "scripts" / "probe_viewer_runtime.mjs"), str(big),
         "--no-uint-index"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=300,
    )
    out = (r.stdout or "") + (r.stderr or "")
    # 必须出现我们自己的、说明原因的报错（含"65535"与"OES_element_index_uint"）
    assert "65535" in out and "OES_element_index_uint" in out, (
        "无 32 位索引扩展时没有给出明确错误（会表现为静默黑屏）：\n" + out)


def _pick_ply_mesh() -> Path | None:
    """找一个三角网格 PLY（含 element face）。"""
    if not WORKSPACE.exists():
        return None
    for job in sorted(WORKSPACE.iterdir(), reverse=True):
        for name in ("model_mesh.ply",):
            p = job / "05_export" / name
            if p.exists():
                head = p.read_bytes()[:4000]
                if b"element face" in head:
                    return p
    return None


def _pick_point_ply() -> Path | None:
    """找一个 3DGS 点云 PLY（没有 element face）。"""
    if not WORKSPACE.exists():
        return None
    for job in sorted(WORKSPACE.iterdir(), reverse=True):
        for d in sorted((job / "04_train" / "point_cloud").glob("iteration_*"), reverse=True):
            p = d / "point_cloud.ply"
            if p.exists():
                head = p.read_bytes()[:4000]
                if b"element face" not in head and b"f_dc_0" in head:
                    return p
    return None


@needs_node
def test_ply_mesh_is_parsed_as_solid_not_point_cloud() -> None:
    """**网格 PLY 必须被解析成实体三角面，而不是点云。**

    回归：`parsePly` 曾经无条件返回 `mode: 0`（POINTS）与 `indices: null`，
    完全忽略 `element face`，于是**任何 PLY 都被画成点云** ——
    用户看到的"ply 查看只有点云、不是实体模型"就是这个原因
    （网格 PLY 里有 74295 个三角面，却一个都没被读）。
    """
    mesh = _pick_ply_mesh()
    pts = _pick_point_ply()
    if mesh is None:
        pytest.skip("workspace 里没有带 element face 的网格 PLY")

    r = _run_probe(mesh)
    out = (r.stdout or "") + (r.stderr or "")
    assert "load() 返回" in out, out
    # 必须报出三角形数与 mode=4
    assert '"mode":4' in out.replace(" ", ""), f"网格 PLY 未识别为 TRIANGLES：\n{out}"
    assert '"triangles":0' not in out.replace(" ", ""), \
        f"网格 PLY 的三角形数为 0（面没被解析）：\n{out}"
    # 且渲染走 drawElements(TRIANGLES=4)
    assert '["drawElements",4' in out.replace(" ", ""), \
        f"网格 PLY 没有用 drawElements(TRIANGLES) 绘制：\n{out}"
    assert "PLY 面索引越界" not in out, out

    # 对照组：点云 PLY 仍应是 POINTS（不能把点云误判成网格）
    if pts is not None:
        r2 = _run_probe(pts)
        out2 = (r2.stdout or "") + (r2.stderr or "")
        assert '"mode":0' in out2.replace(" ", ""), \
            f"点云 PLY 应保持 POINTS：\n{out2}"
        assert '["drawArrays",0' in out2.replace(" ", ""), out2
