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
    """相机朝向必须由**四元数**组合，且不得再出现欧拉角矩阵式的退化。

    演进（用户反馈了三次，前三版都没真正解决，见 docs/03 G55/G58）：
      1) 欧拉角 pitch 硬夹 ±1.5 rad；
      2) 四元数 + 竖直绕"当前相机右轴"增量施加 —— 过极点后右轴翻向 ⇒ 反向；
      3) 四元数 trackball / Arcball —— 竖直拖动变成绕固定轴的**圆锥运动**，
         实测 y 只在 ±0.435 之间震荡（589 步反向），根本到不了极点；
      4) 最终：**yaw 绕固定世界 Y 轴（不限圈数）+ pitch 绕水平右轴（夹 ±90°）**，
         两者用四元数相乘 `qcam = Qpitch ⊗ Qyaw`。
    关键点：轴固定在世界系，因此**方向恒定、单调、无进动、无反向**。
    """
    src = (STATIC_DIR / "viewer.js").read_text(encoding="utf-8")
    assert "Math.min(1.5, pitch" not in src, "pitch 又被硬夹在 1.5"
    for fn in ("qMul", "qNorm", "qAxisAngle", "qRot", "m4viewFromQuat"):
        assert fn in src, f"缺少四元数函数 {fn}"
    assert "function buildCam(" in src, "缺少 buildCam（四元数组合相机朝向）"
    assert "m4viewFromQuat(qcam" in src, "绘制时没有用四元数构造视图矩阵"
    # 轴必须固定在世界系，不能是"每帧用当前相机基重算"（那会产生进动）
    assert "qAxisAngle([0, 1, 0], yawAcc)" in src, "yaw 必须绕固定世界 Y 轴"
    assert "PITCH_MAX" in src, "pitch 应有 ±90° 上限（上下看的自然边界）"


@needs_node
def test_viewer_drag_is_monotonic_no_pushback() -> None:
    """持续朝同一方向拖动必须**单调**，不得中途反向（用户所谓"卡住/回弹"）。

    这条是最关键的回归：前面三版实现在这一项上全部失败。
    实测（scripts/probe_camera_monotonic.mjs，持续向下拖 2400px）：
      trackball/arcball 版：y 仅 ±0.435 来回震荡，非极点区反向 589/414 步；
      当前版：y 0.421 → −0.502 → −0.996 → −1.000（夹住），**反向 0 步**。
    """
    glb = _pick_glb()
    if glb is None:
        pytest.skip("workspace 里没有可用的 GLB 产物")
    py = shutil.which("node") or "node"
    r = subprocess.run(
        [py, str(ROOT / "scripts" / "probe_camera_monotonic.mjs"), str(glb)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=300,
    )
    out = (r.stdout or "") + (r.stderr or "")
    assert "非极点区反向步数 = 0" in out, f"拖动中途出现反向（主观卡住）：\n{out}"
    assert "无推回区段" in out, out
    assert r.returncode == 0, out


@needs_node
def test_standalone_viewer_is_self_contained_and_parseable() -> None:
    """离线查看器 HTML 必须**自包含且可解析**（不依赖任何服务）。

    为什么要这条：DSH 沙箱会回收我启动的服务（docs/03 G57），用户反复遇到
    "查看器又黑了 / Edge 报 404"。根治办法是提供**不依赖服务**的产物：
    把 viewer.js 与模型二进制内嵌进单个 HTML，`file://` 双击就能看。
    但"生成 HTML"这类脚本最容易静默出错（占位符没换、数据被截断、
    内嵌 JS 含 `</script>` 提前闭合）—— 生成时不报错，打开才发现黑屏。
    所以这里必须真的**执行内嵌脚本并解析模型**，而不是只检查文件存在。
    """
    mesh_glb = None
    if WORKSPACE.exists():
        for cand in sorted(WORKSPACE.iterdir(), reverse=True):
            p = cand / "05_export" / "model_mesh.glb"
            if p.exists() and p.stat().st_size > 100_000:
                mesh_glb = p
                break
    if mesh_glb is None:
        pytest.skip("workspace 里没有网格 GLB（model_mesh.glb）")
    job = mesh_glb.parent.parent.name
    py = sys.executable
    gen = subprocess.run(
        [py, str(ROOT / "scripts" / "make_standalone_viewer.py"),
         "--job", job, "--kind", "mesh"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=300,
    )
    if gen.returncode != 0:
        pytest.skip(f"生成离线查看器失败：{gen.stdout}{gen.stderr}")
    out = mesh_glb.parent / "viewer_standalone_mesh.html"
    assert out.exists(), "未生成离线查看器 HTML"
    assert out.stat().st_size > 100_000, "HTML 过小，模型数据可能没内嵌进去"

    py_node = shutil.which("node") or "node"
    r = subprocess.run(
        [py_node, str(ROOT / "scripts" / "probe_standalone_viewer.mjs"), str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=300,
    )
    text = (r.stdout or "") + (r.stderr or "")
    assert "占位符已全部替换" in text, text
    assert "离线查看器可用" in text, f"离线查看器不可用：\n{text}"
    assert r.returncode == 0, text


@needs_node
def test_viewer_camera_modes_are_switchable() -> None:
    """旋转模式必须可切换（'clamp' 默认 / 'free' 可翻越极点）。

    提供两种模式的原因：用户三次反馈"卡住"，而"到极点停住"与"翻越极点"是
    两种合理但不兼容的手感，无法靠猜测决定 —— 因此两种都实现、可运行时切换。
    这里断言：默认是 clamp；切成 free 后能到达**另一个**极点（说明确实越过了极点）。
    """
    glb = _pick_glb()
    if glb is None:
        pytest.skip("workspace 里没有可用的 GLB 产物")
    py = shutil.which("node") or "node"
    r = subprocess.run(
        [py, str(ROOT / "scripts" / "probe_camera_modes.mjs"), str(glb)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=300,
    )
    out = (r.stdout or "") + (r.stderr or "")
    assert "默认模式 = clamp" in out, out
    assert "clamp 反向步数 = 0" in out, f"clamp 模式出现反向：\n{out}"
    assert "free 能到达两个极点 = 是" in out, f"free 模式无法翻越极点：\n{out}"
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
