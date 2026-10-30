"""前端契约：**两套渲染器共存时，按钮必须路由到"当前活动"的那个**。

## 背景（2026-10-25，用户反馈）

> 模型查看的 gui 按钮逻辑优化一下，在交互上会有冲突尤其是引入 gsplat 官方光栅化器后
> 与 webgl 的交互没分开

页面里有两套渲染器：

* `GsRefViewer`（`refviewer.js`，社区库 `gaussian-splats-3d`）—— 3DGS 的 `.ply` / `.splat`；
* `GsViewer`（`viewer.js`，自研 WebGL1）—— `.glb` / 普通 `.ply` / 网格。

旧代码里"重置视角 / 切换线框 / 旋转模式"三个按钮**无条件**调用 `window.GsViewer.*`：
显示 3DGS 时 `GsViewer` 并没有在屏幕上 ⇒ 按钮要么无反应，要么把命令打到**隐藏的另一个渲染器**。

修法：`app.js` 里引入 `ViewerHub`，显式记录 `active`（'ref' | 'gs' | null）与能力表，
按钮只经 Hub 路由；不支持的操作**禁用按钮并在 title 说明原因**。

这些测试是**静态契约**（读源码断言），因为本仓库没有前端运行时测试环境；
它们拦的是"又有人把按钮直接绑回 GsViewer"这类回归。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "gsflow" / "web" / "static" / "app.js"


def _src() -> str:
    return APP.read_text(encoding="utf-8")


def test_viewer_hub_exists_with_capability_table() -> None:
    t = _src()
    assert "const ViewerHub = (() => {" in t, "缺少 ViewerHub 路由层"
    assert "ref: { reset: true, wire: false, rotmode: false }" in t, "缺少参考渲染器的能力表"
    assert "gs: { reset: true, wire: true, rotmode: true }" in t, "缺少自研查看器的能力表"
    assert "function supports(cap)" in t
    assert "function syncButtons()" in t


def test_hub_is_switched_at_every_mount_point() -> None:
    t = _src()
    assert "ViewerHub.set('gs')" in t, "自研查看器就绪后没有设为活动渲染器"
    assert "ViewerHub.set('ref')" in t, "参考渲染器挂载后没有设为活动渲染器"
    assert "ViewerHub.setRefInfo" in t, "参考渲染器缺少 resetCamera 所需的取景信息"
    # 图片/预览模式没有可交互的 3D 渲染器，必须显式置空（否则按钮会打到隐藏 canvas）
    # ⚠ 2026-10-30 修正：临时切开（图片/预览、服务端渲染）必须用 suspend()，
    # 不能直接 set(null)——后者会让「返回 WebGL」时无从恢复（交互被锁）。
    # 旧断言写的正是那个 bug，现在改成断言 suspend 的使用。
    assert t.count("ViewerHub.suspend(") >= 2, "临时切开场景没有用 suspend()（会导致交互锁死）"
    assert "ViewerHub.set(null)" not in t, "不应再直接 set(null)（应用 suspend）"


def test_buttons_route_through_hub_not_directly_to_gsviewer() -> None:
    """直接调用 `GsViewer.*` 只允许出现在 Hub **内部**（那正是路由实现）。

    第一版断言是"全文不得出现"，结果被 Hub 自己的实现打脸 —— 测试要精确到"按钮区"。
    """
    t = _src()
    hub_start = t.index("const ViewerHub = (() => {")
    hub_end = t.index("function ensureViewer()", hub_start)
    hub = t[hub_start:hub_end]
    rest = t[:hub_start] + t[hub_end:]

    for call in ("window.GsViewer.reset()", "window.GsViewer.toggleWire()",
                 "window.GsViewer.setRotMode("):
        assert call in hub, f"Hub 内缺少 {call} 的实现"
        assert call not in rest, f"Hub 之外仍有 {call}，会绕过活动渲染器路由"

    for good in ("ViewerHub.reset()", "ViewerHub.toggleWire()", "ViewerHub.setRotMode("):
        assert good in rest, f"按钮没有经 {good} 路由"


def test_unsupported_buttons_carry_reason() -> None:
    t = _src()
    assert "WHY" in t and "线框只对自研 WebGL 查看器" in t
    assert "el.disabled = !ok" in t, "不支持的按钮没有被禁用"
    assert "el.title = el.dataset.hubWhy" in t, "禁用的按钮没有说明原因"


def test_pointer_events_are_isolated_per_renderer() -> None:
    """只有活动渲染器的画布接收指针事件（用户说的"与 WebGL 的交互没分开"）。"""
    t = _src()
    assert "cv.style.pointerEvents = (active === 'gs') ? 'auto' : 'none'" in t, \
        "自研 canvas 的指针事件没有随活动渲染器切换"
    assert "stage.style.pointerEvents = (active === 'ref') ? 'auto' : 'none'" in t, \
        "参考渲染器容器的指针事件没有随活动渲染器切换"
