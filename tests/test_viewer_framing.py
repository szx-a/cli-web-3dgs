"""回归测试：WebGL1 查看器（glb / 普通 ply）的取景必须用**完整包围盒**。

## 历史

我曾把取景半径从"包围盒"改成"中位数中心 + 95 分位距离"，理由是
"包围盒被 0.61% 的离群点撑大 5.65 倍，相机被拉太远"。**那个判断是错的**：

素材是一座牌坊，质量集中在中央板面，而**屋檐/翼角稀疏但真实**。
按分位数裁会把它们当噪点裁掉 ⇒ 半径 0.79 ⇒ 相机距离 2.2 ⇒
物体跨 6.4×4.7×5.0 ⇒ **相机钻进模型内部**，网格看起来整片深色
（用户："网络都变深色了"）。

⚠️ 唯一保留的"缩"是**按视锥正确取景**，不是按分位数。所以这里断言：
  * `viewer.js` 里不得再出现分位数取景；
  * 必须用包围盒对角线。

（高斯取景现在由参考渲染器负责，见 `test_splat_webgl_contract.py`；
本文件只覆盖仍由我们维护的 glb/ply 路径。）
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER_JS = ROOT / "src" / "gsflow" / "web" / "static" / "viewer.js"


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
    code = _strip_comments(VIEWER_JS.read_text(encoding="utf-8"))
    assert "q95" not in code, "viewer.js 仍在用 95 分位距离定取景半径"
    assert "0.95 *" not in code, "viewer.js 仍在算 95 分位"
    assert "Math.hypot(mx[0]-mn[0]" in code.replace(" ", ""), \
        "viewer.js 应使用包围盒对角线定取景半径"


def test_framing_keeps_robust_flag_for_debug() -> None:
    """返回值里应保留 `robust` 字段（诊断用），但当前必须是 false。"""
    code = _strip_comments(VIEWER_JS.read_text(encoding="utf-8"))
    assert "robust" in code
    assert "robust = false" in code.replace("let center = bboxCenter, radius = bboxRadius, robust = false;",
                                            "robust = false")
