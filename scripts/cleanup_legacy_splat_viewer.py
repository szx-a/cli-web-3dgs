"""一次性清理：从 GUI 移除我手写的光栅化器（splat.js）及其死分支。

## 为什么要做

用户的原话：

> 光栅化你从头写没必要啊，而且也不好，你至少参考一下官方的啊
> … gui 整体逻辑是不是有问题，不要硬嵌入进去

`refviewer.js` 接管高斯显示后，`splat.js` 与 `viewer.js` 里的两处
`GsSplatRenderer` 分支已经**不可达**，但脚本仍被 `<script>` 加载、
`app.js` 仍把它当"必需依赖"检查。结果：
  * 页面加载即为高斯模式也创建一个 WebGL1 上下文（白占）；
  * "到底哪个渲染器在画"无法判断 —— 正是用户说的"硬嵌入"。

本脚本做三件事（幂等）：
  1. `index.html` 删掉 `<script src="/static/splat.js">`；
  2. `app.js` 的缺失脚本检查不再要求 `splat.js`；
  3. `viewer.js` 里 `.splat` 与 `gauss3d` 两个分支整体替换为一句明确报错
     （高斯由参考渲染器负责），不再调用 `GsSplatRenderer`。

用法: python scripts/cleanup_legacy_splat_viewer.py [--check]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "gsflow" / "web" / "static"


def remove_js_block(src: str, start_marker: str, end_marker: str,
                    replacement: str) -> tuple[str, bool]:
    """把 start_marker 所在行到 end_marker 所在行（含）整体替换。"""
    i = src.find(start_marker)
    if i < 0:
        return src, False
    j = src.find(end_marker, i)
    if j < 0:
        return src, False
    # 往前退到行首，往后到行尾
    ls = src.rfind("\n", 0, i) + 1
    le = src.find("\n", j + len(end_marker))
    if le < 0:
        le = len(src)
    return src[:ls] + replacement + src[le:], True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    changes: list[str] = []

    # ---- 1) index.html 不再加载 splat.js ----
    p = STATIC / "index.html"
    t = p.read_text(encoding="utf-8")
    if '<script src="/static/splat.js"></script>' in t:
        t = t.replace('<script src="/static/splat.js"></script>\n', "")
        t = t.replace('<script src="/static/splat.js"></script>', "")
        p.write_text(t, encoding="utf-8")
        changes.append("index.html: 移除 splat.js 的 <script>")

    # ---- 2) app.js 不再把 splat.js 当必需依赖 ----
    p = STATIC / "app.js"
    t = p.read_text(encoding="utf-8")
    old = "  if (!window.GsSplatRenderer) missing.push('splat.js');\n"
    if old in t:
        t = t.replace(old, "")
        p.write_text(t, encoding="utf-8")
        changes.append("app.js: 缺失检查不再要求 splat.js")

    # ---- 3) viewer.js 删掉两个高斯分支 ----
    p = STATIC / "viewer.js"
    t = p.read_text(encoding="utf-8")
    guard = (
        "      // 高斯类产物由 `refviewer.js`（社区参考实现 gaussian-splats-3d）负责渲染。\n"
        "      // 这里保留一句明确报错，避免有人再把 .splat / gauss3d 路由回这个 WebGL1 查看器\n"
        "      // —— 自研光栅化器的差距会被误读成模型质量差（docs/03 G116/G122）。\n"
        "      if (isSplat || (model && model.gauss3d)) {\n"
        "        throw new Error('高斯模型请用参考渲染器（refviewer.js）；'\n"
        "                        + '本查看器只负责 glb / 普通 ply');\n"
        "      }\n"
    )
    # 分支 A：`if (isSplat) {` ... 结束于紧随其后的 `let model;`
    t2, okA = remove_js_block(t, "if (isSplat) {", "      }\n\n      let model;",
                              guard + "\n      let model;")
    if okA:
        t = t2
        changes.append("viewer.js: 移除 .splat 自研渲染分支")
    # 分支 B：`if (model && model.gauss3d) {` 到 `if (!model || !model.position`
    t3, okB = remove_js_block(
        t, "if (model && model.gauss3d) {", "      }\n\n      if (!model || !model.position",
        "      if (!model || !model.position")
    if okB:
        t = t3
        changes.append("viewer.js: 移除 gauss3d 自研渲染分支")
    if okA or okB:
        p.write_text(t, encoding="utf-8")

    if not changes:
        print("无需改动（已经是清理后的状态）")
        return 0
    for c in changes:
        print("  " + c)
    if args.check:
        print("**--check 模式：以上改动尚未写入**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
