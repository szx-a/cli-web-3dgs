"""回归测试：着色器必须兼容 GLSL ES 1.00（WebGL1），且泼溅失败要能降级。

## 真实故障（用户报告）

在 GUI 里打开模型时：
```
泼溅初始化失败：着色器编译失败：ERROR: 0:23:'transpose': no matching overloaded
function found
ERROR: 0:27:'transpose': no matching overloaded function found
```
`transpose()` 是 **GLSL ES 3.00（WebGL2）** 才有的函数，查看器用的是 WebGL1
⇒ 着色器编译失败 ⇒ 泼溅完全不可用。

**这类错误在无头环境里测不出来**（没有 GLSL 编译器），只能做静态检查。
本测试把 `scripts/check_glsl_es1.py` 纳入 CI，并要求它带**负向自测**
（注入 `transpose()` 必须被检出），否则"检查器永远通过"没有意义。

同时断言**降级路径存在**：泼溅初始化失败时必须退回点云 + 明确告知，
而不是抛异常让用户看到黑屏。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def test_shaders_are_glsl_es1_compatible() -> None:
    """着色器不得使用 ES 3.00 专属函数（含负向自测）。"""
    r = subprocess.run([PY, str(ROOT / "scripts" / "check_glsl_es1.py"), "--selftest"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, f"GLSL 兼容性检查失败：\n{out}"
    assert "全部通过 GLSL ES 1.00 语义校验" in out, out
    # 负向自测必须生效（证明检查器真的会报错，而不是恒过）
    assert "被检出 ✅" in out, f"检查器的负向自测失效：\n{out}"
    assert "无误报 ✅" in out, f"原着色器被误报了：\n{out}"


def test_checker_reports_transpose_if_present() -> None:
    """直接对检查器函数做已知坏输入的断言（含"注释里出现不该报"的回归）。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_glsl_es1 as C

    good = "void main() { gl_Position = vec4(1.0); }"
    assert C.check_shader(good, "ok") == [], "合法着色器不应报错"

    # ES 3.00 专属函数必须逐个被抓到
    for fn in ("transpose", "inverse", "texture", "textureLod", "round"):
        bad = good + f"\nmat3 t = {fn}(mat3(1.0));"
        probs = C.check_shader(bad, "bad")
        assert any(fn in p for p in probs), f"{fn} 未被检出：{probs}"

    # 拼写错误也要抓（白名单机制的真正价值）
    typo = good + "\nfloat q = normaize(vec3(1.0));"
    probs = C.check_shader(typo, "typo")
    assert any("normaize" in p for p in probs), f"拼写错误未被检出：{probs}"

    # 注释里的函数名**不该**被报（我曾因为它误报）
    commented = good + "\n// 这里不能写 transpose()\n/* 也不能写 inverse() */\n"
    assert C.check_shader(commented, "c") == [], "注释里的函数名不应被当成违规"


def test_checker_flags_varying_mismatch() -> None:
    """顶点/片元 varying 不一致会导致运行时 link 失败，必须能查出来。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_glsl_es1 as C

    vert = "varying vec2 vPos;\nvoid main() { vPos = vec2(1.0); gl_Position = vec4(1.0); }"
    frag_ok = ("precision mediump float;\nvarying vec2 vPos;\n"
               "void main() { gl_FragColor = vec4(vPos, 0.0, 1.0); }")
    assert C.check_shader(vert, "v", frag_ok) == []
    # 类型不一致
    frag_bad = ("precision mediump float;\nvarying vec3 vPos;\n"
                "void main() { gl_FragColor = vec4(vPos, 1.0); }")
    probs = C.check_shader(vert, "v", frag_bad)
    assert any("类型不一致" in p for p in probs), probs
    # 对侧完全缺失
    frag_missing = ("precision mediump float;\n"
                    "void main() { gl_FragColor = vec4(1.0); }")
    probs2 = C.check_shader(vert, "v", frag_missing)
    assert any("没有声明" in p for p in probs2), probs2


def test_app_shows_degraded_reason() -> None:
    """GUI 状态栏必须显示降级原因，否则用户会把降级误当成模型质量差。"""
    src = (ROOT / "src" / "gsflow" / "web" / "static" / "app.js").read_text(
        encoding="utf-8")
    assert "info.degraded" in src
    assert "已降级为点云" in src
