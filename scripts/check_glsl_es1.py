"""GLSL ES 1.00（WebGL1）**语义**校验器：不是只查几个禁用函数名，而是查所有函数调用。

## 为什么必须升级（真实故障，而且降级不是解决办法）

用户在 GUI 里打开模型时报：
```
泼溅初始化失败：着色器编译失败：ERROR: 0:23:'transpose': no matching overloaded
function found
```
`transpose()` 是 GLSL ES 3.00 才有的函数。我最初只用一个"禁用函数名清单"来查，
这只能挡住**我已经想到的名字**；下一个 ES3 函数（`inverse` / `textureLod` / `trunc`…）
或者拼错的函数名照样漏过去。

**退回点云不是修复，是妥协** —— 用户要的效果（纹理、实感、真实高斯外观）会直接没了。
所以这里做成真正的校验：

  1. **收集用户自定义函数名**（`vec3 foo(...) { }`）；
  2. 找出所有**调用点**（`ident(`），逐个检查：要么是自定义函数，要么在白名单里；
  3. 白名单 = GLSL ES 1.00 规范的内建函数全表 + 关键字构造（`vec3(` / `mat3(` 等）；
  4. 额外检查：`varying` 在顶点/片元两侧必须**同名同类型**（不匹配运行时链接失败）；
     片元着色器必须有 `precision` 声明；不得出现 `attribute`（片元里非法）。

这样任何未声明的标识符调用（含所有 ES3 专属函数、拼写错误）都会被抓到。

用法: python scripts/check_glsl_es1.py [--selftest]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# GLSL ES 1.00 规范的内建函数（完整表）
BUILTINS = {
    # 角度/三角
    "radians", "degrees", "sin", "cos", "tan", "asin", "acos", "atan",
    # 指数
    "pow", "exp", "log", "exp2", "log2", "sqrt", "inversesqrt",
    # 常用
    "abs", "sign", "floor", "ceil", "fract", "mod", "min", "max", "clamp",
    "mix", "step", "smoothstep",
    # 几何
    "length", "distance", "dot", "cross", "normalize", "faceforward",
    "reflect", "refract",
    # 矩阵
    "matrixCompMult",
    # 向量关系
    "lessThan", "lessThanEqual", "greaterThan", "greaterThanEqual",
    "equal", "notEqual", "any", "all", "not",
    # 纹理（ES 1.00 只有 texture2D / textureCube）
    "texture2D", "texture2DProj", "texture2DLod", "texture2DProjLod",
    "textureCube", "textureCubeLod",
    # 其他
    "dFdx", "dFdy", "fwidth",   # 需要 OES_standard_derivatives 扩展
}

# 类型构造器/转换（`vec3(...)` `mat3(...)` 等），以及结构体构造（用户定义，另行收集）
TYPE_CTORS = {
    "void", "bool", "int", "float",
    "vec2", "vec3", "vec4", "bvec2", "bvec3", "bvec4",
    "ivec2", "ivec3", "ivec4",
    "mat2", "mat3", "mat4",
    "sampler2D", "samplerCube",
}

# 明确禁止的 ES 3.00 专属函数（用于给出更友好的错误信息；
# 真正的拦截靠"调用名必须在白名单里"，这条只是提示）
ES3_HINT = {
    "transpose", "inverse", "texture", "textureLod", "textureGrad",
    "textureSize", "round", "roundEven", "trunc", "sinh", "cosh", "tanh",
    "asinh", "acosh", "atanh", "isnan", "isinf", "packSnorm2x16",
    "unpackSnorm2x16", "floatBitsToInt", "intBitsToFloat",
}

KEYWORDS = {
    "if", "else", "for", "while", "do", "break", "continue", "return",
    "discard", "switch", "case", "default", "struct",
    "void", "bool", "int", "float",
    "vec2", "vec3", "vec4", "bvec2", "bvec3", "bvec4",
    "ivec2", "ivec3", "ivec4", "mat2", "mat3", "mat4",
    "sampler2D", "samplerCube",
    "attribute", "uniform", "varying", "const", "precision", "highp",
    "mediump", "lowp", "invariant", "true", "false",
}


def strip_comments(src: str) -> str:
    """去掉 // 与 /* */ 注释（保留换行以便定位行号）。"""
    src = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def extract_shaders(path: Path) -> list[tuple[str, int, str]]:
    """从 JS 里抽出 `const NAME = [ ... ].join('\\n')` 形式的着色器源码。"""
    text = path.read_text(encoding="utf-8")
    out = []
    for m in re.finditer(r"const\s+(\w+)\s*=\s*\[(.*?)\]\.join", text, re.S):
        name, body = m.group(1), m.group(2)
        if not re.search(r"void\s+main\s*\(", body):
            continue
        # 把 JS 数组里的字符串字面量拼起来（去引号与逗号）
        lines = re.findall(r"'((?:[^'\\]|\\.)*)'", body)
        code = "\n".join(x.replace("\\n", "\n").replace("\\'", "'")
                         .replace('\\"', '"') for x in lines)
        start_line = text[: m.start()].count("\n") + 1
        out.append((name, start_line, code))
    return out


def gather_functions(code: str) -> set[str]:
    """收集用户自定义函数名：`type name(args) {`"""
    return set(re.findall(
        r"\b(?:void|bool|int|float|vec[234]|bvec[234]|ivec[234]|mat[234])\s+"
        r"(\w+)\s*\([^;{]*\)\s*\{", code))


def gather_varyings(code: str) -> dict[str, str]:
    """收集 varying 声明：name -> type。"""
    out = {}
    for m in re.finditer(r"^\s*varying\s+(\w+)\s+(\w+)\s*;", code, re.M):
        out[m.group(2)] = m.group(1)
    return out


def check_shader(code: str, name: str, other: str | None = None) -> list[str]:
    """返回问题列表（空 = 通过）。`other` 是配对着色器源码（用于 varying 校验）。"""
    problems: list[str] = []
    clean = strip_comments(code)
    declared = gather_functions(clean) | BUILTINS | TYPE_CTORS | KEYWORDS

    # ---- 1) 所有调用点必须在"自定义函数 / 内建 / 类型构造器"里 ----
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", clean):
        fn = m.group(1)
        if fn in declared:
            continue
        line = clean[: m.start()].count("\n") + 1
        hint = "（**GLSL ES 3.00 专属**，WebGL1 不存在）" if fn in ES3_HINT else \
               "（未声明，或拼写错误）"
        problems.append(f"第 {line} 行调用了未知函数 `{fn}()` {hint}")

    # ---- 2) 片元着色器必须有 precision 声明 ----
    if "gl_FragColor" in clean and "precision" not in clean:
        problems.append("片元着色器缺少 `precision` 声明（ES 1.00 要求显式给出浮点精度）")

    # ---- 3) attribute 只能出现在顶点着色器 ----
    if "gl_FragColor" in clean and re.search(r"^\s*attribute\s+", clean, re.M):
        problems.append("片元着色器里出现 `attribute`（ES 1.00 非法）")

    # ---- 4) varying 必须与配对着色器同名同类型（否则链接失败）----
    if other is not None:
        mine = gather_varyings(clean)
        theirs = gather_varyings(strip_comments(other))
        for k, t in mine.items():
            if k not in theirs:
                problems.append(f"varying `{k}` 在配对着色器里没有声明"
                                f"（运行时 link 失败）")
            elif theirs[k] != t:
                problems.append(f"varying `{k}` 类型不一致：本侧 {t}，对侧 {theirs[k]}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="src/gsflow/web/static/splat.js")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    path = ROOT / args.file
    if not path.exists():
        print(f"找不到 {path}")
        return 1
    shaders = extract_shaders(path)
    if not shaders:
        print(f"**{path.name} 里没找到着色器源码**")
        return 1

    print(f"=== GLSL ES 1.00 语义校验：{path.name} ===")
    bad = 0
    codes = {n: c for n, _l, c in shaders}
    for name, line, code in shaders:
        # 顶点/片元配对检查
        other = None
        for oname, ocode in codes.items():
            if oname != name:
                other = ocode
                break
        probs = check_shader(code, name, other)
        if probs:
            bad += len(probs)
            print(f"  [FAIL] {name}（起始行 {line}）")
            for p in probs[:12]:
                print(f"         - {p}")
            if len(probs) > 12:
                print(f"         … 另有 {len(probs) - 12} 条")
        else:
            print(f"  [OK  ] {name}（起始行 {line}）：函数调用/精度/varying 全部合法")

    if args.selftest:
        print("\n  --- 负向自测 ---")
        base = shaders[0][2]
        cases = [
            ("transpose()", base + "\nmat3 t = transpose(mat3(1.0));"),
            ("inverse()", base + "\nmat3 t = inverse(mat3(1.0));"),
            ("texture()", base + "\nvec4 t = texture(s, uv);"),
            ("拼错的函数名", base + "\nfloat t = normaize(vec3(1.0));"),
        ]
        for label, src in cases:
            probs = check_shader(src, "selftest")
            caught = len(probs) > 0
            if not caught:
                bad += 1
            print(f"    {label:<16} {'被检出 ✅' if caught else '**未被检出 ❌**'}")
        # 干净的源码**不能**误报
        clean_probs = check_shader(base, "selftest")
        if clean_probs:
            bad += 1
            print(f"    {'原始着色器':<16} **误报 ❌**：{clean_probs[:2]}")
        else:
            print(f"    {'原始着色器':<16} 无误报 ✅")

    print("\n结论: " + (f"{len(shaders)} 个着色器全部通过 GLSL ES 1.00 语义校验" if bad == 0
                      else f"**{bad} 个问题**"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
