"""回归测试：文档不能落后于配置。

## 为什么需要

`check_readme_params.py` 原来只做**单向**校验（README 的每一行 → DEFAULTS），
所以"新增配置项但不更新文档"**永远不会被发现**：
我这轮加了 `train.auto_crop_in_train`，README 里没有，检查器照样报"0 项缺失"。

现在补了反向检查：一组**关键**配置项（影响最终质量或用户必须知道）必须出现在
README 参数表里，缺了就失败；其余近百个次要项只提示。

这类检查最大的风险是"恒过"（永远返回 0），所以这里同时验证它能**报错**。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run() -> tuple[int, str]:
    r = subprocess.run([PY, str(ROOT / "scripts" / "check_readme_params.py")],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def test_readme_documents_critical_config_keys() -> None:
    """关键配置项必须都在 README 里，且默认值一致。"""
    code, out = _run()
    assert "关键配置项全部已写进 README" in out, out[-1200:]
    assert code == 0, f"README 参数检查未通过：\n{out[-1200:]}"


def test_reverse_check_covers_known_quality_keys() -> None:
    """本轮的几个关键键必须在 MUST_DOC 集合里（否则反向检查形同虚设）。"""
    src = (ROOT / "scripts" / "check_readme_params.py").read_text(encoding="utf-8")
    for key in ("auto_crop_in_train", "frames", "densify_grad2d_threshold",
                "rasterize_mode", "mesh_drop_unseen"):
        assert key in src, f"MUST_DOC 里缺少 {key}"
    # README 里确实有这些行
    #
    # ⚠️ `frames.auto_crop` 曾被列在这里，但 **DEFAULTS 与 frames 阶段都没有这个键**
    # （frames 只读 `frames.crop`）—— 一个"检查器要求、代码里不存在"的幽灵键。
    # 2026-10-25 已从 check_readme_params.KEYS 与本列表一起移除：文档不该写不存在的参数。
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for row in ("`frames.crop`", "`train.auto_crop`",
                "`train.auto_crop_in_train`", "`train.densify_grad2d_threshold`"):
        assert row in readme, f"README 缺少 {row}"


def test_reverse_check_actually_fails(workdir: Path) -> None:
    """负向测试：把 README 复制一份、删掉一个关键行，检查器必须报错。

    这是"检查器真的会失败"的证明 —— 否则它只是个恒过的装饰。
    ⚠️ 用 `workdir` fixture 而不是 `tmp_path`：本机沙箱里 pytest 的
    `tmp_path` 收尾会 WinError 5（见 AGENTS.md）。
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    lines = [l for l in readme.splitlines(keepends=True)
             if "`train.auto_crop_in_train`" not in l]
    assert len(lines) < len(readme.splitlines(keepends=True)), "测试自身的删除没生效"
    bad_readme = workdir / "README.md"
    bad_readme.write_text("".join(lines), encoding="utf-8")

    r = subprocess.run([PY, str(ROOT / "scripts" / "check_readme_params.py"),
                        "--readme", str(bad_readme)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode != 0, f"删掉关键行后检查器仍然通过（恒过）：\n{out[-900:]}"
    assert "auto_crop_in_train" in out, out[-900:]

    # 正对照：原样 README 必须通过
    r2 = subprocess.run([PY, str(ROOT / "scripts" / "check_readme_params.py")],
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", cwd=str(ROOT))
    assert r2.returncode == 0, ((r2.stdout or "") + (r2.stderr or ""))[-900:]


# ---------------------------------------------------------------------------
# 概念级检查：只对"数值"做一致性检查会漏掉一整类漂移
# ---------------------------------------------------------------------------

def _run_concepts(*args: str) -> tuple[int, str]:
    r = subprocess.run([PY, str(ROOT / "scripts" / "check_docs_concepts.py"), *args],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def test_docs_cover_critical_concepts() -> None:
    """每篇文档都必须提到它该提的新机制（掩码内指标、参考渲染器、自动裁剪…）。

    背景：`check_readme_params.py` 只比**数值**，于是出现过
    "所有默认值一致 ✅，但 01/02/05 三篇 0 处提到 `psnr_masked`" ——
    照着文档做会得出"全场 FAIL"的错误结论（docs/03 G123）。
    """
    code, out = _run_concepts()
    assert code == 0, f"文档概念覆盖不足：\n{out[-1500:]}"
    assert "关键概念全部已覆盖" in out, out[-800:]


def test_concept_checker_has_negative_selftest() -> None:
    """检查器必须能在"已知缺失"上报错，否则就是恒过的装饰。"""
    code, out = _run_concepts("--selftest")
    assert code == 0, f"概念检查器的负向自测失败：\n{out[-800:]}"
    assert "负向自测通过" in out, out[-800:]


def test_quality_metric_caveat_is_documented() -> None:
    """**全图 PSNR 不能当画质指标**这条口径必须写进用户会读到的地方。

    这是本轮最容易被误传的一句话：历史 28.58 dB 与现在 20.47 dB（掩码内）
    完全不是一个量。只要有人照着老数字判断"模型变差了"，就会走回头路。
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "psnr_masked" in readme, "README 没提掩码内 PSNR"
    for name in ("01-架构说明.md", "02-操作手册.md", "05-训练框架.md"):
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        assert "psnr_masked" in text, f"{name} 没提掩码内 PSNR"
    # 必须明确"不可比较"，否则等于没提醒
    hit = any(k in (ROOT / "docs" / n).read_text(encoding="utf-8")
              for n in ("01-架构说明.md", "02-操作手册.md", "05-训练框架.md")
              for k in ("不可直接比较", "无意义", "不代表画质", "不再代表画质"))
    assert hit, "三篇文档都没有明确说明全图指标不可用/不可比较"
