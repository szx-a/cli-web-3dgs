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
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for row in ("`frames.auto_crop`", "`frames.crop`", "`train.auto_crop`",
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
