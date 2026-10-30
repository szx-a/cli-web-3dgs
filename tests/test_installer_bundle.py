"""安装器离线包（--make-bundle / --from-bundle）的契约测试。

不联网、不复制真实 venv，只锁住：
1. CLI 参数存在且可选值正确；
2. 清单结构（bundle.json）包含安装端需要的字段；
3. `.pth` 重写把挂载点指向**新**安装目录（venv 复制后原路径失效，这是最容易错的点）；
4. 复制时跳过缓存/临时目录（复制它们会撞锁文件，2026-10-25 实测 WinError 5）；
5. 不是离线包时干净失败（不能假装装成功）。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "installer" / "gsflow_installer.py"


def _load():
    spec = importlib.util.spec_from_file_location("gsflow_installer_mod", INSTALLER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # ⚠ dataclass 需要能在 sys.modules 里找到所属模块（否则报
    # "'NoneType' object has no attribute '__dict__'"), 必须先注册再执行。
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_cli_exposes_bundle_flags() -> None:
    r = subprocess.run([sys.executable, str(INSTALLER), "--help"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0
    for flag in ("--make-bundle", "--from-bundle", "--bundle-mode", "--wheelhouse"):
        assert flag in r.stdout, f"缺少 {flag}"
    assert "venv" in r.stdout and "wheels" in r.stdout


def test_manifest_has_install_side_fields(workdir: Path) -> None:
    mod = _load()
    root = workdir / "install"
    (root / "env" / "venv" / "Scripts").mkdir(parents=True)
    (root / "env" / "venv" / "Scripts" / "python.exe").write_text("x", encoding="ascii")
    m = mod._bundle_manifest(root, "venv", ["core", "gpu"])
    assert m["format"] == "gsflow-offline-bundle"
    assert m["version"] == 1
    assert m["mode"] == "venv"
    assert m["components"] == ["core", "gpu"]
    assert m["has_venv"] is True
    json.dumps(m)  # 必须可序列化


def test_rewrite_pth_points_at_new_install(workdir: Path) -> None:
    mod = _load()
    venv = workdir / "new-install" / "env" / "venv"
    site = venv / "Lib" / "site-packages"
    site.mkdir(parents=True)
    (site / "gsflow-dev.pth").write_text(r"F:\old\src", encoding="ascii")
    mod._rewrite_pth(venv, workdir / "new-install")
    got = (site / "gsflow-dev.pth").read_text(encoding="ascii")
    assert got == str(workdir / "new-install" / "src")
    assert "old" not in got


def test_copy_tree_skips_caches(workdir: Path) -> None:
    mod = _load()
    src = workdir / "src"
    (src / "pkg" / "__pycache__").mkdir(parents=True)
    (src / "pkg" / "__pycache__" / "x.pyc").write_text("junk", encoding="ascii")
    (src / "env" / "uv-cache").mkdir(parents=True)
    (src / "env" / "uv-cache" / "locked").write_text("junk", encoding="ascii")
    (src / "pkg" / "keep.py").write_text("ok", encoding="utf-8")
    dst = workdir / "dst"
    n = mod._copy_tree_excluding(src, dst)
    assert (dst / "pkg" / "keep.py").exists()
    assert not (dst / "pkg" / "__pycache__").exists()
    assert not (dst / "env" / "uv-cache").exists()
    assert n >= 1


def test_install_from_bundle_rejects_non_bundle(workdir: Path) -> None:
    mod = _load()
    logs: list[str] = []
    runner = mod.Runner(logs.append)
    empty = workdir / "not-a-bundle"
    empty.mkdir()
    ok = mod.install_from_bundle(empty, workdir / "target", mod.Plan(install_dir=workdir / "target"), runner)
    assert ok is False
    assert any("bundle.json" in line for line in logs)


def test_make_bundle_wheels_requires_wheelhouse(workdir: Path, monkeypatch) -> None:
    mod = _load()
    root = workdir / "install"
    root.mkdir()
    logs: list[str] = []
    runner = mod.Runner(logs.append)
    ok = mod.make_bundle(root, workdir / "out", "wheels", ["core"], None, runner)
    assert ok is False
    assert any("wheelhouse" in line for line in logs)


def test_docs_mention_bundle(monkeypatch) -> None:
    """文档里必须写到离线包，否则用户找不到。"""
    doc = ROOT / "docs" / "07-安装.md"
    if not doc.exists():
        pytest.skip("没有安装文档")
    text = doc.read_text(encoding="utf-8")
    assert "--make-bundle" in text or "离线包" in text
