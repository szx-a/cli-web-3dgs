"""安装器契约测试：**装完必须把配置改成本机路径**。

## 背景（2026-10-25，F:\\3dgs-flow 真机实测抓到）

安装器把项目本体复制到目标目录，而 `config/config.toml` 是"本机配置"，也在复制范围内。
结果是新装好的机器上 `workspace_dir` 仍然指向**源安装**：

    [project]
    workspace_dir = "F:\\\\3dgs-and-ai\\\\workspace"

这个坑很隐蔽：界面一切正常，但产物会写到别的目录（或因为无权限直接失败）。
修法是安装器在挂载 `src` 之后执行 `gsflow init --force` 重写配置，并校验内容。

这些测试是**静态契约**（本仓库没有安装器运行时测试环境），拦的是"有人把这一步删掉"。
"""
from __future__ import annotations

from pathlib import Path

INSTALLER = Path(__file__).resolve().parents[1] / "installer" / "gsflow_installer.py"


def test_installer_rewrites_local_config() -> None:
    src = INSTALLER.read_text(encoding="utf-8")
    assert '"init", "--force"' in src, "安装器没有用 gsflow init --force 重写本机配置"
    assert "配置已指向安装目录" in src, "安装器没有校验配置是否指向安装目录"


def test_installer_materializes_payload_before_venv() -> None:
    """空目录里没有 src/ 可挂载，所以必须先复制项目本体再建 venv。"""
    src = INSTALLER.read_text(encoding="utf-8")
    assert "def _materialize_payload" in src
    i_copy = src.find("_materialize_payload(root")
    # venv 创建是动态拼的：[uv, "venv", str(venv), "--python", python_exe]
    i_venv = src.find('"venv", str(venv)')
    assert i_copy > 0 and i_venv > 0, "没找到复制或建 venv 的位置"
    assert i_copy < i_venv, "复制项目本体的步骤必须在建 venv 之前"


def test_installer_excludes_tests_and_dev_files() -> None:
    src = INSTALLER.read_text(encoding="utf-8")
    for name in ('"tests"', '"AGENTS.md"', '"workspace"'):
        assert name in src, f"复制清单没有排除 {name}"


def test_installer_keeps_caches_inside_install_dir() -> None:
    """所有缓存/临时目录必须在安装目录内（沙箱与非管理员环境都要求）。"""
    src = INSTALLER.read_text(encoding="utf-8")
    for key in ('"TEMP"', '"UV_CACHE_DIR"', '"PIP_CACHE_DIR"', '"TORCH_HOME"'):
        assert key in src, f"没有把 {key} 指到安装目录内"


def test_installer_supports_silent_and_check_only() -> None:
    src = INSTALLER.read_text(encoding="utf-8")
    assert '"--silent"' in src and '"--check-only"' in src
    assert '"--reuse-from"' in src, "缺少复用本机缓存（实测网络不稳时是刚需）"
