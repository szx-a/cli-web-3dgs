"""pytest 公共配置。

为什么不用 pytest 自带的 tmp_path / basetemp：
本机受限环境下，pytest 在 session 收尾时会对临时目录做 os.scandir + chmod 清理，
触发 PermissionError（WinError 5），不但用例报错，还会把测试临时目录的权限项弄坏
（之后连 mkdir 都会失败，需要 icacls /reset 修复）。

所以这里：
  1. 把临时根固定到工作区内 `env/pytest-tmp`；
  2. 提供不依赖 pytest 清理逻辑的 `workdir` fixture。
运行测试请加 `-p no:cacheprovider`（缓存目录写入同样会被拒）。
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path(os.environ.get("GSFLOW_TEST_TMP", str(ROOT / "env" / "pytest-tmp")))


def pytest_configure(config) -> None:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(TMP_ROOT)


@pytest.fixture()
def workdir() -> Path:
    """自建临时目录（不依赖 pytest 的 tmp_path 生命周期）。"""
    d = TMP_ROOT / f"t-{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)
