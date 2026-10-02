"""Web 层测试：用 FastAPI TestClient 在进程内验证路由（不需要真正开端口）。

运行：env\\venv\\Scripts\\python.exe -m pytest tests -q -p no:cacheprovider
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

fastapi = pytest.importorskip("fastapi")
from gsflow.web.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health(client) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_index_serves_spa(client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "gsflow" in r.text
    assert 'id="panel-new"' in r.text


def test_static_assets(client) -> None:
    for name in ("app.js", "style.css"):
        r = client.get(f"/static/{name}")
        assert r.status_code == 200 and len(r.content) > 500


def test_stages_endpoint(client) -> None:
    r = client.get("/api/stages")
    names = [s["name"] for s in r.json()]
    assert names == ["frames", "mask", "colmap", "train", "export", "metrics"]


def test_config_get_and_set(client) -> None:
    r = client.get("/api/config")
    assert r.status_code == 200
    cfg = r.json()["config"]
    assert "train" in cfg and "mask" in cfg

    original = cfg["colmap"]["matcher"]
    r2 = client.post("/api/config", json={"set": {"colmap.matcher": "sequential"}})
    assert r2.status_code == 200
    assert client.get("/api/config").json()["config"]["colmap"]["matcher"] == "sequential"

    # 还原，避免影响后续真实作业
    client.post("/api/config", json={"set": {"colmap.matcher": original}})


def test_credentials_roundtrip_is_masked(client) -> None:
    r = client.post("/api/credentials", json={"hf_token": "hf_testtoken9876"})
    assert r.status_code == 200
    masked = r.json()["masked"]["hf_token"]
    assert masked.endswith("9876") and "testtoken" not in masked
    # 清理测试凭据
    client.post("/api/credentials", json={"hf_token": ""})


def test_artifact_path_traversal_blocked(client) -> None:
    r = client.get("/api/artifacts/job_x/..%2F..%2Fconfig%2Fconfig.toml")
    assert r.status_code in (403, 404)


def test_run_requires_input(client) -> None:
    r = client.post("/api/run", json={"title": "空输入"})
    assert r.status_code == 400


def test_cli_equivalent_flattens_nested_params() -> None:
    """GUI 显示的等价 CLI 命令必须能直接执行：嵌套参数要拍平成点路径。"""
    from gsflow.pipeline.orchestrator import JobSpec
    from gsflow.web.app import _cli_command

    spec = JobSpec(
        images_dir=r"D:\data\photos",
        title="测试",
        stages=["frames", "mask"],
        params={"mask": {"backend": "classical", "dilate_px": 2}, "train.iterations": 7000},
    )
    cmd = _cli_command(spec)
    assert "--set mask.backend=classical" in cmd
    assert "--set mask.dilate_px=2" in cmd
    assert "--set train.iterations=7000" in cmd
    assert "{" not in cmd and "}" not in cmd, f"命令里不该出现 dict 字面量：{cmd}"
    assert "'" not in cmd.replace(r"D:\data", ""), f"命令里不该出现 Python 引号：{cmd}"
