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
    for name in ("app.js", "style.css", "viewer.js"):
        r = client.get(f"/static/{name}")
        assert r.status_code == 200 and len(r.content) > 500


def test_gui_has_model_viewer_and_no_about_tab(client) -> None:
    """GUI 结构契约：有「模型查看」页、且「说明」页已移除。

    这条测试守着"每个功能单独一个页面"的排版要求 ——
    说明页被并掉、模型查看有独立页面，都别在后续改动里被回退。
    """
    html = client.get("/").text
    assert 'data-tab="models"' in html
    assert 'id="panel-models"' in html
    assert "/static/viewer.js" in html
    assert 'data-tab="about"' not in html
    assert 'id="panel-about"' not in html


def _mk_job(ws: Path, job: str, files: list[tuple[str, str, str]]) -> None:
    """造一个作业目录：写若干占位产物 + 一份 manifest。

    files 的元素是 (kind, 文件名, 说明)。
    """
    out = ws / job / "05_export"
    out.mkdir(parents=True, exist_ok=True)
    arts = []
    for kind, name, desc in files:
        (out / name).write_bytes(b"x" * 64)
        arts.append({"kind": kind, "path": name, "desc": desc, "size_mb": 0.0})
    (out / "manifest.json").write_text(
        json.dumps({"artifacts": arts, "complete": True, "gaussians": 10},
                   ensure_ascii=False), encoding="utf-8")


def test_models_endpoint_classifies_three_modes(client, workdir: Path, monkeypatch) -> None:
    """`/api/models` 必须把产物准确分到三种查看模式。

    用户明确要求三种查看方式：点云 / 网格 / 带贴图网格。
    这个端点还有一条**兜底**：manifest 缺失时按目录扫描 ——
    因为历史上有作业只留下了 glb/obj/ply 却没有 manifest。
    """
    ws = workdir / "ws_models"
    _mk_job(ws, "jobA", [
        ("glb_points", "model.glb", "点云 glTF"),
        ("mesh_glb", "model_mesh.glb", "三角网格 glTF"),
        ("mesh_obj", "model_mesh.obj", "三角网格 OBJ"),
        ("mesh_ply", "model_mesh.ply", "三角网格 PLY"),
        ("mesh_textured_glb", "model_mesh_textured.glb", "带贴图网格 glTF"),
        ("mesh_texture", "model_mesh_texture.png", "烘焙贴图"),
    ])
    # 造一个"没有 manifest"的作业，验证兜底
    outB = ws / "jobB" / "05_export"
    outB.mkdir(parents=True, exist_ok=True)
    for n in ("model.glb", "model_mesh.glb"):
        (outB / n).write_bytes(b"y" * 64)

    from gsflow.config import load as load_settings

    s = load_settings()
    monkeypatch.setattr(
        "gsflow.web.app.load_settings",
        lambda *a, **k: _Wrap(s, ws),
        raising=False,
    )

    # monkeypatch 之后重新建 app，保证端点用新的 workspace
    from fastapi.testclient import TestClient

    from gsflow.web.app import create_app

    with TestClient(create_app()) as c:
        a = c.get("/api/models/jobA").json()
        assert a["manifest"] is True
        assert a["counts"]["points"] == 1, a
        assert a["counts"]["mesh"] == 3, a
        assert a["counts"]["textured"] == 2, a   # 带贴图 glb + 贴图 png
        names = {i["name"] for i in a["modes"]["textured"]}
        assert "model_mesh_textured.glb" in names

        b = c.get("/api/models/jobB").json()
        assert b["manifest"] is False
        assert b["counts"]["points"] == 1 and b["counts"]["mesh"] == 1, b
        assert "兜底" in (b.get("note") or "")


class _Wrap:
    """让 load_settings() 返回的配置指向测试用 workspace。"""

    def __init__(self, settings, ws: Path) -> None:
        self._s = settings
        self._ws = ws

    def get(self, key, default=None):
        if key == "project.workspace_dir":
            return str(self._ws)
        return self._s.get(key, default)

    def set(self, *a, **k):
        return self._s.set(*a, **k)


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
