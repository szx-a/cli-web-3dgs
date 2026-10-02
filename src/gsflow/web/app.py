"""Web GUI（FastAPI）：与 CLI 完全同构的入口。

设计原则：
  * 这一层只做“参数校验 + 调用流程核心 + 事件流推送”，不含任何重建逻辑；
  * 每个 POST 的行为都等价于一条 CLI 命令（前端会显示对应命令，便于复制到终端）；
  * API Key / HF Token 只在服务端读写 config/credentials.json，响应里一律脱敏。
"""

from __future__ import annotations

import asyncio
import json
import time
import webbrowser
from pathlib import Path
from typing import Any

from .. import credentials, paths
from ..config import load as load_settings
from ..config import write_default_config
from ..pipeline import registry
from ..pipeline.orchestrator import JobSpec, list_jobs, load_job
from ..stages import register_all  # noqa: F401
from ..tools import doctor as doctor_mod

register_all()
# 与 CLI 一致：先把第三方库的配置/缓存目录重定向到工作区
paths.apply_runtime_env()

STATIC_DIR = Path(__file__).parent / "static"


def create_app():
    try:
        from fastapi import Body, FastAPI, HTTPException, Query
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "未安装 Web 依赖，请先执行：pip install fastapi 'uvicorn[standard]'\n"
            f"（原始错误：{exc}）"
        )

    from .jobs import JobManager

    app = FastAPI(title="gsflow", description="AI 图像清理 + 3DGS 全流程 GUI", version="0.1.0")
    manager = JobManager()

    # ------------------------------------------------------------------ 静态页
    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/static/{name}")
    def static_file(name: str):
        p = STATIC_DIR / name
        if not p.exists() or p.parent != STATIC_DIR:
            raise HTTPException(404, "not found")
        media = "text/css" if name.endswith(".css") else "application/javascript" if name.endswith(".js") else "text/plain"
        return FileResponse(p, media_type=media)

    # ------------------------------------------------------------------ 环境
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "time": time.time(), "workspace": str(paths.WORKSPACE_DIR)}

    @app.get("/api/doctor")
    def doctor(save: bool = False) -> dict[str, Any]:
        report = doctor_mod.collect_report(load_settings())
        if save:
            doctor_mod.write_report(report)
        return report

    @app.get("/api/stages")
    def stages() -> list[dict[str, Any]]:
        return registry.describe()

    # ------------------------------------------------------------------ 配置
    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return {"config": load_settings().data, "config_path": str(paths.CONFIG_DIR / "config.toml")}

    @app.post("/api/config")
    def set_config(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        settings = load_settings()
        updates = payload.get("set", payload)
        if not isinstance(updates, dict):
            raise HTTPException(400, "set 需要是 {key: value}")
        for k, v in updates.items():
            try:
                settings.set(k, v)
            except Exception as exc:
                raise HTTPException(400, str(exc)) from exc
        path = settings.save()
        return {"ok": True, "path": str(path)}

    @app.post("/api/config/reset")
    def reset_config() -> dict[str, Any]:
        path = write_default_config()
        return {"ok": True, "path": str(path)}

    # ------------------------------------------------------------------ 凭据
    @app.get("/api/credentials")
    def get_credentials() -> dict[str, Any]:
        return {"masked": credentials.mask_credentials(), "known_keys": credentials.KNOWN_KEYS}

    @app.post("/api/credentials")
    def set_credentials(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        data = {k: v for k, v in payload.items() if k in credentials.KNOWN_KEYS}
        credentials.save_credentials(data)
        return {"ok": True, "masked": credentials.mask_credentials()}

    # ------------------------------------------------------------------ 作业
    @app.get("/api/jobs")
    def jobs() -> dict[str, Any]:
        return {
            "running": manager.list(),
            "history": list_jobs(Path(str(load_settings().get("project.workspace_dir")))),
        }

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str) -> dict[str, Any]:
        try:
            return load_job(job_id, Path(str(load_settings().get("project.workspace_dir"))))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/run")
    def run_job(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        settings = load_settings(payload.get("params") or {})
        spec = JobSpec(
            input_path=payload.get("input_path", ""),
            images_dir=payload.get("images_dir", ""),
            job_id=payload.get("job_id", "") or "",
            title=payload.get("title", "3DGS 重建"),
            stages=[s for s in (payload.get("stages") or []) if s],
            params=payload.get("params") or {},
            force_stages=payload.get("force_stages") or [],
            halt_on_error=bool(payload.get("halt_on_error", True)),
        )
        if not spec.input_path and not spec.images_dir:
            if not payload.get("job_id"):
                raise HTTPException(400, "需要提供 images_dir / input_path，或指定已有 job_id 续跑")
        handle = manager.submit(spec, dry_run=bool(payload.get("dry_run", False)))
        return {
            "run_id": handle.run_id,
            "job_id": handle.job_id,
            "cli_equivalent": _cli_command(spec),
            "dry_run": bool(payload.get("dry_run", False)),
        }

    @app.post("/api/run/{run_id}/cancel")
    def cancel(run_id: str) -> dict[str, Any]:
        return {"ok": manager.cancel(run_id)}

    @app.get("/api/run/{run_id}/events")
    def events(run_id: str, since: int = 0) -> dict[str, Any]:
        return {"events": manager.events(run_id, since)}

    @app.get("/api/run/{run_id}/stream")
    async def stream(run_id: str, since: int = 0):
        """SSE 事件流：GUI 的实时日志与进度条都吃这个。"""

        async def gen():
            cursor = since
            idle = 0
            while True:
                handle = manager.get(run_id)
                if handle is None:
                    yield f"event: error\ndata: {json.dumps({'message': 'run not found'})}\n\n"
                    return
                batch = handle.bus.history(cursor)
                for evt in batch:
                    cursor = max(cursor, evt.seq)
                    yield f"data: {evt.to_json()}\n\n"
                    idle = 0
                if handle.status in ("done", "failed", "cancelled"):
                    yield f"event: end\ndata: {json.dumps({'status': handle.status, 'result': handle.result, 'error': handle.error})}\n\n"
                    return
                idle += 1
                if idle % 100 == 0:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.35)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    # ------------------------------------------------------------------ 产物
    @app.get("/api/artifacts/{job_id}/{path:path}")
    def artifact(job_id: str, path: str):
        root = Path(str(load_settings().get("project.workspace_dir"))) / job_id
        target = (root / path).resolve()
        if not str(target).startswith(str(root.resolve())):
            raise HTTPException(403, "路径越界")
        if not target.exists() or not target.is_file():
            raise HTTPException(404, "文件不存在")
        return FileResponse(target)

    @app.get("/api/logs/{job_id}")
    def job_logs(job_id: str, tail: int = Query(400, ge=1, le=5000)) -> dict[str, Any]:
        from ..events import read_jsonl

        f = Path(str(load_settings().get("project.workspace_dir"))) / job_id / "logs" / "events.jsonl"
        events = read_jsonl(f)
        return {"events": [e.to_json() for e in events[-tail:]]}

    return app


def _cli_command(spec: JobSpec) -> str:
    """给 GUI 显示"等价的 CLI 命令"，方便用户复制到终端或脚本里。

    注意：params 可能是嵌套字典（GUI 提交时就是），
    必须拍平成 `--set a.b=v` 的点路径形式，否则会渲染成 Python 的 dict 字面量、
    复制到终端根本无法执行。
    """
    parts = ["gsflow run"]
    if spec.images_dir:
        parts.append(f'--images "{spec.images_dir}"')
    elif spec.input_path:
        parts.append(f'--video "{spec.input_path}"')
    if spec.job_id:
        parts.append(f"--job-id {spec.job_id}")
    if spec.title and spec.title != "3DGS 重建":
        parts.append(f'--title "{spec.title}"')
    if spec.stages:
        parts.append("--stages " + ",".join(spec.stages))
    if spec.force_stages:
        parts.append("--force " + ",".join(spec.force_stages))
    for k, v in _flatten_params(spec.params or {}).items():
        parts.append(f"--set {k}={_fmt_value(v)}")
    return " ".join(parts)


def _flatten_params(params: dict, prefix: str = "") -> dict:
    """把 {'mask': {'backend': 'classical'}} 拍平成 {'mask.backend': 'classical'}。"""
    out: dict = {}
    for k, v in params.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_params(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def _fmt_value(v) -> str:
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(str(x) for x in v) + "]"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def serve(host: str = "127.0.0.1", port: int = 8787, open_browser: bool = False) -> None:
    import uvicorn

    paths.ensure_dirs()
    app = create_app()
    url = f"http://{host}:{port}"
    print(f"gsflow GUI: {url}")
    print(f"工作区: {paths.WORKSPACE_DIR}")
    if open_browser:
        threading_open(url)
    uvicorn.run(app, host=host, port=port, log_level="info")


def threading_open(url: str) -> None:
    import threading

    def _open() -> None:
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()
