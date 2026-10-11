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
        # ⚠️ 必须显式禁用缓存。默认 `HTMLResponse` 不带 Cache-Control，
        # 浏览器会**启发式缓存**首页；而我们改前端（index.html/app.js/viewer.js）很频繁，
        # 结果就是"代码改了、界面没变" —— 用户会以为修复没生效。
        # 用 `no-cache` 而不是 `no-store`：仍然允许条件请求（ETag/Last-Modified），
        # 命中就返回 304，不牺牲性能，但保证不会用过期的副本。
        return HTMLResponse(html, headers={"Cache-Control": "no-cache, must-revalidate"})

    @app.get("/static/{name}")
    def static_file(name: str):
        p = STATIC_DIR / name
        if not p.exists() or p.parent != STATIC_DIR:
            raise HTTPException(404, "not found")
        media = "text/css" if name.endswith(".css") else "application/javascript" if name.endswith(".js") else "text/plain"
        # 同上：静态资源也要 no-cache，否则前端改动在浏览器里看不到
        return FileResponse(p, media_type=media,
                            headers={"Cache-Control": "no-cache, must-revalidate"})

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

    @app.get("/api/models/{job_id}")
    def job_models(job_id: str) -> dict[str, Any]:
        """列出该作业可用于**查看**的模型产物，分为三种模式。

        三种模式是用户明确要求的查看方式：
          * `points`  —— 点云（PLY / 点云 GLB），最忠实反映 3DGS 的原始形态
          * `mesh`    —— 三角网格（无贴图），看几何是否成立
          * `textured`—— 带贴图的网格，看最终"实感"产物

        数据来源是 `05_export/manifest.json`（导出阶段会**增量刷新**，
        所以即使网格/贴图那一步失败或超时，已完成的产物也能列出）。
        文件不存在时不报错，只返回空列表 —— GUI 要能优雅处理"还没导出"。
        """
        root = Path(str(load_settings().get("project.workspace_dir"))) / job_id
        out = root / "05_export"
        modes: dict[str, list[dict[str, Any]]] = {"points": [], "mesh": [], "textured": []}
        info: dict[str, Any] = {"job_id": job_id, "manifest": False, "complete": False}

        man_p = out / "manifest.json"
        arts: list[dict[str, Any]] = []
        if man_p.exists():
            try:
                man = json.loads(man_p.read_text(encoding="utf-8"))
                arts = man.get("artifacts") or []
                info["manifest"] = True
                info["complete"] = bool(man.get("complete"))
                info["note"] = man.get("note", "")
                info["gaussians"] = man.get("gaussians")
            except Exception:  # noqa: BLE001
                info["note"] = "manifest.json 解析失败"

        # 兜底：没有 manifest 就直接扫目录（导出被中断时仍有产物可用）
        if not arts and out.exists():
            for p in sorted(out.iterdir()):
                if p.is_file():
                    arts.append({"kind": p.suffix.lstrip(".") or "file",
                                 "path": p.name,
                                 "size_mb": round(p.stat().st_size / 1024 / 1024, 2)})
            if arts:
                info["note"] = info.get("note") or "manifest 缺失，已按目录扫描兜底"

        # 查看器只支持 GLB 与 PLY 两种容器（见 static/viewer.js 顶部说明）。
        # OBJ / SPZ 仍然导出（给 Blender 等外部工具用），但**不能放进"可查看"列表** ——
        # 否则用户点进去只会看到一片黑，却不知道是格式不支持。
        VIEWABLE = {".glb", ".ply"}

        def _add(mode: str, kind: str, name: str, desc: str = "") -> None:
            p = out / name
            if not p.exists():
                return
            if p.suffix.lower() not in VIEWABLE:
                return
            modes[mode].append({
                "kind": kind, "name": name, "desc": desc,
                "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
                "url": f"/api/artifacts/{job_id}/05_export/{name}",
            })

        def _add_download(kind: str, name: str, desc: str = "") -> None:
            """记录一个**可下载但不可查看**的产物（OBJ/SPZ 等）。"""
            p = out / name
            if not p.exists():
                return
            info.setdefault("downloadable", []).append({
                "kind": kind, "name": name, "desc": desc,
                "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
                "url": f"/api/artifacts/{job_id}/05_export/{name}",
            })

        for a in arts:
            name = str(a.get("path") or "")
            kind = str(a.get("kind") or "")
            desc = str(a.get("desc") or "")
            low = name.lower()
            if low.endswith(".glb") and "mesh" in low and "textured" in low:
                _add("textured", "glb_textured", name, desc or "带贴图网格 glTF")
            elif low.endswith(".glb") and "mesh" in low:
                _add("mesh", "glb_mesh", name, desc or "三角网格 glTF（顶点色）")
            elif low.endswith(".obj") and "mesh" in low:
                # 三角网格 OBJ：查看器不解析 OBJ，只能提供下载
                _add_download("obj_mesh", name, desc or "三角网格 OBJ（用 Blender 等打开）")
            elif low.endswith(".ply") and "mesh" in low:
                _add("mesh", "ply_mesh", name, desc or "三角网格 PLY")
            elif low.endswith(".glb"):
                _add("points", "glb_points", name, desc or "点云 glTF")
            elif low.endswith(".ply"):
                _add("points", "ply", name, desc or "3DGS 点云 PLY")
            elif low.endswith(".spz"):
                _add_download("spz", name, desc or "SPZ 压缩点云（查看器不支持）")
            elif low.endswith(".obj"):
                _add_download("obj", name, desc or "点云 OBJ（查看器不支持）")
            elif low.endswith(".png") and "texture" in low:
                _add_download("texture_png", name, desc or "烘焙贴图（贴图文件）")
            elif low.endswith("viewer.html"):
                info["viewer"] = f"/api/artifacts/{job_id}/05_export/{name}"

        info["modes"] = modes
        info["counts"] = {k: len(v) for k, v in modes.items()}
        return info

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
