"""Web GUI（FastAPI）：与 CLI 完全同构的入口。

设计原则：
  * 这一层只做“参数校验 + 调用流程核心 + 事件流推送”，不含任何重建逻辑；
  * 每个 POST 的行为都等价于一条 CLI 命令（前端会显示对应命令，便于复制到终端）；
  * API Key / HF Token 只在服务端读写 config/credentials.json，响应里一律脱敏。
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import threading
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
        from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, Response,
                               StreamingResponse)
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "未安装 Web 依赖，请先执行：pip install fastapi 'uvicorn[standard]'\n"
            f"（原始错误：{exc}）"
        )

    from .jobs import JobManager

    app = FastAPI(title="gsflow", description="AI 图像清理 + 3DGS 全流程 GUI", version="0.1.0")
    manager = JobManager()

    def _normalize_artifacts(data: dict[str, Any], job_id: str) -> dict[str, Any]:
        """把每个阶段的产物路径归一化成**可下载的相对路径 + URL**。

        为什么必须做（这就是用户看到 `Failed to load resource: 404` 的根因）：
        `job.json` 的 `stages[*].artifacts` 里存的是**绝对路径**
        （如 `F:\\3dgs-and-ai\\workspace\\verify-e2e2\\01_images_raw`），
        而前端把它直接拼进 URL：
            /api/artifacts/<job>/F%3A%5C3dgs-and-ai%5Cworkspace%5C...
        绝对路径在工作区之外，服务器**无论怎么解码都取不到** ⇒ 必然 404。
        （前端当时还有个正则层数写错的问题，让反斜杠没被替换，双重叠加。）

        修法：服务端负责把它变成相对路径并给出最终 URL；前端不再自己拼路径。
        同时保留原始值在 `path` 字段里，便于排查。
        """
        root = (Path(str(load_settings().get("project.workspace_dir"))) / job_id).resolve()
        stages = data.get("stages")
        if not isinstance(stages, dict):
            return data
        for _name, st in stages.items():
            if not isinstance(st, dict):
                continue
            arts = st.get("artifacts")
            if not isinstance(arts, list):
                continue
            norm: list[dict[str, Any]] = []
            for a in arts:
                raw = str(a)
                is_dir = False
                try:
                    p = Path(raw)
                    if p.is_absolute():
                        rp = p.resolve()
                        rel = rp.relative_to(root).as_posix() if rp.is_relative_to(root) \
                            else p.name
                        is_dir = rp.is_dir()
                    else:
                        rel = p.as_posix()
                        is_dir = (root / p).is_dir()
                except (ValueError, OSError):
                    # 不在该作业目录内（或不是路径）：退回文件名，至少给个可用链接
                    rel = Path(raw.replace("\\", "/")).name
                norm.append({
                    "path": raw,
                    "rel": rel,
                    "name": Path(rel).name,
                    # 目录也要给出 url（便于排查），但前端据 is_dir 决定是否可点：
                    # 端点对目录返回 404（它只服务文件），把目录渲染成链接就是"点了报 404"。
                    "url": f"/api/artifacts/{job_id}/{rel}",
                    "is_dir": bool(is_dir),
                })
            st["artifacts"] = norm
        return data

    # ------------------------------------------------------------------ 静态页
    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        # ⚠️ 必须显式禁用缓存。默认 `HTMLResponse` 不带 Cache-Control，
        # 浏览器会**启发式缓存**首页；而我们改前端（index.html/app.js/viewer.js）很频繁，
        # 结果就是"代码改了、界面没变" —— 用户会以为修复没生效。
        # 用 `no-cache` 而不是 `no-store`：仍然允许条件请求（ETag/Last-Modified），
        # 命中就返回 304，不牺牲性能，但保证不会用过期的副本。
        #
        # ---- 给脚本引用加**内容版本号**（防新旧混用）----
        #
        # 真实风险：用户页面开着的时候我改了 `viewer.js`/新增了 `splat.js`，
        # 浏览器可能只重新拉到一部分脚本 ⇒ 新版 `app.js` 调用 `splat` 模式，
        # 而旧版 `viewer.js` 不认识 ⇒ 画面全黑、控制台还可能出现 404。
        # 用文件 mtime 作为 `?v=`，任何脚本一改，URL 就变，缓存必然失效。
        def _v(name: str) -> str:
            f = STATIC_DIR / name
            try:
                return f"{int(f.stat().st_mtime)}"
            except Exception:  # noqa: BLE001
                return "0"

        for script in ("splat.js", "viewer.js", "app.js"):
            html = re.sub(
                r'src="/static/' + re.escape(script) + r'"',
                f'src="/static/{script}?v={_v(script)}"', html)
        return HTMLResponse(html, headers={"Cache-Control": "no-cache, must-revalidate"})

    @app.get("/api/selftest")
    def selftest(job: str = "") -> dict[str, Any]:
        """**自检**：一次回答"前端为什么打不开/全黑"。

        只靠客户端报错很难定位 —— 这里在服务端把关键条件都验证一遍：
          * 三个前端脚本是否都存在（缺 `splat.js` ⇒ 泼溅不可用 ⇒ 全黑）；
          * 指定作业的 `/api/models` 里每个 URL 是否真的能取到；
          * `/api/gauss/{job}`（完整 59 维高斯）是否可用；
          * manifest 里每条产物路径是否都能取到。
        """
        root = Path(str(load_settings().get("project.workspace_dir")))
        rep: dict[str, Any] = {"static": {}, "job": job or None, "problems": []}
        for s in ("index.html", "splat.js", "viewer.js", "app.js"):
            f = STATIC_DIR / s
            rep["static"][s] = {"exists": f.exists(),
                                "bytes": f.stat().st_size if f.exists() else 0}
            if not f.exists():
                rep["problems"].append(f"缺少前端文件 {s}")
        if job:
            jd = root / job
            if not jd.is_dir():
                rep["problems"].append(f"作业不存在：{job}")
            else:
                out = jd / "05_export"
                man = out / "manifest.json"
                rep["job_dir"] = str(jd)
                rep["has_export"] = out.is_dir()
                rep["has_manifest"] = man.exists()
                if man.exists():
                    try:
                        md = json.loads(man.read_text(encoding="utf-8"))
                        miss = [a.get("path") for a in (md.get("artifacts") or [])
                                if a.get("path") and not (out / a["path"]).exists()]
                        rep["manifest_artifacts"] = len(md.get("artifacts") or [])
                        rep["manifest_missing"] = miss
                        if miss:
                            rep["problems"].append(
                                f"manifest 声明了 {len(miss)} 个不存在的产物"
                                f"（下载会 404）：{miss[:3]}")
                    except Exception as exc:  # noqa: BLE001
                        rep["problems"].append(f"manifest 解析失败：{exc}")
                # 完整 3DGS PLY
                g = None
                mp = jd / "04_train" / "metrics.json"
                if mp.exists():
                    try:
                        g = json.loads(mp.read_text(encoding="utf-8")).get("best_ply")
                    except Exception:  # noqa: BLE001
                        g = None
                gp = Path(g) if g else None
                if gp and not gp.is_absolute():
                    gp = jd / gp
                ok_g = bool(gp and gp.exists())
                if not ok_g:
                    cands = sorted((jd / "04_train" / "point_cloud").glob(
                        "*/point_cloud.ply"))
                    ok_g = bool(cands)
                    gp = cands[-1] if cands else None
                rep["gauss3d"] = {"available": ok_g,
                                  "path": str(gp) if gp else None,
                                  "bytes": gp.stat().st_size if ok_g and gp else 0}
                if not ok_g:
                    rep["problems"].append(
                        "没有训练产物 point_cloud.ply ⇒ 泼溅模式（真实高斯）不可用")
        rep["ok"] = not rep["problems"]
        return rep

    @app.get("/static/{path:path}")
    def static_file(path: str):
        """静态资源（支持子目录，如 `vendor/three.module.js`）。

        ⚠️ 原来是 `/static/{name}`，只接受**单层文件名**，且用
        `p.parent != STATIC_DIR` 做防穿越。要服务 `vendor/` 下的参考渲染库
        就必须支持多级路径 —— 这里改成**先 resolve 再校验前缀**，
        既支持子目录又不放松穿越防护。
        """
        p = (STATIC_DIR / path).resolve()
        if not p.exists() or not p.is_file():
            raise HTTPException(404, "not found")
        try:
            p.relative_to(STATIC_DIR.resolve())
        except ValueError:
            raise HTTPException(404, "not found") from None
        low = path.lower()
        if low.endswith(".css"):
            media = "text/css"
        elif low.endswith(".js") or low.endswith(".mjs"):
            media = "application/javascript"
        elif low.endswith(".json") or low.endswith(".map"):
            media = "application/json"
        elif low.endswith(".png"):
            media = "image/png"
        elif low.endswith((".jpg", ".jpeg")):
            media = "image/jpeg"
        elif low.endswith(".svg"):
            media = "image/svg+xml"
        else:
            media = "text/plain"
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
    @app.get("/api/capture_modes")
    def capture_modes() -> dict[str, Any]:
        """列出"拍摄模式"的选项与各自生成的参数。

        与 `/api/opacity_cleanup` 同样的理由：CLI 的 `--capture-mode` 与 GUI 下拉框
        必须产出**完全一致**的参数集合，所以映射只在
        `gsdata/capture_mode_params.py` 一处定义，前端从这里取。
        """
        from ..gsdata.capture_mode_params import CHOICES, capture_mode_params

        return {
            "choices": [
                {"value": c, "params": capture_mode_params(c)}
                for c in CHOICES
            ],
        }

    @app.get("/api/opacity_cleanup")
    def opacity_cleanup_options() -> dict[str, Any]:
        """列出"不透明度治理方案"的选项与各自生成的参数。

        为什么要有这个接口：CLI 的 `--opacity-cleanup` 与 GUI 的下拉框必须产出
        **完全相同**的参数集合，否则两端会漂移（本项目已多次因"同一决定写两遍"
        而出错，最严重的一次是 22 个脚本都用 sh_degree=0 渲染）。
        所以映射只在 `trainers/opacity_cleanup.py` 一处定义，
        前端通过本接口取，不在 JS 里重写。
        """
        from ..trainers.opacity_cleanup import CHOICES, opacity_cleanup_params

        return {
            "choices": [
                {"value": c, "params": opacity_cleanup_params(c)}
                for c in CHOICES
            ],
        }

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
    @app.get("/api/scene_info/{job_id}")
    def scene_info(job_id: str) -> dict[str, Any]:
        """给出参考实现（three.js 高斯渲染库）需要的场景参数。

        三项都要**算**出来，不能猜：
          * `center`/`radius`：高斯包围盒（与查看器取景一致）；
          * `up`：**训练相机中心点云的最小方差主轴**。COLMAP 世界坐标没有约定
            "哪个轴朝上"，GLB 导出时是靠人工 y/z 翻转的；直接把 [0,1,0] 当上方向
            会让模型躺倒。相机是绕着物体转的，它们近似共面，**该平面的法向就是上方向**。
          * `cameraPosition`：沿 up 抬高约 12°，距离 2.8×半径（与默认取景一致）。
        """
        import numpy as np

        g = _gauss_for(job_id)
        xyz = np.asarray(g["xyz"], dtype=np.float64)
        mn, mx = xyz.min(0), xyz.max(0)
        center = (mn + mx) / 2.0
        radius = max(1e-6, float(np.linalg.norm(mx - mn) / 2.0))

        up = np.array([0.0, 1.0, 0.0])
        cams_p = None
        root = (Path(str(load_settings().get("project.workspace_dir")))
                / job_id).resolve()
        try:
            from ..gsdata.colmap_io import read_model

            cands = sorted({p.parent for p in
                            (root / "03_colmap" / "sparse").rglob("cameras.bin")})
            best = None
            for c in cands:
                rec = read_model(c)
                if best is None or len(rec.images) > len(best.images):
                    best = rec
            if best is not None and len(best.images) >= 3:
                cams_p = np.array([im.cam_to_world()[1] for im in best.images])
        except Exception:  # noqa: BLE001
            cams_p = None

        if cams_p is not None and len(cams_p) >= 3:
            d = cams_p - cams_p.mean(axis=0)
            # 最小方差方向 = 相机所在平面的法向
            _, _, vt = np.linalg.svd(d, full_matrices=False)
            n = vt[-1] / max(np.linalg.norm(vt[-1]), 1e-12)
            # 让法向大致朝上（与质心到相机的平均方向同侧就不再翻转）
            if float(np.dot(n, np.mean(cams_p - cams_p.mean(0), axis=0))) < 0:
                pass
            # 取更接近 +Y 的那个朝向作为"上"，保证不同作业一致
            if float(np.dot(n, np.array([0.0, 1.0, 0.0]))) < 0:
                n = -n
            up = n

        # 相机位置：沿 up 抬高 12°，距离 2.8R，朝向 -Z 方向偏移
        elev = math.radians(12.0)
        horiz = math.cos(elev)
        fwd_hint = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(fwd_hint, up))) > 0.9:
            fwd_hint = np.array([1.0, 0.0, 0.0])
        side = np.cross(up, fwd_hint)
        side /= max(np.linalg.norm(side), 1e-12)
        back = np.cross(side, up)
        d = radius * 2.8
        cam = center + (back * horiz + up * math.sin(elev)) * d
        return {"center": [float(x) for x in center],
                "radius": float(radius),
                "up": [float(x) for x in up],
                "cameraPosition": [float(x) for x in cam],
                "cameraLookAt": [float(x) for x in center],
                "gaussians": int(len(xyz)),
                "hasPoses": cams_p is not None}

    @app.get("/api/jobs")
    def jobs() -> dict[str, Any]:
        return {
            "running": manager.list(),
            "history": list_jobs(Path(str(load_settings().get("project.workspace_dir")))),
        }

    # ------------------------------------------------------------------ 服务端渲染
    #
    # 为什么要有这条路由：GUI 里的"泼溅"是我用 WebGL **重新实现**的渲染器
    # （逐高斯桶排序、四边形 2σ 近似、固定低通），与 `preview_*.png` 用的
    # **gsplat 官方光栅化器**（逐像素排序、精确 EWA、完整 SH）必然有差距。
    # 用户反复看到"查看器里的模型比对照图差很多"，而任何 WebGL 复刻都消不掉这个差距。
    #
    # 正确做法：**按用户当前相机，用同一个 gsplat 渲染器出图**，返回 PNG。
    # 这样查看器和预览图**按构造完全一致**，不再依赖我复刻 WebGL。
    _RENDER_CACHE: dict[str, Any] = {}
    _RENDER_LOCK = threading.Lock()

    def _gauss_for(job_id: str):
        """按 (路径, mtime) 缓存高斯，避免每次渲染都读 29MB PLY。"""
        root = (Path(str(load_settings().get("project.workspace_dir")))
                / job_id).resolve()
        if not root.is_dir():
            raise HTTPException(404, f"作业不存在：{job_id}")
        ply = None
        mp = root / "04_train" / "metrics.json"
        if mp.exists():
            try:
                best = str(json.loads(mp.read_text(encoding="utf-8"))
                           .get("best_ply") or "")
                if best:
                    p = Path(best)
                    if not p.is_absolute():
                        p = root / p
                    if p.exists():
                        ply = p
            except Exception:  # noqa: BLE001
                ply = None
        if ply is None:
            cands = sorted((root / "04_train" / "point_cloud").glob("*/point_cloud.ply"))
            if cands:
                ply = cands[-1]
        if ply is None:
            raise HTTPException(404, "找不到训练产物 point_cloud.ply")
        ply = ply.resolve()
        if not str(ply).startswith(str(root)):
            raise HTTPException(400, "路径越界")
        key = f"{ply}::{ply.stat().st_mtime_ns}"
        hit = _RENDER_CACHE.get(key)
        if hit is None:
            from ..gsdata.shio import read_gaussians_full

            hit = read_gaussians_full(ply)
            # 只保留最近一份，避免长时间运行把显存/内存撑爆
            _RENDER_CACHE.clear()
            _RENDER_CACHE[key] = hit
        return hit

    @app.get("/api/render/{job_id}")
    def render(job_id: str, yaw: float = 0.0, pitch: float = 0.0,
               dist: float = 0.0, width: int = 720, height: int = 720):
        """用 gsplat 渲染一帧，返回 PNG。

        `yaw`/`pitch` 为角度（度），`dist` 为相机到物体中心的距离
        （<=0 时用 2.8×包围球半径，与查看器的默认取景一致）。
        """
        import io
        import math

        width = max(64, min(int(width), 1600))
        height = max(64, min(int(height), 1600))
        try:
            g = _gauss_for(job_id)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"读取高斯失败：{type(exc).__name__}: {exc}") from exc

        import numpy as np

        xyz = np.asarray(g["xyz"], dtype=np.float64)
        mn, mx = xyz.min(0), xyz.max(0)
        center = (mn + mx) / 2.0
        radius = max(1e-6, float(np.linalg.norm(mx - mn) / 2.0))
        d = float(dist) if dist and dist > 0 else radius * 2.8

        # 球坐标 → 相机位置（与查看器的约定一致：yaw 绕 Y，pitch 抬高）
        cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        direction = np.array([cp * sy, sp, cp * cy], dtype=np.float64)
        eye = center + direction * d

        # look-at：相机看 -Z，up 取 +Y
        fwd = center - eye
        fwd /= max(np.linalg.norm(fwd), 1e-12)
        up_world = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(fwd, up_world))) > 0.999:
            up_world = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up_world)
        right /= max(np.linalg.norm(right), 1e-12)
        up = np.cross(right, fwd)
        w2c = np.eye(4, dtype=np.float64)
        w2c[0, :3], w2c[1, :3], w2c[2, :3] = right, up, -fwd
        w2c[:3, 3] = -w2c[:3, :3] @ eye

        fov = math.radians(45.0)
        f = 0.5 * height / math.tan(fov / 2.0)
        K = np.array([[f, 0, width / 2.0],
                      [0, f, height / 2.0],
                      [0, 0, 1]], dtype=np.float64)

        try:
            from ..gsdata.shio import render_view

            # GPU 同一时刻只跑一个渲染：训练在跑时排队，而不是抢显存
            with _RENDER_LOCK:
                colors, alphas = render_view(g, w2c, K, width, height)
            img = np.asarray(colors.detach().cpu().numpy()
                             if hasattr(colors, "detach") else colors)
            a = np.asarray(alphas.detach().cpu().numpy()
                           if hasattr(alphas, "detach") else alphas)
            if img.ndim == 3 and img.shape[-1] >= 3:
                img = img[..., :3]
            if a.ndim == 3:
                a = a[..., 0]
            # 与 canvas 背景同色，便于对比
            bg = np.array([0.02, 0.03, 0.045], dtype=np.float32)
            comp = img * a[..., None] + bg * (1.0 - a[..., None])
            u8 = np.clip(comp * 255.0, 0, 255).astype(np.uint8)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                503, f"服务端渲染不可用（{type(exc).__name__}: {exc}）") from exc

        from PIL import Image as _Image

        buf = io.BytesIO()
        _Image.fromarray(u8).save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str) -> dict[str, Any]:
        try:
            data = load_job(job_id, Path(str(load_settings().get("project.workspace_dir"))))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return _normalize_artifacts(data, job_id)

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
    # 带 `.ply` 后缀的别名：参考渲染库（gaussian-splats-3d）**按文件扩展名**
    # 选择解析器，URL 没有 `.ply` 会被当成未知格式。所以给出这条带后缀的路由，
    # 内部完全复用 `gauss()`。
    @app.get("/api/gauss/{job_id}/point_cloud.ply")
    def gauss_ply_alias(job_id: str):
        return gauss(job_id)

    @app.get("/api/gauss/{job_id}")
    def gauss(job_id: str):
        """直接返回训练产物的**完整 3DGS PLY**（59 维/高斯，含 SH 48）。

        为什么不走 `/api/artifacts`：那个路由固定映射到 `05_export`，
        而完整高斯在 `04_train/point_cloud/iteration_*/point_cloud.ply`。
        复制一份到 `05_export` 会多占 28MB 且容易与训练产物不同步，
        所以这里按 job 直接定位并流式返回。

        ⚠️ 路径必须限制在 job 目录内（防目录穿越）。
        """
        from fastapi.responses import FileResponse

        root = (Path(str(load_settings().get("project.workspace_dir"))) / job_id).resolve()
        if not root.is_dir():
            raise HTTPException(status_code=404, detail=f"作业不存在：{job_id}")
        ply: Path | None = None
        mp = root / "04_train" / "metrics.json"
        if mp.exists():
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
                best = str(m.get("best_ply") or "")
                if best:
                    bp = Path(best)
                    if not bp.is_absolute():
                        bp = root / bp
                    bp = bp.resolve()
                    # 防目录穿越：解析后的路径必须在 job 目录之内
                    if bp.is_file() and str(bp).startswith(str(root)):
                        ply = bp
            except Exception:  # noqa: BLE001
                ply = None
        if ply is None:
            cands = sorted((root / "04_train" / "point_cloud").glob("*/point_cloud.ply"))
            if cands:
                c = cands[-1].resolve()
                if str(c).startswith(str(root)):
                    ply = c
        if ply is None:
            raise HTTPException(status_code=404,
                                detail="找不到训练产物 point_cloud.ply（先完成 train 阶段）")
        return FileResponse(str(ply), media_type="application/octet-stream",
                            filename="point_cloud.ply")

    @app.get("/api/artifacts/{job_id}/{path:path}")
    def artifact(job_id: str, path: str):
        # ---- 路径归一化（防御性，修 404 的第二道保险）----
        # 前端曾把 Windows 反斜杠编码成 %5C 传进来（app.js 的正则层数写错，
        # 见 G66），于是 URL 形如
        #   /api/artifacts/<job>/F%3A%5C3dgs-and-ai%5Cworkspace...
        # Starlette 会把它解码成含反斜杠的字符串，`root / path` 自然找不到文件 -> 404。
        # 前端已修，但服务端**也必须**自己归一化：URL 里的分隔符写法不该决定成败。
        p = path.replace("\\", "/")
        # 绝对路径（如 F:/... 或 /C:/...）一律视为越界尝试，不允许直接使用
        if p.startswith("/") or (len(p) > 1 and p[1] == ":"):
            raise HTTPException(403, "路径越界")
        root = (Path(str(load_settings().get("project.workspace_dir"))) / job_id).resolve()
        target = (root / p).resolve()
        # ⚠️ 必须用 relative_to 判断包含关系，**不能用字符串前缀**：
        #    `str(target).startswith(str(root))` 会让 `<workspace>/job-evil` 通过
        #    `<workspace>/job` 的检查（前缀相同），是一个真实的越界漏洞。
        try:
            target.relative_to(root)
        except ValueError:
            raise HTTPException(403, "路径越界") from None
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
        modes: dict[str, list[dict[str, Any]]] = {"splat": [], "points": [], "mesh": [],
                                                  "textured": []}
        info: dict[str, Any] = {"job_id": job_id, "manifest": False, "complete": False}

        # ---- 完整 3DGS PLY（59 维/高斯，含 SH 48）⇒ 泼溅渲染的真实形态 ----
        #
        # 为什么单独提供：`05_export` 里的 glb/ply 点云只有 6 维（xyz+rgb），
        # 用它渲染必然"稀疏、扁平"，用户会以为模型质量差。
        # 训练产物 `04_train/point_cloud/iteration_*/point_cloud.ply` 才是全集
        # （位置3 + 尺度3 + 旋转4 + 不透明度1 + SH 48 = 59）。
        # 直接给这个文件的 URL（不经 05_export，免去 28MB 复制）。
        try:
            metrics_p = root / "04_train" / "metrics.json"
            gauss_url = ""
            gauss_ply: Path | None = None
            if metrics_p.exists():
                _m = json.loads(metrics_p.read_text(encoding="utf-8"))
                best = str(_m.get("best_ply") or "")
                if best:
                    bp = Path(best)
                    if not bp.is_absolute():
                        bp = root / bp
                    if bp.exists():
                        gauss_url = f"/api/gauss/{job_id}"
                        gauss_ply = bp
            if not gauss_url:
                cands = sorted((root / "04_train" / "point_cloud").glob("*/point_cloud.ply"))
                if cands:
                    gauss_url = f"/api/gauss/{job_id}"
                    gauss_ply = cands[-1]
            if gauss_url:
                # ⚠️ `size_mb` 必须给**真实值**：我之前写死成 None，
                # 前端模板拼成 `${size_mb} MB`，界面上就成了
                # 「point_cloud.ply（完整 59 维）  null MB」——
                # 用户会以为文件不存在。这里是显示 bug，不是缺文件。
                n_gauss = None
                if gauss_ply is not None and gauss_ply.exists():
                    nbytes = gauss_ply.stat().st_size
                    size_mb = round(nbytes / 1024 / 1024, 2)
                    # ⚠️ 这里不能读 `man`：manifest 是在**后面**才加载的，
                    # 早先的版本写成 `man.get("gaussians")` ⇒ NameError 被
                    # 外层 try/except 静默吞掉 ⇒ 高斯数永远是 None
                    # （界面只显示 size/count 缺失，看不出是 bug）。
                    # 改成从训练产物 metrics.json 取，语义上也更直接。
                    try:
                        _mp = root / "04_train" / "metrics.json"
                        if _mp.exists():
                            n_gauss = json.loads(
                                _mp.read_text(encoding="utf-8")).get("gaussians")
                    except Exception:  # noqa: BLE001
                        n_gauss = None
                modes["splat"].append({
                    "kind": "gauss3d", "name": "point_cloud.ply（完整 59 维）",
                    "url": gauss_url, "size_mb": size_mb,
                    "bytes": (gauss_ply.stat().st_size
                              if gauss_ply is not None and gauss_ply.exists() else None),
                    "gaussians": n_gauss,
                    "desc": "真实高斯：位置+尺度+旋转+不透明度+SH(48)，"
                            "按视线方向做 SH 着色",
                })
        except Exception:  # noqa: BLE001
            pass

        # ---- 服务端渲染预览（**不依赖 WebGL，保证正确**）----
        #
        # 为什么必须提供：WebGL 泼溅在部分环境下打不开（用户两次遇到黑屏）。
        # 而 `preview_*.png` 是**训练用的同一个 gsplat 渲染器**在服务端出的图，
        # 与 `preview_front.png` 左半完全一致 ⇒ 无论浏览器什么状况，
        # 用户都能在 GUI 里看到模型的真实外观。
        #
        # 这也是验收判据"看到的效果应与预览图一致"的可操作形式：
        # 不是靠 WebGL 复刻，而是直接看权威渲染结果。
        try:
            modes.setdefault("preview", [])
            pv = []
            for name in ("preview_front.png", "preview_iso.png",
                         "preview_side.png", "preview_top.png"):
                f = out / name
                if f.exists():
                    pv.append({
                        "kind": "image", "name": name,
                        "url": f"/api/artifacts/{job_id}/05_export/{name}",
                        "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
                        "bytes": f.stat().st_size,
                        "desc": "gsplat 服务端渲染（与训练同一渲染器，左=渲染 右=真值）",
                    })
            tex = out / "model_mesh_texture.png"
            if tex.exists():
                pv.append({
                    "kind": "image", "name": "model_mesh_texture.png",
                    "url": f"/api/artifacts/{job_id}/05_export/model_mesh_texture.png",
                    "size_mb": round(tex.stat().st_size / 1024 / 1024, 2),
                    "bytes": tex.stat().st_size,
                    "desc": "烘焙贴图图集（2048²，UV 展开后）",
                })
            if pv:
                modes["preview"] = pv
        except Exception:  # noqa: BLE001
            pass

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
            # ---- 高斯泼溅（`.splat`）：**优先显示**，因为它才是模型的真实形态 ----
            #
            # 背景：用户反馈"GUI 查看器里的模型比 preview_front.png 差很多"。
            # 根因是查看器此前只能解析点云/网格 —— 每个高斯被压成 x,y,z,r,g,b，
            # 丢了 3×3 协方差（尺度+旋转）、不透明度、SH ⇒ 必然是稀疏点阵。
            # `.splat` 保留这些属性，查看器用专用泼溅渲染器画朝向各异的椭球 + alpha 合成。
            if low.endswith(".splat"):
                _add("splat", "splat", name,
                     desc or "高斯泼溅 .splat（真实高斯形态：椭球 + alpha 合成）")
            elif low.endswith(".glb") and "mesh" in low and "textured" in low:
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
