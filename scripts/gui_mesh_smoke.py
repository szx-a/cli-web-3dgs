"""通过 Web GUI 的 HTTP API 提交一个**含网格 + 纹理烘焙**的完整作业并等它跑完。

用途：验证「GUI 表单 → 参数 → 流程核心 → 一键出网格与贴图」这条链路端到端可用，
以及 GUI 与 CLI 共用同一 Orchestrator 的架构红线。

用法：
  scripts\\gpu_python.cmd scripts\\gui_mesh_smoke.py --port 8787 --job <已有训练的作业>
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def post(base: str, path: str, body: dict):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def get(base: str, path: str, timeout: int = 300):
    """默认超时给得比较长：网格导出会占住服务端（CPU 密集），
    期间事件请求要排队，短超时会直接 TimeoutError。"""
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--job", required=True, help="已有训练的作业 id（复用其 04_train）")
    ap.add_argument("--images", default=str(ROOT / "workspace" / "real-final" / "02_images_clean"))
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    print("健康检查:", get(base, "/api/health"))

    job_dir = ROOT / "workspace" / args.job
    ply = job_dir / "04_train" / "point_cloud"
    body = {
        "input_path": args.images,
        "images_dir": "",
        "job_id": args.job,
        "title": "GUI 纹理烘焙验证",
        "stages": ["export"],
        "force_stages": ["export"],
        "params": {
            "export.ply": "",
            "export.formats": "glb_points,mesh,viewer_html",
            "export.mesh_method": "poisson",
            "export.mesh_poisson_depth": 9,
            "export.mesh_target_triangles": 80000,
            "export.mesh_width": 512,
            "export.mesh_max_views": 20,
            "export.mesh_smooth_iters": 5,
            "export.mesh_texture": "true",
            "export.mesh_texture_size": 1024,
        },
        "halt_on_error": True,
    }
    print(f"复用点云目录: {ply}")
    res = post(base, "/api/run", body)
    rid = res.get("run_id")
    job_id = res.get("job_id")
    print(f"提交成功 run_id={rid} job_id={job_id}")
    cli = res.get("cli") or res.get("command")
    if cli:
        print("等价 CLI 命令:")
        print("   ", cli)

    # 注意：没有 `/api/run/{id}` 这个端点（我一开始写错，得到 404）。
    # 状态要从 `/api/jobs/{job_id}` 读，事件流用 `/api/run/{id}/events`。
    since = 0
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        d = get(base, f"/api/run/{rid}/events?since={since}")
        for ev in d.get("events", []):
            since = max(since, int(ev.get("seq", 0)) + 1)
            msg = ev.get("message", "")
            lvl = ev.get("level", "info")
            if lvl in ("success", "warn", "error") or "网格" in msg or "贴图" in msg:
                print(f"  [{lvl:>7}] {msg}")
        j = get(base, f"/api/jobs/{job_id}")
        st = (j.get("job") or j).get("status")
        if st in ("done", "failed", "cancelled"):
            print(f"\n最终状态: {st}（耗时 {time.time()-t0:.1f}s）")
            break
        time.sleep(3)

    out = job_dir / "05_export"
    print(f"\n产物目录: {out}")
    if out.exists():
        for p in sorted(out.iterdir()):
            if p.is_file():
                print(f"  {p.name:32s} {p.stat().st_size/1024/1024:8.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
