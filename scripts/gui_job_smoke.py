"""通过 GUI 的 HTTP API 提交一次真实作业并观察事件流。

用途：证明「GUI 与 CLI 共用同一流程核心」——同一个 Orchestrator、同一套阶段、
同一份 job.json 与事件总线，而不是两套逻辑。

前置：先启动 GUI（scripts\\web.cmd 或 python -m gsflow.cli web）
用法：env\\venv\\Scripts\\python.exe scripts\\gui_job_smoke.py --port 8791
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
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--images", default=str(ROOT / "env" / "synthetic" / "images"))
    ap.add_argument("--stages", default="frames,mask,colmap")
    ap.add_argument("--backend", default="classical")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    # 健康检查
    h = get(base, "/api/health")
    print(f"GUI 健康检查: {h}")

    st, body = 0, None
    body = {
        "images_dir": args.images,
        "input_path": args.images,
        "title": "GUI 提交的全链路作业",
        "stages": [s for s in args.stages.split(",") if s],
        # 同时带上"点路径"形式的参数（GUI 的 _flatten_params 就是产出这种），
        # 以及嵌套形式，验证后端两种都吃得住、且 CLI 侧能同构复现。
        "params": {
            "mask": {"backend": args.backend, "dilate_px": 2},
            "train.densify_until": 4000,
            "train.mask_loss_until": 1.0,
            "export.formats": "glb_points,mesh,viewer_html",
            "export.mesh_method": "poisson",
            "export.mesh_poisson_depth": 9,
            "export.mesh_target_triangles": 400000,
        },
    }
    res = post(base, "/api/run", body)
    run_id, job_id = res["run_id"], res.get("job_id", "")
    print(f"提交成功 run_id={run_id} job_id={job_id}")
    print("GUI 显示的等价 CLI 命令：")
    print("   ", res["cli_equivalent"])
    print("-" * 72)

    since = 0
    started = time.time()
    final = None
    while time.time() - started < args.timeout:
        d = get(base, f"/api/run/{run_id}/events?since={since}")
        for e in d.get("events", []):
            since = max(since, e.get("seq", 0))
            lvl = e.get("level", "info")
            stage = e.get("stage", "")
            if lvl in ("warn", "error", "success") or stage == "orchestrator":
                print(f"  [{stage:<12}] {lvl:<7} {e.get('message', '')[:100]}")
        js = get(base, "/api/jobs")
        final = next((r["status"] for r in js["running"] if r["run_id"] == run_id), None)
        if final in ("done", "failed", "cancelled"):
            break
        time.sleep(1.0)

    print("-" * 72)
    print(f"最终状态: {final}（耗时 {time.time()-started:.1f}s）")
    if job_id:
        hist = get(base, "/api/jobs")
        j = next((x for x in hist["history"] if x.get("job_id") == job_id), None)
        if j:
            print("作业目录:", j.get("job_dir"))
            for name, info in (j.get("stages") or {}).items():
                print(f"  {name:<8} {info.get('status', '?'):<10} {info.get('message', '')[:70]}")
    return 0 if final == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
