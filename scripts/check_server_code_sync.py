"""检查**运行中的 GUI 服务**是否跑的是当前代码 —— 而不只是"端口有响应"。

为什么需要（真实踩坑）：我加了 `/api/opacity_cleanup` 接口后，
`/static/app.js` 与 `index.html` 已经是新版（静态文件每次请求都从磁盘读），
但 `/api/opacity_cleanup` 却 404 —— 因为 **uvicorn 不会热重载 `app.py`**，
服务进程仍在跑旧代码。这种"半新半旧"状态最容易误判：
前端会去调一个不存在的接口，而我可能把它当成"功能没做对"去debug前端。

判据（三层，逐层加严）：
  1. 端口有响应（`/api/health` 200）—— 只能说明"服务活着"；
  2. 服务暴露的**路由集合**包含代码里定义的路由 —— 说明跑的是新代码；
  3. 静态资源含最新标记 —— 说明前端也是新的。
只有 2、3 都通过，才算"服务已同步到当前代码"。

用法: python scripts/check_server_code_sync.py [--url http://127.0.0.1:8787]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def get(url: str, timeout: float = 5.0) -> tuple[int | None, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:      # 有响应但是错误码
        return e.code, b""
    except Exception:                        # noqa: BLE001
        return None, b""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8787")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    print(f"目标服务：{base}")

    # ---- 1) 活着吗 ----
    st, body = get(base + "/api/health")
    if st != 200:
        print(f"  [FAIL] /api/health -> {st}（服务没在跑？先执行 scripts\\web.cmd）")
        return 2
    try:
        info = json.loads(body)
    except Exception:  # noqa: BLE001
        info = {}
    print(f"  [OK]   /api/health -> 200  workspace={info.get('workspace')}")

    # ---- 2) 路由集合是否含代码里定义的全部路由 ----
    from fastapi.testclient import TestClient

    from gsflow.web.app import create_app

    app = create_app()
    # ⚠️ 排除**慢接口**：`/api/doctor` 会实际探测 CUDA/COLMAP/FFmpeg，
    # 单次可能要几十秒。第一版把它也纳入检查，结果超时被当成"路由缺失"，
    # 报出"服务缺 1 条路由"的**假警报**（而真正缺的 opacity_cleanup 反而没报）。
    # 检查器必须只依赖"确定的、快速的"证据。
    SLOW = {"/api/doctor"}
    code_routes = set()
    for r in app.routes:
        p = getattr(r, "path", "")
        if p.startswith("/api/") and "{" not in p and p not in SLOW:
            code_routes.add(p)
    if SLOW:
        print(f"  （跳过慢接口：{', '.join(sorted(SLOW))}）")

    missing = []
    for p in sorted(code_routes):
        st2, _ = get(base + p, timeout=8.0)
        # ⚠️ 判据要分清三种情况（两次踩坑换来的）：
        #   * **404** ⇒ 路由**不存在** ⇒ 服务跑的是旧代码（要抓的就是这个）；
        #   * **405** ⇒ 路由存在但方法不允许（例如 `/api/run` 只接受 POST）
        #             ⇒ **证明路由存在**，不能算缺失（第一版把 405 当缺失，误报 2 条）；
        #   * **5xx / 无响应** ⇒ 服务有故障，也要报。
        # 第一版的判据是 `st is None or st >= 500`，把 404 也放过了，
        # 于是真正缺的 `/api/opacity_cleanup` 没被抓到，检查器反而说"全部可用"。
        if st2 is None or st2 == 404 or st2 >= 500:
            missing.append((p, st2))
    if missing:
        print(f"  [FAIL] 服务缺 {len(missing)} 条代码里已有的路由"
              f" ⇒ **服务进程跑的是旧代码，需要重启**：")
        for p, s in missing:
            print(f"           {p} -> {s}")
        print("         重启方法：Ctrl+C 停掉旧服务，再运行 scripts\\web.cmd")
        code_stale = True
    else:
        print(f"  [OK]   代码里定义的 {len(code_routes)} 条 /api 路由服务端全部可用")
        code_stale = False

    # ---- 3) 静态资源是否含最新标记 ----
    marks = {
        "/static/index.html": "p-train-opacityclean",
        "/static/app.js": "loadOpacityMap",
    }
    for path, token in marks.items():
        st3, b = get(base + path, timeout=8.0)
        has = token.encode() in b
        flag = "OK  " if (st3 == 200 and has) else "FAIL"
        print(f"  [{flag}] {path} -> {st3}  含最新标记 {token!r}: {has}")

    if code_stale:
        print("\n结论: **服务跑的是旧代码**（静态资源可能是新的，但后端不是）")
        print("      ⇒ 必须重启 Web 服务，否则界面里的新控件会去调不存在的接口。")
        return 1
    print("\n结论: 服务已同步到当前代码")
    return 0


if __name__ == "__main__":
    sys.exit(main())
