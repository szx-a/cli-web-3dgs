"""负向测试：验证 `check_server_code_sync.py` **真的能抓到"服务跑旧代码"**。

为什么需要：一个只在"服务是新的"时通过的检查器毫无价值。
本脚本的第一版判据是 `st is None or st >= 500`，**把所有 404 都放过了** ——
于是"服务还在跑旧代码、新路由 404"这种**最该被抓到**的情况反而报"全部可用"。
我是在直接探测发现 404 之后才注意到这个矛盾的。

做法：起一个**故意不注册新路由**的最小 FastAPI 应用（模拟旧服务），
让检查器指向它，断言它**报失败**；再起一个注册了新路由的（模拟新服务），
断言它**报成功**。

用法: python scripts/selftest_server_sync_checker.py
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_up(port: int, timeout: float = 25.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2):
                return True
        except Exception:  # noqa: BLE001
            time.sleep(0.4)
    return False


def run_checker(port: int) -> int:
    p = subprocess.run([PY, str(ROOT / "scripts" / "check_server_code_sync.py"),
                        "--url", f"http://127.0.0.1:{port}"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    return p.returncode


STUB = """
import sys
sys.path.insert(0, {src!r})
import uvicorn
from fastapi import FastAPI
app = FastAPI()

@app.get("/api/health")
def health():
    return {{"ok": True}}

{extra}uvicorn.run(app, host="127.0.0.1", port={port}, log_level="error")
"""

NEW_ROUTE = '''
@app.get("/api/opacity_cleanup")
def oc():
    return {"choices": [{"value": "none", "params": {"train.opacity_cleanup": "none"}}]}

@app.get("/api/stages")
def stages():
    return []

@app.get("/api/config")
def cfg():
    return {"config": {}}

@app.get("/api/jobs")
def jobs():
    return {"running": [], "history": []}

@app.get("/api/credentials")
def creds():
    return {}

@app.get("/api/version")
def ver():
    return {}

@app.get("/api/models")
def models():
    return []

@app.get("/api/meshes")
def meshes():
    return []

# 只有 POST 的路由：用 GET 探会得到 **405**，而 405 表示"路由存在、方法不允许"。
# 检查器必须把它算作**存在**（第一版把 405 当缺失，误报 2 条）。
@app.post("/api/run")
def run():
    return {}

@app.post("/api/config/reset")
def reset():
    return {}
'''


def start_stub(port: int, new: bool) -> subprocess.Popen:
    code = STUB.format(src=str(ROOT / "src"), port=port, extra=NEW_ROUTE if new else "")
    f = ROOT / "env" / "tmp" / f"stub_server_{port}.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(code, encoding="utf-8")
    return subprocess.Popen([PY, str(f)], cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    bad = 0

    # ---- 用例 1：旧服务（缺新路由）⇒ 检查器必须报失败 ----
    port = free_port()
    proc = start_stub(port, new=False)
    try:
        if not wait_up(port):
            print(f"  **FAIL** 桩服务未能启动（端口 {port}）")
            return 2
        rc = run_checker(port)
        ok = rc != 0
        print(f"  [{'OK' if ok else '**FAIL**'}] 旧服务（缺 /api/opacity_cleanup）"
              f" ⇒ 检查器退出码 {rc}（应非 0）")
        bad += 0 if ok else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()

    # ---- 用例 2：新服务（路由齐全）⇒ 检查器必须通过 ----
    # 说明：桩服务的路由集合与真实 app 不完全一致，
    # 所以这里只验证"检查器在**没有 404** 时不会误报"。
    port2 = free_port()
    proc2 = start_stub(port2, new=True)
    try:
        if not wait_up(port2):
            print(f"  **FAIL** 桩服务 2 未能启动（端口 {port2}）")
            return 2
        rc = run_checker(port2)
        # 桩已补齐真实 app 的全部 GET 路由，并包含两个**仅 POST** 的路由
        # （GET 探它们会得到 405 —— 必须被算作"存在"，这正是本用例要锁住的行为）。
        # 因此这里期望退出码为 0；若为 1，说明要么路由没补齐，
        # 要么检查器又把 405 当成了缺失。
        ok2 = rc == 0
        print(f"  [{'OK' if ok2 else '**FAIL**'}] 新服务桩（含仅 POST 路由，GET 得 405）"
              f" ⇒ 检查器退出码 {rc}（应为 0，即不把 405 当缺失）")
        bad += 0 if ok2 else 1
    finally:
        proc2.terminate()
        try:
            proc2.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc2.kill()

    verdict = "检查器能抓到服务跑旧代码" if bad == 0 else f"{bad} 项失败"
    print(f"\n结论: {verdict}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
