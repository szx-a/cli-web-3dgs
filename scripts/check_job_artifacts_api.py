"""验证 `/api/jobs/{id}` 是否已把 `model_gauss.glb` 等产物补进列表。

用 TestClient（需要 httpx；venv 里没有，所以用 gpu_python 跑）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"F:\3dgs-and-ai")
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from gsflow.paths import apply_runtime_env  # noqa: E402
from gsflow.web.app import create_app  # noqa: E402

apply_runtime_env()

client = TestClient(create_app())
r = client.get("/api/jobs/img5-off")
assert r.status_code == 200, r.text[:200]
d = r.json()
arts = []
for st, v in (d.get("stages") or {}).items():
    for a in (v.get("artifacts") or []):
        arts.append((st, a.get("rel") if isinstance(a, dict) else str(a)))
print(f"  产物条目数 = {len(arts)}")
hit = [a for a in arts if a[1] and "model_gauss.glb" in a[1]]
print("  model_gauss.glb 是否出现:", bool(hit), hit[:1])
for st, rel in arts[:16]:
    print(f"   [{st}] {rel}")
assert hit, "GUI 仍拿不到 model_gauss.glb"
print("  通过：接口已返回 model_gauss.glb")
