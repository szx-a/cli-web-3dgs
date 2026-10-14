"""确认服务端交付的网格 PLY 内容正确（HTTP 层），供"PLY 查看是否实体"这一验收用。

为什么单独做：用户报告"PLY 查看只有点云"。修好查看器后，还要确认
**服务端交付的字节**确实包含 `element face`（否则解析器再对也没有面可读）。

用法：env\\venv\\Scripts\\python.exe scripts\\check_ply_delivery.py --job img2-2dgs-base
"""

from __future__ import annotations

import argparse
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="img2-2dgs-base")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    cases = [
        ("model_mesh.ply", True, "网格 PLY（应有 element face）"),
        ("model_mesh.glb", False, "网格 GLB"),
    ]
    ok = True
    for name, want_face, label in cases:
        url = f"{base}/api/artifacts/{args.job}/05_export/{name}"
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                body = r.read()
                status = r.status
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
            ok = False
            continue
        magic = body[:4]
        head = body[:5000]
        has_face = b"element face" in head
        vline = next((ln for ln in head.split(b"\n") if ln.startswith(b"element vertex")), b"")
        fline = next((ln for ln in head.split(b"\n") if ln.startswith(b"element face")), b"")
        good = status == 200 and len(body) > 1000
        if name.endswith(".ply"):
            good = good and (has_face == want_face)
        print(f"  [{'OK' if good else 'FAIL'}] {label}: HTTP {status}  {len(body):,} 字节  "
              f"magic={magic}")
        if name.endswith(".ply"):
            print(f"        {vline.decode(errors='replace').strip()}")
            print(f"        {fline.decode(errors='replace').strip()}")
        ok &= good

    # 点云 PLY（训练检查点）应**没有** element face
    url = f"{base}/api/artifacts/{args.job}/05_export/model_mesh.obj"
    print("\n结论:", "服务端交付的网格 PLY 含面数据，查看器可解析为实体" if ok
          else "**交付内容异常**")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
