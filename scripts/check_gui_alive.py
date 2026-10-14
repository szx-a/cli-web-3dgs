"""检查运行中的 GUI 服务是否**健康**，以及它加载的是**新版还是旧版**代码。

为什么需要：这个项目反复出现"改了前端/后端但用户看到的是旧行为"。
静态资源有 `no-cache` 能保证浏览器拿新的，但**服务进程本身**如果没重启，
`app.py` 的改动就不会生效（uvicorn 不会热加载）——本轮就踩了这个，
服务还在返回绝对路径产物，而磁盘上的 app.py 已经修好了。

用法: python scripts/check_gui_alive.py [--port 8787]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

# (标记, 检查函数)  —— 检查"新版特征是否已生效"
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    bad = 0

    def get(path: str, timeout: int = 20) -> bytes:
        return urllib.request.urlopen(base + path, timeout=timeout).read()

    print(f"服务 {base}")
    try:
        h = json.loads(get("/api/health"))
        print(f"  health: {h}")
    except Exception as e:  # noqa: BLE001
        print(f"  **服务不可用**: {type(e).__name__}: {e}")
        print("  -> 请在终端运行 F:\\3dgs-and-ai\\scripts\\web.cmd")
        return 1

    # 静态资源里的新版标记
    marks = {
        "app.js": ["is_dir", "setRotMode", "不要再自己拼 URL"],
        "viewer.js": ["setRotMode", "OES_element_index_uint", "buildCam", "getExtension"],
        "index.html": ["btn-m-rotmode"],
    }
    print("\n静态资源（浏览器端）:")
    for name, keys in marks.items():
        try:
            txt = get("/static/" + name).decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            print(f"  **{name} 取不到**: {e}")
            bad += 1
            continue
        miss = [k for k in keys if k not in txt]
        print(f"  {name:12s} {len(txt):>7} 字节  " +
              ("最新" if not miss else f"**缺少标记 {miss}**"))
        if miss:
            bad += 1

    # 服务端 app.py 的新版行为：产物必须是归一化对象
    print("\n服务端逻辑（需要进程已重启）:")
    try:
        jobs = json.loads(get("/api/jobs"))
        items = list(jobs.get("history") or []) + list(jobs.get("running") or [])
        if not items:
            print("  （没有作业可检查）")
        else:
            jid = items[0]["job_id"] if isinstance(items[0], dict) else str(items[0])
            d = json.loads(get(f"/api/jobs/{jid}"))
            found = None
            for stage, v in (d.get("stages") or {}).items():
                a = (v or {}).get("artifacts") or []
                if a:
                    found = (stage, a[0])
                    break
            if found is None:
                print(f"  {jid}: 没有产物可判断")
            else:
                stage, a = found
                if isinstance(a, dict) and a.get("url"):
                    print(f"  产物已归一化（新 app.py 已生效）: {a.get('rel')}  is_dir={a.get('is_dir')}")
                else:
                    print(f"  **产物仍是裸路径 -> 运行的是旧 app.py，必须重启服务**")
                    print(f"     {jid}/{stage}: {a!r}")
                    bad += 1
    except Exception as e:  # noqa: BLE001
        print(f"  检查失败: {type(e).__name__}: {e}")
        bad += 1

    print("\n结论: " + ("服务健康且已是新版" if bad == 0
                        else f"**{bad} 项需要处理（多半是没重启服务）**"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
