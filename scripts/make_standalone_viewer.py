"""生成**自包含的离线查看器**：一个 HTML 文件，双击即可看模型，**不需要任何服务**。

为什么要做这个（本项目反复踩的坑）：
DSH 沙箱会回收我启动的服务进程（详见 docs/03 G57），于是用户经常遇到
"查看器打不开 / Edge 报 404"。用户已经反馈多次"又黑了"。
根治办法不是让我把服务保活（做不到），而是提供一个**根本不依赖服务**的产物：
把 `viewer.js` 与模型二进制**内嵌进同一个 HTML**，用 `file://` 打开就能看。

用法：
    python scripts/make_standalone_viewer.py --job img2-2dgs-base
    python scripts/make_standalone_viewer.py --job img2-2dgs-base --kind cloud

输出：workspace/<job>/05_export/viewer_standalone_<kind>.html
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "gsflow" / "web" / "static"

# 独立页面：只保留"看模型"必需的东西（画布 + 三个按钮 + 状态行）
PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>离线查看器 · {title}</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #0b0e14; color: #dfe6f3;
                font: 13px/1.5 system-ui, "Segoe UI", "Microsoft YaHei", sans-serif; }}
  header {{ padding: 10px 14px; border-bottom: 1px solid #1e2738; display: flex;
            gap: 10px; align-items: center; flex-wrap: wrap; }}
  header b {{ color: #7fd1ff; font-weight: 600; }}
  button {{ background: #1a2233; color: #dfe6f3; border: 1px solid #2c3a52;
            border-radius: 6px; padding: 5px 11px; cursor: pointer; font: inherit; }}
  button:hover {{ background: #223049; }}
  #stage {{ position: absolute; inset: 46px 0 0 0; }}
  canvas {{ width: 100%; height: 100%; display: block; }}
  #status {{ position: absolute; left: 12px; bottom: 10px; color: #8fa3c0;
             background: rgba(11,14,20,.82); padding: 4px 9px; border-radius: 6px; }}
  #overlay {{ position: absolute; inset: 46px 0 0 0; display: flex; align-items: center;
              justify-content: center; color: #7f8ea6; pointer-events: none; }}
  .hint {{ color: #8fa3c0; margin-left: auto; }}
</style>
</head>
<body>
<header>
  <b>{title}</b>
  <button id="btn-reset">重置视角</button>
  <button id="btn-wire">切换线框</button>
  <button id="btn-mode">旋转模式：到顶底停住</button>
  <span class="hint">左键拖拽旋转 · 右键/Shift 平移 · 滚轮缩放</span>
</header>
<div id="stage"><canvas id="c"></canvas></div>
<div id="overlay">载入中…</div>
<div id="status"></div>
<script>
{viewer_js}

/* ---- 模型数据（base64 内嵌，避免 file:// 下的 fetch/CORS 限制）---- */
const MODEL_B64 = "{model_b64}";
const MODEL_KIND = "{kind}";

function b64ToArrayBuffer(b64) {{
  const bin = atob(b64);
  const len = bin.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}}

(async function () {{
  const canvas = document.getElementById('c');
  const overlay = document.getElementById('overlay');
  const status = document.getElementById('status');
  try {{
    window.GsViewer.init(canvas, (msg) => {{ status.textContent = msg || ''; }});
    const info = await window.GsViewer.loadBuffer(
      b64ToArrayBuffer(MODEL_B64), '{title}.{kind}', MODEL_KIND);
    overlay.style.display = 'none';
    status.textContent = `顶点 ${{info.vertices.toLocaleString()}}` +
      (info.triangles ? ` · 三角面 ${{info.triangles.toLocaleString()}}` : ' · 点云') +
      (info.normals ? ' · 有法线' : '') + (info.colors ? ' · 有顶点色' : '');
    document.getElementById('btn-reset').onclick = () => window.GsViewer.reset();
    document.getElementById('btn-wire').onclick = (e) => {{
      const on = window.GsViewer.toggleWire();
      e.target.textContent = on ? '关闭线框' : '切换线框';
    }};
    const bm = document.getElementById('btn-mode');
    bm.onclick = () => {{
      const cur = window.GsViewer.getRotMode ? window.GsViewer.getRotMode() : 'clamp';
      const next = cur === 'free' ? 'clamp' : 'free';
      window.GsViewer.setRotMode(next);
      bm.textContent = next === 'free' ? '旋转模式：自由翻转' : '旋转模式：到顶底停住';
    }};
  }} catch (err) {{
    overlay.textContent = '载入失败：' + (err && err.message ? err.message : err);
    overlay.style.color = '#ff8a8a';
    console.error(err);
  }}
}})();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--kind", default="mesh", choices=["mesh", "cloud", "textured"],
                    help="mesh=网格 glb；cloud=点云 glb；textured=带贴图网格")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    job_dir = ROOT / "workspace" / args.job
    export = job_dir / "05_export"
    names = {
        # 点云 GLB 在不同时期的导出里叫过两个名字，两个都认
        "mesh": ("model_mesh.glb", "glb"),
        "cloud": ("model.glb", "glb"),
        "textured": ("model_mesh_textured.glb", "glb"),
    }
    fname, kind = names[args.kind]
    src = export / fname
    if not src.exists() and args.kind == "cloud":
        alt = export / "model_points.glb"
        if alt.exists():
            src = alt
    if not src.exists():
        # 点云 GLB 可能叫别的名字，退一步扫描
        cands = sorted(export.glob("*.glb"))
        print(f"找不到 {src}")
        if cands:
            print("可用的 glb：")
            for c in cands:
                print(f"  {c.name}  {c.stat().st_size / 1048576:.2f} MB")
        return 1

    raw = src.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    viewer_js = (STATIC / "viewer.js").read_text(encoding="utf-8")
    title = f"{args.job} · {args.kind}"
    html = PAGE.format(title=title, viewer_js=viewer_js, model_b64=b64, kind=kind)

    out = Path(args.out) if args.out else (export / f"viewer_standalone_{args.kind}.html")
    out.write_text(html, encoding="utf-8")
    print(f"已生成 {out}")
    print(f"  模型 {src.name}: {len(raw) / 1048576:.2f} MB")
    print(f"  HTML: {out.stat().st_size / 1048576:.2f} MB（含内嵌数据）")
    print(f"  ⇢ 直接在浏览器里打开这个 HTML 即可，**不需要启动任何服务**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
