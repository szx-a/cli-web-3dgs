"""回归测试：前端资源版本号、自检端点、以及"黑屏必须自带诊断信息"。

## 背景（真实故障）

用户在 GUI 里打开模型：控制台只有 `Failed to load resource: 404`，画面**全黑**。
排查发现服务端一切正常（`/api/gauss/*`、三个脚本、manifest 里 17 条路径全是 200），
真正的原因是**页面是在前端文件更新前打开的**，浏览器缓存了旧脚本：
新版 `app.js` 会请求 `splat` 模式，而旧版 `viewer.js` 不认识 ⇒ 抛错 ⇒ 黑屏。

两类修复，各配一个测试：
1. 首页给脚本引用加**内容版本号**（`?v=<mtime>`），任何改动都会让缓存失效；
2. 前端在"脚本缺失 / 载入失败"时把**原因和 URL**写到界面上，而不是只在控制台报 404。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _client():
    from fastapi.testclient import TestClient

    from gsflow.web.app import create_app

    return TestClient(create_app())


def test_index_script_tags_have_version_token() -> None:
    """三个脚本引用都必须带 `?v=`（否则改完前端看不到变化）。"""
    r = _client().get("/")
    assert r.status_code == 200
    html = r.text
    for name in ("splat.js", "viewer.js", "app.js"):
        pat = re.compile(r'src="/static/' + re.escape(name) + r'\?v=(\d+)"')
        m = pat.search(html)
        assert m, f"{name} 的 script 标签缺少版本号（浏览器会缓存旧版）"
        assert m.group(1) != "0", f"{name} 的版本号退化成了 0（取 mtime 失败）"


def test_index_disables_cache() -> None:
    r = _client().get("/")
    assert "no-cache" in (r.headers.get("cache-control") or "")


def test_selftest_endpoint_reports_static_and_gauss() -> None:
    """`/api/selftest` 必须能回答"前端为什么打不开"。"""
    r = _client().get("/api/selftest")
    assert r.status_code == 200
    d = r.json()
    assert "static" in d and "problems" in d
    for name in ("index.html", "splat.js", "viewer.js", "app.js"):
        assert d["static"][name]["exists"], f"{name} 缺失"
        assert d["static"][name]["bytes"] > 0
    assert d["ok"] is True, f"自检报了问题：{d.get('problems')}"


def test_selftest_flags_missing_job() -> None:
    """负向测试：不存在的作业必须被自检报出来（否则自检没有意义）。"""
    d = _client().get("/api/selftest?job=__nope__").json()
    assert d["ok"] is False
    assert any("作业不存在" in p for p in d["problems"]), d["problems"]


def test_frontend_surfaces_url_on_failure() -> None:
    """载入失败时界面必须显示 URL 与格式，而不是只留一块黑屏。"""
    src = (ROOT / "src" / "gsflow" / "web" / "static" / "app.js").read_text(
        encoding="utf-8")
    assert "地址：" in src, "失败信息应包含请求地址"
    assert "/api/selftest?job=" in src, "失败信息应引导用户查看自检接口"
    assert "前端脚本未加载" in src, "缺少脚本时应有显式提示"
    assert "whiteSpace = 'pre-line'" in src, "多行诊断需要保留换行"


def test_app_all_model_urls_checker_exists() -> None:
    """整体检查工具必须在：它逐个请求 `/api/models` 返回的每一个 URL。"""
    p = ROOT / "scripts" / "check_all_model_urls.py"
    assert p.exists()
    src = p.read_text(encoding="utf-8")
    assert "manifest" in src, "检查器还应校验 manifest 里的每条产物路径"


def test_model_entries_have_real_size() -> None:
    """每个产物条目都必须有**真实**大小，不能是 null。

    回归：训练产物 PLY（`gauss3d` 条目）此前把 `size_mb` 写死成 None，
    前端模板直接拼 `${size_mb} MB` ⇒ 界面显示
    「point_cloud.ply（完整 59 维）  null MB」，用户以为**文件不存在**。
    """
    c = _client()
    jobs = [p.name for p in (ROOT / "workspace").iterdir()
            if p.is_dir() and (p / "05_export").exists()]
    assert jobs, "工作区里没有作业，测试无意义"
    checked = 0
    for job in sorted(jobs)[:12]:
        d = c.get(f"/api/models/{job}").json()
        for mode, files in (d.get("modes") or {}).items():
            for f in files:
                size = f.get("size_mb")
                byt = f.get("bytes")
                ok = ((isinstance(size, (int, float)) and size > 0)
                      or (isinstance(byt, (int, float)) and byt > 0))
                assert ok, (f"作业 {job} 的产物 {f.get('name')} 没有真实大小："
                            f"size_mb={size!r} bytes={byt!r}")
                checked += 1
    assert checked > 0


def test_frontend_never_prints_null_size() -> None:
    """前端不得把 `null`/`undefined` 当大小印出来（必须走 fmtSize 兜底）。"""
    src = (ROOT / "src" / "gsflow" / "web" / "static" / "app.js").read_text(
        encoding="utf-8")
    assert "function fmtSize(" in src, "应有统一的尺寸格式化函数"
    # ⚠️ 必须排除注释：`fmtSize` 的说明里**引用**了那句错误模板，
    # 直接 `in src` 会把注释也当成违规（我第一版就这么误判了）。
    code_lines = []
    in_block = False
    for line in src.splitlines():
        s = line.strip()
        if in_block:
            if "*/" in s:
                in_block = False
            continue
        if s.startswith("/*"):
            in_block = "*/" not in s
            continue
        if s.startswith("//"):
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)
    assert "${f.size_mb}" not in code, "仍在直接拼 size_mb（会印出 null MB）"
    # 兜底必须是"—"而不是空/0
    assert "return '—'" in src
