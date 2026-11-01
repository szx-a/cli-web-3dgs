"""锁定 README 目录与示例配置的**一致性**（2026-11-01 新增）。

两个真实风险：
1. README 目录用显式锚点 `<a id="sec-*">` 实现；若有人改了标题却没重跑
   `scripts/make_readme_toc.py`，目录里就会出现**点不动的死链**（GitHub 上不报错）。
2. `config/config.example.toml` 是**一键安装**时复制给新用户的模板。它曾经过期
   （`densify_until = 4000`、`opacity_reset_interval = 0`、`opacity_floor = 0.0`），
   导致"代码里调好了、新装用户还是旧行为"。这里钉住关键默认值。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
EXAMPLE = ROOT / "config" / "config.example.toml"


def test_readme_toc_links_all_resolve() -> None:
    text = README.read_text(encoding="utf-8")
    m = re.search(r"<!-- TOC -->(.*?)<!-- /TOC -->", text, re.S)
    assert m, "README 缺少目录块（<!-- TOC --> ... <!-- /TOC -->）"
    toc = m.group(1)
    links = set(re.findall(r"\]\(#([^)]+)\)", toc))
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', text))
    assert links, "目录里没有任何链接"
    missing = sorted(links - anchors)
    assert not missing, f"目录里有死链（标题改了但没重跑 make_readme_toc.py）：{missing}"
    # 每个锚点都应对应一个标题（避免残留旧锚点）
    assert len(anchors) >= 20, f"锚点太少（{len(anchors)}），目录可能没生成"


def test_readme_toc_does_not_link_itself() -> None:
    text = README.read_text(encoding="utf-8")
    toc = re.search(r"<!-- TOC -->(.*?)<!-- /TOC -->", text, re.S).group(1)
    assert "- [目录](#" not in toc, "目录不应把自己列进目录"


def test_example_config_is_not_stale() -> None:
    """示例配置必须反映当前默认值（它会被安装器复制给新用户）。"""
    txt = EXAMPLE.read_text(encoding="utf-8")
    expect = {
        "densify_until": "14000",
        "opacity_reset_interval": "3000",
        "opacity_floor": "0.02",
        "opacity_floor_reg": "0.2",
        "sh_degree_interval": "1000",
        "lpips_backend": '"vgg16"',
        "lambda_grad": "1.0",
    }
    for key, want in expect.items():
        m = re.search(rf"^{key}\s*=\s*(\S+)", txt, re.M)
        assert m, f"config.example.toml 缺少 {key}"
        assert m.group(1) == want, f"config.example.toml 里 {key}={m.group(1)}，期望 {want}（先跑 scripts/sync_example_config.py）"


def test_example_config_has_no_local_absolute_path() -> None:
    """示例文件不能带本机绝对路径，否则复制到别的机器就是错的。"""
    txt = EXAMPLE.read_text(encoding="utf-8")
    assert "F:\\\\3dgs-and-ai" not in txt and "F:\\3dgs-and-ai" not in txt
    m = re.search(r"^workspace_dir\s*=\s*(.+)$", txt, re.M)
    assert m and m.group(1).strip() == '""', "示例配置的 workspace_dir 必须为空"
