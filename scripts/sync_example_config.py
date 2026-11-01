"""从当前 `config/config.toml` 重新生成 `config/config.example.toml`。

为什么必须做（2026-11-01 审计）：`config.example.toml` 已经过期 ——
`densify_until = 4000`、`absgrad = false`、`opacity_reset_interval = 0`、
`opacity_floor = 0.0`。一键安装器会把 `config/` 整个复制到目标目录，
于是**全新安装的用户拿到的仍是旧默认值**，与我们刚刚调优（重置 3000、窗口 14000、
floor 0.02、SH 递增 1000）完全不一致。

做法：保留示例文件头的两句说明，正文用当前 config.toml 内容（把本机绝对路径清空）。
"""
from __future__ import annotations

import re
from pathlib import Path

CFG = Path(r"F:\3dgs-and-ai\config\config.toml")
EX = Path(r"F:\3dgs-and-ai\config\config.example.toml")

body = CFG.read_text(encoding="utf-8")
# 清掉本机绝对路径（示例必须可移植）
body = re.sub(r'workspace_dir\s*=\s*"[^"]*"', 'workspace_dir = ""', body)

HEADER = """# gsflow 示例配置：复制为 config.toml 后按本机情况修改。
# 注意：config.toml 不入库（它含本机绝对路径）。
# 本文件由 config.toml 同步生成（scripts/sync_example_config.py），
# 所以训练默认值（不透明度重置 3000、densify_until 14000、opacity_floor 0.02、
# SH 递增 1000、lpips_backend 等）与代码保持一致。

"""

EX.write_text(HEADER + body, encoding="utf-8")
print("  已生成 config.example.toml：", len(EX.read_text(encoding='utf-8').splitlines()), "行")

# 校验关键默认值
txt = EX.read_text(encoding="utf-8")
for key, want in (("densify_until", "14000"), ("opacity_reset_interval", "3000"),
                  ("opacity_floor", "0.02"), ("sh_degree_interval", "1000"),
                  ("lpips_backend", '"vgg16"')):
    m = re.search(rf"^{key}\s*=\s*(\S+)", txt, re.M)
    got = m.group(1) if m else "缺失"
    flag = "OK" if got == want else "**不符**"
    print(f"    {key:24s} = {got:10s} 期望 {want:8s} {flag}")
