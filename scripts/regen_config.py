"""重新生成 `config/config.toml`（把新默认值写进文件），然后核对同步。

为什么要单独一个脚本：本项目反复出现"代码里加了配置项，但 config.toml 没更新，
于是界面/文件里看不到、也改不到"（`check_config_sync.py` 就是为此写的）。
把"写文件 + 校验"合成一条命令，避免漏做其中一步。

用法: python scripts/regen_config.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from gsflow import config
    p = config.write_default_config()
    print(f"已写入 {p}")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_config_sync.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=str(ROOT))
    print((r.stdout or "") + (r.stderr or ""))
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
