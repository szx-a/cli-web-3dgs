"""检查配置同步：config.toml 与代码默认值是否一致。

为什么需要：本项目踩过"改了默认值却发现配置里根本没这个键"的坑
（默认值曾有**三处**来源，见 docs/03 G44）。这个脚本把差异直接列出来。

用法: python scripts/check_config_sync.py
"""
from __future__ import annotations

import sys
import tomllib
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from gsflow.config import DEFAULTS
    from gsflow.trainers.gsplat_trainer import TrainConfig
    from gsflow.gsdata.mesh_export import __name__ as _  # noqa: F401

    cfg_path = ROOT / "config" / "config.toml"
    print(f"config.toml: {cfg_path}  存在={cfg_path.exists()}  "
          f"大小={cfg_path.stat().st_size if cfg_path.exists() else 0}")
    if not cfg_path.exists():
        return 1
    with cfg_path.open("rb") as f:
        on_disk = tomllib.load(f)

    bad = 0
    for section, defaults in DEFAULTS.items():
        if not isinstance(defaults, dict):
            continue
        disk = on_disk.get(section, {})
        missing = [k for k in defaults if k not in disk]
        if missing:
            print(f"\n[{section}] 代码有、config.toml **缺失** {len(missing)} 个键：")
            for k in missing:
                print(f"    {k} = {defaults[k]!r}")
            bad += len(missing)
        diff = {k: (defaults[k], disk[k]) for k in defaults
                if k in disk and disk[k] != defaults[k]}
        if diff:
            print(f"\n[{section}] 默认值不一致：")
            for k, (a, b) in diff.items():
                print(f"    {k}: 代码={a!r}  文件={b!r}")

    # TrainConfig dataclass 与 DEFAULTS['train'] 必须一致
    td = {f.name: f.default for f in fields(TrainConfig)}
    train = DEFAULTS["train"]
    mism = {k: (train[k], td[k]) for k in train if k in td and train[k] != td[k]}
    only_dc = [k for k in td if k not in train]
    if mism:
        print(f"\n[train] DEFAULTS 与 TrainConfig 不一致：{mism}")
    if only_dc:
        print(f"\n[train] 只在 TrainConfig、不在 DEFAULTS 的键：{only_dc}")

    print("\n结论: " + ("配置同步一致" if bad == 0 and not mism and not only_dc
                        else f"**发现 {bad} 个缺失键 + 不一致项，需重新生成 config.toml**"))
    return 0 if (bad == 0 and not mism and not only_dc) else 1


if __name__ == "__main__":
    sys.exit(main())
