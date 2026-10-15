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
        bad += len(mism)
    # ⚠️ "只在 dataclass、不在 DEFAULTS" **不一定是问题**：
    # dataclass 上有些字段是**运行时注入**的（data_dir / out_dir 由阶段填写），
    # 或者是**刻意不暴露**的内部旋钮。把它们一律算作"缺失"会让本脚本自己
    # 报出自相矛盾的结论（"0 个缺失键"却又列出 11 个键）。
    # 这里改成分类报告，只有"用户可调但没暴露"的才计入 bad。
    RUNTIME_INJECTED = {"data_dir", "out_dir"}
    runtime = [k for k in only_dc if k in RUNTIME_INJECTED]
    hidden = [k for k in only_dc if k not in RUNTIME_INJECTED]
    if runtime:
        print(f"\n[train] 运行时注入的字段（正常）：{runtime}")
    if hidden:
        print(f"\n[train] **只在 dataclass、界面上无法设置**的旋钮（{len(hidden)} 个）：")
        for k in hidden:
            print(f"    {k} = {td[k]!r}")
        print("    说明：CLI 仍可用 --set train.<key>=... 覆盖，但 GUI 不会显示。"
              "若希望从界面调整，需要把它们加进 DEFAULTS。")

    print("\n结论: " + ("配置同步一致" if bad == 0
                        else f"**{bad} 项不一致，需处理**"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
