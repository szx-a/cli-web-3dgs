"""创建桌面 / 开始菜单快捷方式（不需要管理员权限）。

生成的快捷方式：
  1. 桌面:      gsflow Web GUI.lnk      -> scripts\\web.cmd（启动图形界面并打开浏览器）
  2. 桌面:      gsflow CLI.lnk          -> cmd.exe 里进入项目根并预置 GPU 环境
  3. 开始菜单:  gsflow 三维重建\\ 下放上面两个 + 环境自检 + 便携 CUDA 构建

用法：
    env\\venv\\Scripts\\python.exe scripts\\make_shortcuts.py
    env\\venv\\Scripts\\python.exe scripts\\make_shortcuts.py --remove
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_CMD = ROOT / "scripts" / "web.cmd"
GSFLOW_GPU = ROOT / "scripts" / "gsflow_gpu.cmd"
GSFLOW = ROOT / "scripts" / "gsflow.cmd"
BUILD_CUDA = ROOT / "scripts" / "build_portable_cuda.cmd"
ICON = ROOT / "docs" / "gsflow.ico"


def _desktop() -> Path:
    import os

    for env in ("USERPROFILE", "HOME"):
        base = os.environ.get(env)
        if base:
            cand = Path(base) / "Desktop"
            if cand.exists():
                return cand
    return Path.home() / "Desktop"


def _start_menu() -> Path:
    """开始菜单分组目录。

    刻意用**纯 ASCII 且无空格**的名字：带空格/中文的目录名在多层
    powershell -Command 调用里很容易因为引号嵌套而建不出来。
    """
    import os

    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return base / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "gsflow"


def _make_lnk(path: Path, target: Path, args: str, workdir: Path, desc: str) -> Path:
    """用 WScript.Shell 创建 .lnk（纯 COM，不需要管理员）。

    注意：开始菜单目录在沙箱授权范围之外时，mkdir 会 WinError 5；
    这种情况只警告，不让整个脚本失败（桌面部分通常仍能创建）。
    """
    import subprocess

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"  跳过（目录不可写）: {path.parent}  -> {exc}")
        return path
    ps = f"""
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut('{path}')
$sc.TargetPath = '{target}'
$sc.Arguments = '{args}'
$sc.WorkingDirectory = '{workdir}'
$sc.Description = '{desc}'
$sc.WindowStyle = 1
{'$sc.IconLocation = "' + str(ICON) + '"' if ICON.exists() else ''}
$sc.Save()
Write-Output 'created: {path}'
"""
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, encoding="utf-8", errors="ignore")
    # 必须自己验证产物：WScript.Shell 保存失败时 PowerShell 仍可能返回 0，
    # 而且脚本里的 Test-Path 也在同一个（会被权限拦住的）进程里，不可信。
    if path.exists():
        print(f"  已创建: {path}")
        return path
    why = (r.stderr or r.stdout or "").strip().splitlines()
    hint = next((l for l in why if "无法保存" in l or "Unauthorized" in l or "拒绝" in l), "")
    print(f"  失败（未生成）: {path}")
    if hint:
        print(f"           {hint.strip()[:160]}")
    print("           提示：桌面/开始菜单在沙箱授权之外，需要提权或手动运行脚本。")
    return path


def create() -> int:
    desktop, sm = _desktop(), _start_menu()
    print(f"桌面目录  : {desktop}")
    print(f"开始菜单  : {sm}")
    print("创建快捷方式：")

    # 1) Web GUI：直接跑 web.cmd（它会 chcp 并打开浏览器）
    _make_lnk(desktop / "gsflow Web GUI.lnk", WEB_CMD, "", ROOT,
              "启动 gsflow 三维重建图形界面（127.0.0.1:8787）")
    # 2) CLI：开一个带 MSVC + 便携 CUDA 环境的命令行
    _make_lnk(desktop / "gsflow 命令行.lnk", Path("cmd.exe"),
              f'/k "chcp 65001 >nul && title gsflow && cd /d {ROOT} && echo 提示: 用 scripts\\gsflow_gpu.cmd 跑训练(带CUDA环境)"',
              ROOT, "gsflow 命令行（项目根目录）")
    # 3) 开始菜单一组
    _make_lnk(sm / "1. Web GUI.lnk", WEB_CMD, "", ROOT, "启动 gsflow 图形界面")
    _make_lnk(sm / "2. 命令行（含 GPU 环境）.lnk", Path("cmd.exe"),
              f'/k "chcp 65001 >nul && title gsflow && cd /d {ROOT}"', ROOT, "gsflow 命令行")
    _make_lnk(sm / "3. 环境自检.lnk", Path("cmd.exe"),
              f'/k "chcp 65001 >nul && cd /d {ROOT} && scripts\\gsflow.cmd doctor"', ROOT,
              "gsflow 环境自检")
    _make_lnk(sm / "4. 构建便携 CUDA.lnk", BUILD_CUDA, "", ROOT,
              "从已下载的 CUDA 安装包构建免安装工具链")
    _make_lnk(sm / "5. 项目文件夹.lnk", ROOT, "", ROOT, "打开 gsflow 项目目录")
    print("\n完成。桌面 2 个；开始菜单若因权限被跳过，可在项目目录里直接运行 scripts\\gsflow.cmd。")
    return 0


def remove() -> int:
    desktop, sm = _desktop(), _start_menu()
    targets = [desktop / "gsflow Web GUI.lnk", desktop / "gsflow 命令行.lnk",
               sm / "1. Web GUI.lnk", sm / "2. 命令行（含 GPU 环境）.lnk",
               sm / "3. 环境自检.lnk", sm / "4. 构建便携 CUDA.lnk", sm / "5. 项目文件夹.lnk"]
    for t in targets:
        if t.exists():
            t.unlink()
            print(f"  已删除: {t}")
    if sm.exists() and not any(sm.iterdir()):
        sm.rmdir()
        print(f"  已删除空目录: {sm}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="创建/删除 gsflow 快捷方式")
    ap.add_argument("--remove", action="store_true", help="删除已创建的快捷方式")
    args = ap.parse_args()
    return remove() if args.remove else create()


if __name__ == "__main__":
    raise SystemExit(main())
