"""gsflow 安装器：环境检测 + 可选的组件安装（**纯标准库**，含 Tkinter 图形界面）。

## 设计要点（为什么这样写）

1. **必须在建 venv 之前就能跑**：所以只用标准库（`tkinter` / `subprocess` / `urllib`），
   不 import 本项目任何模块 —— 装完才会有依赖。
2. **与既有脚本同源**：创建环境/装依赖的逻辑复用 `scripts/bootstrap.ps1` 的做法
   （uv 建 venv、`.pth` 挂 `src`、按需 cu128 的 torch），下载外部工具/权重复用
   `scripts/download_tools.py` / `download_colmap.py` / `download_weights.py` /
   `fetch_torch_weights.py`。安装器不重写这些，只做**编排 + 进度可视化**。
3. **可静默**：`--silent` 供自动化测试与无人值守安装；GUI 只是同一套动作的界面。
4. **可复用本机缓存**：把已有安装的 `env/uv-cache`、`env/pip-cache`、`models/`、`tools/`
   拷过来，避免重复下载（本机实测能省下数 GB 流量与十几分钟）。
5. **不含测试内容**：不安装 `tests/`、不创建 `workspace/` 里的实验产物。

## 用法

```bat
:: 图形界面（推荐）
python installer\\gsflow_installer.py

:: 无人值守
python installer\\gsflow_installer.py --silent --path F:\\3dgs-flow ^
       --components core,gpu,tools,weights --reuse-from F:\\3dgs-and-ai

:: 只做环境检测，不装任何东西
python installer\\gsflow_installer.py --check-only
```
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 常量

COMPONENTS = {
    "core": "核心依赖（numpy / pillow / fastapi / uvicorn …，必需）",
    "gpu": "GPU 训练（torch cu128 + gsplat，约 2.6 GB）",
    "segment": "AI 分割后端（rembg / ultralytics + 权重）",
    "tools": "外部工具（COLMAP、FFmpeg，约 900 MB）",
    "weights": "模型权重（VGG16/LPIPS、FastSAM）",
}
MIN_FREE_GB = 8.0
MIN_FREE_GB_FULL = 20.0

_REQUIRED_PY = (3, 10)


# ---------------------------------------------------------------- 环境检测

@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def check_environment(install_dir: Path, components: list[str]) -> list[Check]:
    out: list[Check] = []

    v = sys.version_info
    out.append(Check("Python", v >= _REQUIRED_PY,
                     f"{v.major}.{v.minor}.{v.micro}（要求 ≥{_REQUIRED_PY[0]}.{_REQUIRED_PY[1]}）"))

    uv = shutil.which("uv")
    out.append(Check("uv", bool(uv), uv or "未安装（安装器会尝试用 pip 装 uv）"))

    tk_ok = True
    try:
        import tkinter  # noqa: F401
    except Exception:  # noqa: BLE001
        tk_ok = False
    out.append(Check("Tkinter（GUI 需要）", tk_ok, "可用" if tk_ok else "不可用（仍可用 --silent）"))

    # GPU
    gpu_msg, gpu_ok = "未检测到（不影响 CPU 流程）", False
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
            if r.returncode == 0 and r.stdout.strip():
                gpu_ok = True
                gpu_msg = r.stdout.strip().splitlines()[0]
        except Exception as exc:  # noqa: BLE001
            gpu_msg = f"查询失败：{type(exc).__name__}"
    out.append(Check("NVIDIA GPU", gpu_ok, gpu_msg))

    # 磁盘
    need = MIN_FREE_GB_FULL if ("gpu" in components or "tools" in components) else MIN_FREE_GB
    try:
        target = install_dir if install_dir.exists() else install_dir.parent
        free_gb = shutil.disk_usage(str(target)).free / (1024 ** 3)
        out.append(Check("磁盘空间", free_gb >= need, f"可用 {free_gb:.1f} GB（需要 ≥{need:.0f} GB）"))
    except Exception as exc:  # noqa: BLE001
        out.append(Check("磁盘空间", False, f"无法检测：{type(exc).__name__}"))

    # 写权限
    try:
        install_dir.mkdir(parents=True, exist_ok=True)
        probe = install_dir / ".installer-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        out.append(Check("安装目录可写", True, str(install_dir)))
    except Exception as exc:  # noqa: BLE001
        out.append(Check("安装目录可写", False, f"{install_dir}（{type(exc).__name__}: {exc}）"))

    return out


# ---------------------------------------------------------------- 安装动作

@dataclass
class Plan:
    install_dir: Path
    components: list[str] = field(default_factory=lambda: ["core"])
    reuse_from: Path | None = None
    python_exe: str = ""
    dry_run: bool = False


class Runner:
    """执行命令并把输出逐行回调（GUI 的实时日志 / 静默模式的 stdout）。"""

    def __init__(self, log):
        self.log = log
        self.cancelled = False

    def run(self, cmd: list[str], cwd: Path, env: dict | None = None) -> int:
        self.log("$ " + " ".join(cmd))
        p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace", bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            self.log(line.rstrip())
            if self.cancelled:
                p.terminate()
                break
        return p.wait()


def _install_env(root: Path) -> dict:
    """子进程环境：所有缓存/临时目录都关进安装目录（沙箱与整洁部署都要求这样做）。"""
    env = os.environ.copy()
    tmp = root / "env" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env.update({
        "TEMP": str(tmp), "TMP": str(tmp),
        "UV_CACHE_DIR": str(root / "env" / "uv-cache"),
        "PIP_CACHE_DIR": str(root / "env" / "pip-cache"),
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "YOLO_CONFIG_DIR": str(root / "env" / "config" / "ultralytics"),
        "MPLCONFIGDIR": str(root / "env" / "config" / "matplotlib"),
        "REMBG_HOME": str(root / "models" / "rembg"),
        "U2NET_HOME": str(root / "models" / "rembg"),
        "TORCH_HOME": str(root / "models" / "torch"),
    })
    return env


def _ensure_uv(runner: Runner, root: Path, env: dict, python_exe: str) -> str:
    uv = shutil.which("uv")
    if uv:
        return uv
    runner.log("未找到 uv，尝试用 pip 安装（uv 建 venv 比 python -m venv 稳定）...")
    runner.run([python_exe, "-m", "pip", "install", "--user", "uv"], root, env)
    uv = shutil.which("uv") or str(Path.home() / "AppData" / "Roaming" / "Python" /
                                   f"Python{sys.version_info.major}{sys.version_info.minor}" /
                                   "Scripts" / "uv.exe")
    return uv


def _copy_reuse(src: Path, dst: Path, runner: Runner) -> None:
    """复用已有安装的缓存与模型（省流量、省时间）。"""
    # 只复制**纯数据**目录（模型权重、外部工具）；
    # uv/pip 缓存含锁文件与 .tmp* 临时目录，复制会报 WinError 5，
    # 且半截缓存会让后续安装仍需联网（实测）。缓存改为直接指向源。
    pairs = [("models", "models"), ("tools", "tools")]
    for a, b in pairs:
        s, d = src / a, dst / b
        if not s.exists():
            continue
        runner.log(f"复用 {s} → {d}")
        try:
            shutil.copytree(s, d, dirs_exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            runner.log(f"  复制失败（忽略）：{type(exc).__name__}: {exc}")


def _materialize_payload(root: Path, source: Path, runner: Runner) -> None:
    """把项目本体复制到安装目录（安装器自身所在目录就是"源"）。

    **这是"安装即用"的关键**：目标目录是空的，没有 `src/`、`requirements.txt`、脚本，
    光建 venv 是跑不起来的。复制清单刻意**排除**运行期产物与测试内容：
      `env/ workspace/ tools/ models/ downloads/ data/ logs/ tests/`
    """
    if (root / "src" / "gsflow").exists():
        runner.log("安装目录已有项目源码，跳过复制")
        return
    if not (source / "src" / "gsflow").exists():
        raise FileNotFoundError(f"源目录里没有 src/gsflow：{source}")

    runner.log(f"复制项目本体：{source} → {root}")
    skip = {"env", "workspace", "tools", "models", "downloads", "data", "logs",
            "tests", ".git", "__pycache__", ".pytest_cache",
            # 开发者向的内部文件：不属于给用户的安装内容
            "AGENTS.md"}
    for item in source.iterdir():
        if item.name in skip:
            continue
        dst = root / item.name
        try:
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)
        except Exception as exc:  # noqa: BLE001
            runner.log(f"  复制 {item.name} 失败（忽略）：{type(exc).__name__}: {exc}")


def install(plan: Plan, runner: Runner) -> bool:
    root = plan.install_dir
    root.mkdir(parents=True, exist_ok=True)
    env = _install_env(root)
    python_exe = plan.python_exe or sys.executable
    log = runner.log

    log(f"安装目录：{root}")
    log(f"组件：{', '.join(plan.components)}")

    # 先把项目本体放进去（空目录时）
    try:
        _materialize_payload(root, Path(__file__).resolve().parents[1], runner)
    except Exception as exc:  # noqa: BLE001
        log(f"复制项目本体失败：{type(exc).__name__}: {exc}")
        return False

    if plan.reuse_from and plan.reuse_from.exists():
        # ⚠️ **不要复制 uv/pip 缓存**（2026-10-25 实测）：
        #   * 缓存目录里有 `.tmp*` 临时目录与锁定文件，`copytree` 会因 `WinError 5` 中断，
        #     半截的缓存反而导致后续 `uv pip install` 仍要联网并失败；
        #   * 直接把 `UV_CACHE_DIR` / `PIP_CACHE_DIR` **指向源缓存**，零拷贝、零风险。
        # 只复制"纯数据"目录（models/tools），它们没有锁文件。
        env["UV_CACHE_DIR"] = str(plan.reuse_from / "env" / "uv-cache")
        env["PIP_CACHE_DIR"] = str(plan.reuse_from / "env" / "pip-cache")
        log(f"复用缓存（直接指向）：{env['UV_CACHE_DIR']}")
        _copy_reuse(plan.reuse_from, root, runner)

    if plan.dry_run:
        log("（dry-run）以下步骤会被执行，但本次不真正执行。")
        return True

    venv = root / "env" / "venv"
    venv_py = venv / "Scripts" / "python.exe"

    if not venv_py.exists():
        uv = _ensure_uv(runner, root, env, python_exe)
        if not Path(uv).exists():
            log("uv 不可用，退回 python -m venv")
            if runner.run([python_exe, "-m", "venv", str(venv)], root, env) != 0:
                log("创建虚拟环境失败")
                return False
        else:
            if runner.run([uv, "venv", str(venv), "--python", python_exe], root, env) != 0:
                log("uv venv 失败")
                return False
            runner.run([uv, "pip", "install", "--python", str(venv_py), "pip"], root, env)
    else:
        log("复用已有虚拟环境")

    uv = shutil.which("uv")
    def pipi(args: list[str]) -> int:
        if uv:
            return runner.run([uv, "pip", "install", "--python", str(venv_py)] + args, root, env)
        return runner.run([str(venv_py), "-m", "pip", "install"] + args, root, env)

    req = root / "requirements.txt"
    if req.exists():
        # 本机网络会间歇性失败（error sending request for url）⇒ 重试 3 次
        ok = False
        for attempt in range(1, 4):
            if pipi(["-r", str(req)]) == 0:
                ok = True
                break
            log(f"依赖安装第 {attempt} 次失败，重试…")
        if not ok:
            log("核心依赖安装失败（已重试 3 次）")
            return False
    else:
        log("警告：未找到 requirements.txt（安装包不完整？）")

    # 挂载 src（本机 PEP517 不可用，统一用 .pth）
    site = venv / "Lib" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    (site / "gsflow-dev.pth").write_text(str(root / "src"), encoding="ascii")
    log("已挂载 src → site-packages/gsflow-dev.pth")

    if "gpu" in plan.components:
        log("安装 torch(cu128) + gsplat（约 2.6 GB）...")
        if pipi(["--index-url", "https://download.pytorch.org/whl/cu128",
                 "torch", "torchvision"]) != 0:
            log("torch 安装失败（可稍后重试）")
            return False
        if pipi(["gsplat"]) != 0:
            log("gsplat 安装失败")
            return False

    if "segment" in plan.components:
        log("安装分割后端...")
        pipi(["--no-deps", "ultralytics", "rembg"])
        pipi(["onnxruntime", "opencv-python-headless", "pyyaml", "tqdm",
              "psutil", "py-cpuinfo", "pymatting", "pooch", "scikit-image",
              "numba", "lazy-loader", "platformdirs", "jsonschema", "attrs"])

    dl = root / "scripts"
    if "tools" in plan.components and (dl / "download_tools.py").exists():
        runner.run([str(venv_py), str(dl / "download_tools.py")], root, env)
        if (dl / "download_colmap.py").exists():
            runner.run([str(venv_py), str(dl / "download_colmap.py")], root, env)
    if "weights" in plan.components:
        if (dl / "fetch_torch_weights.py").exists():
            runner.run([str(venv_py), str(dl / "fetch_torch_weights.py")], root, env)
        if (dl / "download_weights.py").exists():
            runner.run([str(venv_py), str(dl / "download_weights.py"), "fastsam"], root, env)

    # 配置与目录
    for d in ("config", "workspace", "logs", "models", "tools", "downloads"):
        (root / d).mkdir(parents=True, exist_ok=True)

    # ---- 重写本机配置（2026-10-25）----
    #
    # 实测：装完后 config/config.toml 里的 workspace_dir 仍指向**源安装**
    # （载荷复制把开发机的配置带过来了）—— 用户一跑作业就会写到别的目录。
    # 用 gsflow init --force 重写，它的路径推导基于包位置（此时 src 已挂到安装目录）。
    try:
        runner.run([str(venv_py), "-m", "gsflow", "init", "--force"], root, env)
        cfg = root / "config" / "config.toml"
        if cfg.exists():
            txt = cfg.read_text(encoding="utf-8")
            # Windows 路径在 toml 里是双反斜杠，比较时统一成正斜杠再判断
            norm = txt.replace("\\\\", "/").replace("\\", "/")
            if str(root).replace("\\", "/") in norm:
                log("配置已指向安装目录")
            else:
                log("⚠ 配置里的 workspace_dir 似乎不是安装目录，请检查 " + str(cfg))
    except Exception as exc:  # noqa: BLE001
        log(f"重写配置失败（可手动运行 gsflow init）：{type(exc).__name__}: {exc}")
    settings = root / "config" / "settings.json"
    if not settings.exists():
        settings.write_text(json.dumps({"installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "components": plan.components}, ensure_ascii=False,
                                       indent=2), encoding="utf-8")

    # 验证
    log("环境验证：")
    runner.run([str(venv_py), "-m", "gsflow", "doctor"], root, env)
    log("安装完成。运行 scripts\\web.cmd 启动 Web GUI，或 scripts\\gsflow.cmd doctor 自检。")
    return True


# ---------------------------------------------------------------- GUI

def run_gui() -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("gsflow 安装器")
    root.geometry("860x640")
    state = {"dir": tk.StringVar(value=str(Path("F:/3dgs-flow"))),
             "reuse": tk.StringVar(value=""),
             "comps": {k: tk.BooleanVar(value=k == "core") for k in COMPONENTS},
             "running": False}

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=10, pady=10)

    # 页 1：环境
    f1 = ttk.Frame(nb)
    nb.add(f1, text="① 环境检测")
    t1 = tk.Text(f1, height=16, wrap="word")
    t1.pack(fill="both", expand=True)
    def do_check():
        t1.delete("1.0", "end")
        comps = [k for k, v in state["comps"].items() if v.get()]
        for c in check_environment(Path(state["dir"].get()), comps):
            t1.insert("end", f"{'✅' if c.ok else '⚠️'} {c.name}: {c.detail}\n")
    ttk.Button(f1, text="重新检测", command=do_check).pack(pady=6)
    do_check()

    # 页 2：路径与组件
    f2 = ttk.Frame(nb)
    nb.add(f2, text="② 路径与组件")
    row = ttk.Frame(f2); row.pack(fill="x", pady=8)
    ttk.Label(row, text="安装路径").pack(side="left")
    ttk.Entry(row, textvariable=state["dir"], width=64).pack(side="left", padx=6)
    ttk.Button(row, text="浏览…", command=lambda: state["dir"].set(
        filedialog.askdirectory(initialdir="/") or state["dir"].get())).pack(side="left")
    row2 = ttk.Frame(f2); row2.pack(fill="x", pady=4)
    ttk.Label(row2, text="复用已有安装（可选，省流量）").pack(side="left")
    ttk.Entry(row2, textvariable=state["reuse"], width=64).pack(side="left", padx=6)
    ttk.Button(row2, text="浏览…", command=lambda: state["reuse"].set(
        filedialog.askdirectory() or state["reuse"].get())).pack(side="left")
    for k, desc in COMPONENTS.items():
        ttk.Checkbutton(f2, text=f"{k} — {desc}", variable=state["comps"][k]).pack(anchor="w", pady=2)

    # 页 3：安装
    f3 = ttk.Frame(nb)
    nb.add(f3, text="③ 安装")
    log_w = tk.Text(f3, height=22, wrap="word")
    log_w.pack(fill="both", expand=True)
    pb = ttk.Progressbar(f3, mode="indeterminate")
    pb.pack(fill="x", pady=6)

    def log(msg: str) -> None:
        log_w.insert("end", msg + "\n")
        log_w.see("end")
        root.update_idletasks()

    def start():
        if state["running"]:
            return
        state["running"] = True
        pb.start(12)
        comps = [k for k, v in state["comps"].items() if v.get()]
        plan = Plan(install_dir=Path(state["dir"].get()), components=comps or ["core"],
                    reuse_from=Path(state["reuse"].get()) if state["reuse"].get() else None)
        runner = Runner(log)
        def work():
            ok = False
            try:
                ok = install(plan, runner)
            except Exception as exc:  # noqa: BLE001
                log(f"安装异常：{type(exc).__name__}: {exc}")
            pb.stop(); state["running"] = False
            if ok:
                log("✔ 安装成功")
                messagebox.showinfo("gsflow", "安装完成。\n可运行 scripts\\web.cmd 启动 GUI。")
            else:
                log("✘ 安装失败，请查看上方日志")
                messagebox.showerror("gsflow", "安装失败，详见日志。")
        threading.Thread(target=work, daemon=True).start()

    ttk.Button(f3, text="开始安装", command=start).pack(pady=4)
    root.mainloop()
    return 0


# ---------------------------------------------------------------- 入口

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="gsflow 安装器")
    ap.add_argument("--silent", action="store_true", help="无界面安装")
    ap.add_argument("--check-only", action="store_true", help="只做环境检测")
    ap.add_argument("--path", default="", help="安装目录")
    ap.add_argument("--components", default="core",
                    help="逗号分隔：core,gpu,segment,tools,weights")
    ap.add_argument("--reuse-from", default="", help="复用该目录的缓存与模型")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划")
    ap.add_argument("--python", default="", help="用于创建 venv 的 Python 可执行文件")
    args = ap.parse_args(argv)

    comps = [c.strip() for c in args.components.split(",") if c.strip()]
    bad = [c for c in comps if c not in COMPONENTS]
    if bad:
        print(f"未知组件：{bad}；可用：{list(COMPONENTS)}")
        return 2

    target = Path(args.path) if args.path else Path.cwd() / "gsflow-install"
    if args.check_only:
        for c in check_environment(target, comps):
            print(f"{'OK ' if c.ok else 'WARN'} {c.name}: {c.detail}")
        return 0

    if not args.silent:
        try:
            import tkinter  # noqa: F401
            return run_gui()
        except Exception as exc:  # noqa: BLE001
            print(f"GUI 不可用（{type(exc).__name__}: {exc}），退回静默安装")
    plan = Plan(install_dir=target, components=comps,
                reuse_from=Path(args.reuse_from) if args.reuse_from else None,
                python_exe=args.python, dry_run=args.dry_run)
    return 0 if install(plan, Runner(print)) else 1


if __name__ == "__main__":
    sys.exit(main())
