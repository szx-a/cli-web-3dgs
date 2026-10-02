"""环境自检（gsflow doctor）：一次性输出软硬件 + 工具 + 依赖 + 待办清单。

这是“新人第一次跑”的唯一入口，也是排障第一站：
  * 探到的每一项都给出结论和“缺什么、怎么补”；
  * 输出 JSON（供 GUI 渲染）+ 纯文本（供 CLI 阅读）。
"""

from __future__ import annotations

import ctypes
import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .. import paths
from ..config import Settings, load as load_settings

PY_PACKAGES: list[tuple[str, str, str]] = [
    # (import 名, pip 名, 用途)
    ("numpy", "numpy", "数值/图像处理"),
    ("PIL", "pillow", "图像读写"),
    ("plyfile", "plyfile", "3DGS PLY 读写/导出"),
    ("torch", "torch", "训练与分割后端"),
    ("torchvision", "torchvision", "图像变换"),
    ("gsplat", "gsplat", "默认 3DGS 训练后端"),
    ("fastapi", "fastapi", "Web GUI"),
    ("uvicorn", "uvicorn", "Web GUI 服务器"),
    ("onnxruntime", "onnxruntime", "无门槛分割后端推理"),
    ("rembg", "rembg", "一键抠图后端"),
    ("ultralytics", "ultralytics", "FastSAM 后端"),
    ("transformers", "transformers", "SAM2/SAM3 后端"),
]


def _mem_gb() -> dict[str, float]:
    """取物理内存（不依赖 WMI，避免沙箱下 CIM 被拒）。"""
    out: dict[str, float] = {}
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            out["total_gb"] = round(stat.ullTotalPhys / 1024**3, 1)
            out["available_gb"] = round(stat.ullAvailPhys / 1024**3, 1)
            out["load_percent"] = float(stat.dwMemoryLoad)
    except Exception:
        pass
    return out


def _gpu_info() -> dict[str, Any]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"available": False, "reason": "未找到 nvidia-smi"}
    info: dict[str, Any] = {"available": True, "nvidia_smi": smi}
    try:
        proc = subprocess.run(
            [smi, "--query-gpu=name,driver_version,memory.total,memory.used,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=25, encoding="utf-8", errors="ignore",
        )
        rows = [l.strip() for l in (proc.stdout or "").splitlines() if l.strip()]
        info["gpus"] = []
        for row in rows:
            parts = [p.strip() for p in row.split(",")]
            if len(parts) >= 5:
                info["gpus"].append(
                    {
                        "name": parts[0],
                        "driver": parts[1],
                        "vram_total_mb": float(parts[2]),
                        "vram_used_mb": float(parts[3]),
                        "compute_cap": parts[4],
                    }
                )
    except (OSError, subprocess.SubprocessError) as exc:
        info["error"] = str(exc)
    return info


def _disk_info() -> list[dict[str, Any]]:
    out = []
    if sys.platform.startswith("win"):
        try:
            import string

            for letter in string.ascii_uppercase:
                root = Path(f"{letter}:\\")
                if not root.exists():
                    continue
                try:
                    usage = shutil.disk_usage(root)
                except OSError:
                    continue
                out.append(
                    {
                        "drive": f"{letter}:",
                        "total_gb": round(usage.total / 1024**3, 1),
                        "free_gb": round(usage.free / 1024**3, 1),
                    }
                )
        except Exception:
            pass
    else:
        usage = shutil.disk_usage("/")
        out.append({"drive": "/", "total_gb": round(usage.total / 1024**3, 1), "free_gb": round(usage.free / 1024**3, 1)})
    return out


def _cuda_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    nvcc = shutil.which("nvcc")
    info["nvcc"] = nvcc or ""
    for base in (r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA", r"D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"):
        p = Path(base)
        if p.exists():
            info.setdefault("toolkit_dirs", []).append(str(p))
            info.setdefault("toolkit_versions", []).extend(sorted(x.name for x in p.iterdir() if x.is_dir()))
    info["CUDA_HOME"] = str(Path(nvcc).parent.parent) if nvcc else ""
    info["CUDA_PATH"] = ""
    import os

    info["CUDA_PATH"] = os.environ.get("CUDA_PATH", "")
    return info


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        try:
            info["device"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            info["compute_capability"] = f"sm_{cap[0]}{cap[1]}"
            info["arch_list"] = list(torch.cuda.get_arch_list())
            sm = f"sm_{cap[0]}{cap[1]}"
            info["kernel_support"] = any(sm in a or a.endswith(f"{cap[0]}{cap[1]}") for a in info["arch_list"])
        except Exception as exc:  # pragma: no cover
            info["error"] = str(exc)
    return info


def _packages() -> list[dict[str, Any]]:
    from importlib.metadata import version as dist_version

    out = []
    for mod, pip_name, purpose in PY_PACKAGES:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "") or ""
            if not ver:
                # 有些包（如 plyfile）不暴露 __version__，从发行元数据取
                try:
                    ver = dist_version(pip_name)
                except Exception:
                    ver = ""
            out.append({"module": mod, "pip": pip_name, "purpose": purpose, "installed": True, "version": str(ver)})
        except Exception as exc:
            out.append({"module": mod, "pip": pip_name, "purpose": purpose, "installed": False, "error": type(exc).__name__})
    return out


def _tools(settings: Settings) -> list[dict[str, Any]]:
    from .locate import ADVICE, find_tool, probe_version, version_args

    out = []
    for name in ("ffmpeg", "ffprobe", "colmap"):
        configured = str(settings.get(f"tools.{name}", "") or "")
        try:
            info = find_tool(name, configured, required=False)
        except Exception as exc:
            out.append({"name": name, "found": False, "advice": ADVICE.get(name, ""), "error": str(exc)})
            continue
        if info is None:
            out.append({"name": name, "found": False, "advice": ADVICE.get(name, "")})
        else:
            version = probe_version(info.path, version_args(name))
            out.append({"name": name, "found": True, "path": info.path, "source": info.source, "version": version})
    return out


def _venv_info() -> dict[str, Any]:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "in_venv": in_venv,
        "expected_venv": str(paths.VENV_DIR),
        "using_expected_venv": Path(sys.executable).parent.parent.resolve() == paths.VENV_DIR.resolve()
        if sys.executable
        else False,
    }


def _build_plan(report: dict[str, Any]) -> list[dict[str, str]]:
    """根据探测结果生成有序待办清单（severity: blocker | warn | info）。"""
    plan: list[dict[str, str]] = []
    torch_info = report["torch"]
    gpus = report["gpu"].get("gpus") or []

    if gpus:
        cap = gpus[0].get("compute_cap", "")
        if cap.startswith("12"):
            plan.append(
                {
                    "severity": "info",
                    "item": "GPU 为 Blackwell（sm_120）",
                    "action": "CUDA 必须 12.8+；torch 必须 cu128 及以上；不要使用文档里的 CUDA 11.8。",
                }
            )
    else:
        plan.append({"severity": "warn", "item": "未检测到 NVIDIA GPU", "action": "训练阶段无法执行，仅能跑通清理与 COLMAP。"})

    if not report["cuda"].get("nvcc"):
        plan.append(
            {
                "severity": "warn",
                "item": "未检测到 nvcc（CUDA Toolkit）",
                "action": "gsplat 纯 pip 路线可暂不装；若要官方 rasterizer 或自编译扩展，装 CUDA 12.8/12.9。",
            }
        )

    if not torch_info.get("installed"):
        plan.append({"severity": "blocker", "item": "未安装 torch", "action": "运行 scripts\\bootstrap_m0.ps1 -Torch"})
    elif not torch_info.get("cuda_available"):
        plan.append({"severity": "blocker", "item": "torch 无法使用 CUDA", "action": "确认装的是 cu128+ 轮子且驱动 ≥ 对应版本。"})
    elif torch_info.get("kernel_support") is False:
        plan.append(
            {
                "severity": "blocker",
                "item": f"torch 内核不包含 {torch_info.get('compute_capability')}",
                "action": "升级 torch 到含 sm_120 的版本（≥2.7 + cu128）。",
            }
        )

    tools = {t["name"]: t for t in report["tools"]}
    if not tools.get("ffmpeg", {}).get("found"):
        plan.append({"severity": "info", "item": "缺少 ffmpeg", "action": "仅视频抽帧需要；图片输入可忽略。"})
    if not tools.get("colmap", {}).get("found"):
        plan.append({"severity": "blocker", "item": "缺少 COLMAP", "action": "下载官方 Windows CUDA 包到 tools\\colmap\\。"})

    pkgs = {p["module"]: p for p in report["packages"]}
    if not pkgs.get("gsplat", {}).get("installed"):
        plan.append({"severity": "warn", "item": "未安装 gsplat", "action": "默认训练后端需要：pip install gsplat"})
    if not pkgs.get("fastapi", {}).get("installed"):
        plan.append({"severity": "warn", "item": "未安装 fastapi/uvicorn", "action": "Web GUI 需要。"})
    if not (pkgs.get("rembg", {}).get("installed") or pkgs.get("ultralytics", {}).get("installed")):
        plan.append(
            {
                "severity": "warn",
                "item": "未安装任何无门槛分割后端",
                "action": "先装 ultralytics(FastSAM) 或 rembg，SAM3 需要 HF 授权，作为增强后端后接。",
            }
        )

    home = report.get("network", {})
    if home.get("huggingface_reachable") is False:
        plan.append(
            {
                "severity": "warn",
                "item": "huggingface.co 不可达",
                "action": "设置 HF 镜像（gui 设置页填 hf_endpoint=https://hf-mirror.com）或配置代理。",
            }
        )

    if not plan:
        plan.append({"severity": "info", "item": "环境检查通过", "action": "可以直接建作业开跑。"})
    return plan


def _network() -> dict[str, Any]:
    import socket
    import ssl

    out: dict[str, Any] = {}
    for host in ("pypi.org", "download.pytorch.org", "huggingface.co", "hf-mirror.com", "github.com"):
        try:
            sock = socket.create_connection((host, 443), timeout=6)
            ctx = ssl.create_default_context()
            ctx.wrap_socket(sock, server_hostname=host).close()
            out[f"{host.replace('.', '_')}_reachable"] = True
        except Exception:
            out[f"{host.replace('.', '_')}_reachable"] = False
    out["huggingface_reachable"] = out.get("huggingface_co_reachable", False)
    return out


def collect_report(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    report: dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "node": platform.node(),
        },
        "venv": _venv_info(),
        "memory": _mem_gb(),
        "gpu": _gpu_info(),
        "cuda": _cuda_info(),
        "torch": _torch_info(),
        "disks": _disk_info(),
        "tools": _tools(settings),
        "packages": _packages(),
        "network": _network(),
        "paths": {
            "project_root": str(paths.PROJECT_ROOT),
            "workspace_dir": str(settings.get("project.workspace_dir", "")),
            "tools_dir": str(paths.TOOLS_DIR),
            "tmp_dir": str(paths.TMP_DIR),
        },
    }
    report["plan"] = _build_plan(report)
    return report


def render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    p = report["platform"]
    lines.append("=" * 72)
    lines.append("gsflow doctor —— 环境自检")
    lines.append("=" * 72)
    lines.append(f"系统      : {p['system']} {p['release']} ({p['version']})")
    v = report["venv"]
    lines.append(f"Python    : {v['python']}  venv={v['in_venv']}  {v['executable']}")
    mem = report["memory"]
    if mem:
        lines.append(f"内存      : 总 {mem.get('total_gb')} GB / 可用 {mem.get('available_gb')} GB（占用 {mem.get('load_percent')}%）")

    gpu = report["gpu"]
    if gpu.get("gpus"):
        for g in gpu["gpus"]:
            lines.append(
                f"GPU       : {g['name']}  显存 {g['vram_total_mb']:.0f}MB（已用 {g['vram_used_mb']:.0f}MB）"
                f"  驱动 {g['driver']}  算力 {g['compute_cap']}"
            )
    else:
        lines.append(f"GPU       : 未检测到（{gpu.get('reason') or gpu.get('error', '')}）")

    cuda = report["cuda"]
    lines.append(f"CUDA      : nvcc={'有' if cuda.get('nvcc') else '无'}  版本目录={cuda.get('toolkit_versions') or '无'}")

    t = report["torch"]
    if t.get("installed"):
        lines.append(
            f"torch     : {t['version']}  cuda_build={t.get('cuda_build')}  可用={t.get('cuda_available')}"
            + (f"  {t.get('compute_capability')}  内核支持={t.get('kernel_support')}" if t.get("cuda_available") else "")
        )
    else:
        lines.append("torch     : 未安装")

    lines.append("磁盘      : " + " | ".join(f"{d['drive']} 余 {d['free_gb']}GB/{d['total_gb']}GB" for d in report["disks"]))
    lines.append("-" * 72)
    lines.append("外部工具")
    for tool in report["tools"]:
        if tool.get("found"):
            lines.append(f"  [OK]   {tool['name']:<8} {tool.get('version') or ''}  ({tool['source']}) {tool['path']}")
        else:
            lines.append(f"  [MISS] {tool['name']:<8} {tool.get('advice', '')}")
    lines.append("-" * 72)
    lines.append("Python 依赖")
    for pkg in report["packages"]:
        mark = "[OK]  " if pkg["installed"] else "[MISS]"
        lines.append(f"  {mark} {pkg['module']:<14} {pkg.get('version', ''):<12} {pkg['purpose']}")
    lines.append("-" * 72)
    lines.append("网络")
    for k, val in report["network"].items():
        if k.endswith("_reachable"):
            lines.append(f"  {'[OK]  ' if val else '[FAIL]'} {k.replace('_reachable', '').replace('_', '.')}")
    lines.append("-" * 72)
    lines.append("待办清单")
    for i, item in enumerate(report["plan"], 1):
        tag = {"blocker": "!! 阻断", "warn": "!  注意", "info": "   提示"}[item["severity"]]
        lines.append(f"  {i}. {tag} {item['item']}")
        lines.append(f"      -> {item['action']}")
    lines.append("=" * 72)
    return "\n".join(lines)


def write_report(report: dict[str, Any], path: Path | None = None) -> Path:
    target = path or (paths.PROJECT_ROOT / "docs" / "env-report.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return target
