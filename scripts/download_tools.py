"""工具下载器：COLMAP / FFmpeg / 分割权重（统一用 urllib，避开 PowerShell 的 Schannel 问题）。

用法：
    python scripts/download_tools.py colmap
    python scripts/download_tools.py ffmpeg
    python scripts/download_tools.py --list
    python scripts/download_tools.py --url <url> --out <path>       # 任意文件
"""

from __future__ import annotations

import argparse
import os
import shutil
import ssl
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
DOWNLOADS = ROOT / "downloads"

CATALOG: dict[str, dict[str, str]] = {
    "ffmpeg": {
        "url": "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
        "file": "ffmpeg-release-essentials.zip",
        "extract_to": "ffmpeg",
        "desc": "FFmpeg（视频抽帧 / ffprobe）",
    },
    "colmap": {
        "url": "https://github.com/colmap/colmap/releases/download/3.12.0/colmap-x64-windows-cuda.zip",
        "file": "colmap-x64-windows-cuda.zip",
        "extract_to": "colmap",
        "desc": "COLMAP 3.12.0 Windows CUDA 预编译版",
    },
    # CUDA Toolkit：gsplat / 官方光栅化器编译内核必须有 nvcc。
    # 注意：本机驱动 573.22 对应 CUDA 12.8，用 12.8.1 安装包最稳妥。
    "cuda": {
        "url": "https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_572.61_windows.exe",
        "file": "cuda_12.8.1_572.61_windows.exe",
        "extract_to": "",           # 安装包，不需要解压
        "desc": "CUDA Toolkit 12.8.1（提供 nvcc，RTX 50 / sm_120 必需）",
    },
}


def download_resumable(url: str, out: Path, name: str = "", chunk: int = 1024 * 256, retries: int = 200) -> Path:
    """支持断点续传的下载：写入 .part，记录已有字节数，失败自动重试。

    本机网络实测只有 0.2 MB/s 量级，3.3GB 的 CUDA 安装包必须能断点续传，
    否则一次抖动就要重来。
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    if out.exists() and out.stat().st_size > 0:
        print(f"已存在，跳过下载：{out}（{out.stat().st_size/1024/1024:.1f} MB）")
        return out

    total_remote = 0
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "gsflow/0.1"})
        with urllib.request.urlopen(req, context=_ctx(), timeout=30) as r:
            total_remote = int(r.headers.get("Content-Length") or 0)
    except Exception:
        pass

    attempt = 0
    while attempt < retries:
        attempt += 1
        have = part.stat().st_size if part.exists() else 0
        if total_remote and have >= total_remote:
            break
        headers = {"User-Agent": "gsflow/0.1"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=_ctx(), timeout=90) as resp:
                mode = "ab" if have and resp.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                t0 = time.time()
                last = time.time()
                with part.open(mode) as fh:
                    while True:
                        data = resp.read(chunk)
                        if not data:
                            break
                        fh.write(data)
                        have += len(data)
                        now = time.time()
                        if now - last > 3:
                            last = now
                            speed = have / max(now - t0, 1e-6) / 1024 / 1024
                            pct = f"{have*100/total_remote:5.1f}%" if total_remote else "  ?  "
                            print(f"  {name or out.name}: {pct}  {have/1024/1024:8.1f}/{total_remote/1024/1024:.1f} MB  {speed:5.2f} MB/s", flush=True)
        except Exception as exc:
            print(f"  第 {attempt} 次中断：{type(exc).__name__}: {exc}；5 秒后续传（已下载 {have/1024/1024:.1f} MB）", flush=True)
            time.sleep(5)
            continue
        if not total_remote or have >= total_remote:
            break
        time.sleep(2)

    if total_remote and part.stat().st_size < total_remote:
        raise RuntimeError(f"下载未完成：{part.stat().st_size}/{total_remote} 字节")
    part.replace(out)
    print(f"完成：{out}（{out.stat().st_size/1024/1024:.1f} MB）")
    return out


def download_and_extract(key: str) -> None:
    item = CATALOG[key]
    path = download_resumable(item["url"], DOWNLOADS / item["file"], item["desc"])
    if item["extract_to"]:
        dest = TOOLS / item["extract_to"]
        if dest.exists() and any(dest.iterdir()):
            print(f"目标已存在，跳过解压：{dest}")
        else:
            extract_zip(path, dest)
            flatten_single_dir(dest)
        print(f"完成：{key} -> {dest}")
    else:
        print(f"完成：{key} 安装包已就绪 -> {path}（请以管理员身份运行 scripts\\install_cuda.ps1 安装）")


def _ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    # 允许通过环境变量指定 CA bundle（企业代理/离线证书场景）
    ca = os.environ.get("GSFLOW_CA_BUNDLE")
    if ca and Path(ca).exists():
        ctx.load_verify_locations(ca)
    return ctx


def _progress_hook(name: str):
    state = {"t": time.time(), "last": 0}

    def hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = block_num * block_size
        now = time.time()
        if now - state["t"] < 2.0:
            return
        state["t"] = now
        pct = min(100.0, done * 100.0 / total_size)
        speed = (done - state["last"]) / 2.0 / 1024 / 1024
        state["last"] = done
        sys.stdout.write(f"\r  {name}: {pct:5.1f}%  ({done/1024/1024:7.1f} MB / {total_size/1024/1024:.1f} MB)  {speed:5.2f} MB/s")
        sys.stdout.flush()

    return hook


def download(url: str, out: Path, name: str = "") -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"已存在，跳过下载：{out}（{out.stat().st_size/1024/1024:.1f} MB）")
        return out
    print(f"下载 {name or url}\n  -> {out}")
    req = urllib.request.Request(url, headers={"User-Agent": "gsflow/0.1 (+windows)"})
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(req, context=_ctx(), timeout=60) as resp, tmp.open("wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        hook = _progress_hook(name or out.name)
        read = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            fh.write(chunk)
            read += len(chunk)
            hook(read // (1024 * 256), 1024 * 256, total)
    print()
    tmp.replace(out)
    print(f"完成：{out}（{out.stat().st_size/1024/1024:.1f} MB）")
    return out


def extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"解压 {zip_path.name} -> {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    print("解压完成")


def flatten_single_dir(dest: Path) -> None:
    """有些包的 zip 里多套了一层目录，这里把它提上来，便于固定路径引用。"""
    entries = [p for p in dest.iterdir() if p.is_dir()]
    files = [p for p in dest.iterdir() if p.is_file()]
    if len(entries) == 1 and not files:
        inner = entries[0]
        for item in inner.iterdir():
            shutil.move(str(item), str(dest / item.name))
        inner.rmdir()


def install(key: str) -> None:
    if key not in CATALOG:
        print(f"未知工具 '{key}'，可用的：{', '.join(CATALOG)}")
        return
    download_and_extract(key)


def main() -> int:
    ap = argparse.ArgumentParser(description="gsflow 工具下载器")
    ap.add_argument("tool", nargs="?", help="要安装的工具：colmap / ffmpeg")
    ap.add_argument("--list", action="store_true", help="列出可下载的工具")
    ap.add_argument("--url", help="直接下载任意 URL")
    ap.add_argument("--out", help="配合 --url 的输出路径")
    args = ap.parse_args()

    if args.list:
        for k, v in CATALOG.items():
            print(f"{k:<8} {v['desc']}\n         {v['url']}")
        return 0
    if args.url:
        if not args.out:
            print("--url 需要同时给 --out")
            return 2
        download(args.url, Path(args.out))
        return 0
    if not args.tool:
        ap.print_help()
        return 2
    install(args.tool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
