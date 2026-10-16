"""尝试把 torchvision 的 VGG16 权重下载到工作区内的 TORCH_HOME。

为什么单独做：本项目**网络受限**（raw.githubusercontent.com SSL 失败、curl/PowerShell 的
HTTPS 报 SEC_E_NO_CREDENTIALS），而 LPIPS 需要 VGG/AlexNet 权重。
先把权重落到工作区，训练时就不会在第一个 step 上卡住或报错。

同时把 `TORCH_HOME` 固定到工作区内 —— 默认会写 `%USERPROFILE%/.cache/torch`，
沙箱下不可写（本项目已多次踩这类坑）。

用法: python scripts/fetch_torch_weights.py [--model vgg16]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

URLS = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vgg16", choices=sorted(URLS))
    args = ap.parse_args()

    torch_home = ROOT / "models" / "torch"
    (torch_home / "hub" / "checkpoints").mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)
    print(f"TORCH_HOME = {torch_home}")

    url = URLS[args.model]
    name = url.rsplit("/", 1)[-1]
    dest = torch_home / "hub" / "checkpoints" / name
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        print(f"  已存在：{dest}  ({dest.stat().st_size / 1048576:.1f} MB)")
        return 0

    import urllib.request

    print(f"  下载 {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=60) as r, dest.open("wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = r.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r    {got / 1048576:6.1f} / {total / 1048576:.1f} MB", end="")
        print()
    except Exception as e:  # noqa: BLE001
        print(f"\n  **下载失败** {type(e).__name__}: {str(e)[:120]}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        print("  ⇒ LPIPS 将不可用；训练侧必须**优雅退化**为 L1+SSIM，而不是中断或静默给 0。")
        return 1

    print(f"  已保存 {dest}  ({dest.stat().st_size / 1048576:.1f} MB)")
    # 立刻验证能被 torchvision 加载
    try:
        import torchvision.models as M
        import torch
        net = M.vgg16(weights=None)
        sd = torch.load(dest, map_location="cpu", weights_only=True)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f"  校验加载：missing={len(missing)} unexpected={len(unexpected)}")
    except Exception as e:  # noqa: BLE001
        print(f"  **校验失败** {type(e).__name__}: {str(e)[:120]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
