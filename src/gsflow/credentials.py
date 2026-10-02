"""凭据存储：API Key / HuggingFace Token / 镜像地址。

存放于 config/credentials.json，权限收紧到仅当前用户可读写，
并默认写入 .gitignore。GUI 设置页与 CLI `gsflow creds` 共用同一份实现。

字段（可扩展，未知字段原样保留）：
  hf_token       HuggingFace Token（SAM2/SAM3 权重下载）
  hf_endpoint    镜像地址，例如 https://hf-mirror.com
  sam3_api_key   SAM3 在线 API Key（若走托管接口）
  openai_api_key 预留：后续接 SD-inpaint / VLM 打标
  http_proxy     形如 http://127.0.0.1:7890
  https_proxy
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import paths
from .errors import CredentialMissingError

CRED_FILE = paths.CONFIG_DIR / "credentials.json"
EXAMPLE_FILE = paths.CONFIG_DIR / "credentials.example.json"

KNOWN_KEYS = [
    "hf_token",
    "hf_endpoint",
    "sam3_api_key",
    "openai_api_key",
    "http_proxy",
    "https_proxy",
]

_EXAMPLE = {k: "" for k in KNOWN_KEYS}


def load_credentials() -> dict[str, Any]:
    if not CRED_FILE.exists():
        return {}
    try:
        return json.loads(CRED_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_credentials(data: dict[str, Any]) -> Path:
    """写入凭据文件，并把文件权限收紧到当前用户。"""
    CRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged = load_credentials()
    merged.update({k: v for k, v in data.items() if v is not None})
    CRED_FILE.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    _harden(CRED_FILE)
    _ensure_gitignore()
    if not EXAMPLE_FILE.exists():
        EXAMPLE_FILE.write_text(
            json.dumps(_EXAMPLE, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return CRED_FILE


def mask_credentials(data: dict[str, Any] | None = None) -> dict[str, str]:
    """给 GUI 展示用：不泄露明文，只显示是否已配置与尾 4 位。"""
    data = data if data is not None else load_credentials()
    out: dict[str, str] = {}
    for k in KNOWN_KEYS:
        v = str(data.get(k, "") or "")
        if not v:
            out[k] = ""
        elif k.endswith("endpoint") or k.endswith("proxy"):
            out[k] = v
        else:
            out[k] = "****" + v[-4:] if len(v) > 4 else "****"
    return out


def get(key: str, required_for: str = "") -> str | None:
    val = str(load_credentials().get(key, "") or "")
    if not val:
        return None
    return val


def require(key: str, advice: str) -> str:
    val = get(key)
    if not val:
        raise CredentialMissingError(key, advice)
    return val


def apply_to_env(env: dict[str, str]) -> dict[str, str]:
    """把凭据/代理注入子进程环境（供 huggingface_hub / requests 使用）。"""
    creds = load_credentials()
    if creds.get("hf_token"):
        env.setdefault("HF_TOKEN", str(creds["hf_token"]))
        env.setdefault("HUGGING_FACE_HUB_TOKEN", str(creds["hf_token"]))
    if creds.get("hf_endpoint"):
        env.setdefault("HF_ENDPOINT", str(creds["hf_endpoint"]))
    if creds.get("http_proxy"):
        env.setdefault("HTTP_PROXY", str(creds["http_proxy"]))
    if creds.get("https_proxy"):
        env.setdefault("HTTPS_PROXY", str(creds["https_proxy"]))
    return env


def _harden(path: Path) -> None:
    """尽量收紧权限；失败不致命（NTFS 上已由 ACL 继承控制）。"""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _ensure_gitignore() -> None:
    gi = paths.PROJECT_ROOT / ".gitignore"
    wanted = ["config/credentials.json", "env/", "workspace/", "downloads/", "models/", "logs/", "__pycache__/", "*.pyc"]
    existing = gi.read_text(encoding="utf-8").splitlines() if gi.exists() else []
    missing = [w for w in wanted if w not in existing]
    if missing:
        text = "\n".join(existing + missing).strip() + "\n"
        gi.write_text(text, encoding="utf-8")
