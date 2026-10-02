"""统一异常类型。

外部工具（colmap/ffmpeg/分割模型）失败一律包成 ExternalToolError，
带“可执行的修复建议”，GUI 与 CLI 原样展示 advice 字段。
"""

from __future__ import annotations


class GsflowError(Exception):
    """所有本项目异常的基类。"""


class ConfigError(GsflowError):
    """配置缺失或非法。"""


class ToolMissingError(GsflowError):
    """外部工具不存在。"""

    def __init__(self, tool: str, advice: str, searched: list[str] | None = None):
        self.tool = tool
        self.advice = advice
        self.searched = searched or []
        super().__init__(f"缺少外部工具 {tool}。{advice}")


class StageError(GsflowError):
    """某个阶段执行失败。"""

    def __init__(self, stage: str, message: str, advice: str = ""):
        self.stage = stage
        self.advice = advice
        super().__init__(f"[{stage}] {message}" + (f" 建议：{advice}" if advice else ""))


class StageCancelled(GsflowError):
    """作业被用户取消。"""


class CredentialMissingError(GsflowError):
    """需要 API Key / Token 但未配置。"""

    def __init__(self, key: str, advice: str):
        self.key = key
        self.advice = advice
        super().__init__(f"缺少凭据 {key}。{advice}")
