"""事件总线：流程核心 -> CLI 控制台 / GUI SSE 的唯一通道。

core 只负责 emit，不关心谁在听：
  * CLI 注册一个把事件打印到终端的 sink；
  * Web 注册一个把事件塞进内存环形缓冲区 + 每作业订阅队列的 sink。

线程安全：pipeline 在后台线程跑，Web 在事件循环里读，因此所有
订阅表与缓冲区都用一把 RLock 保护。
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

LEVELS = ("debug", "info", "warn", "error", "success")


@dataclass
class Event:
    level: str
    message: str
    stage: str = ""
    job_id: str = ""
    progress: float | None = None      # 0..1，None 表示不确定进度
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    seq: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Event":
        raw = json.loads(line)
        known = {f for f in Event.__dataclass_fields__}  # type: ignore[attr-defined]
        return Event(**{k: v for k, v in raw.items() if k in known})

    def human(self) -> str:
        clock = time.strftime("%H:%M:%S", time.localtime(self.ts))
        tag = f"[{self.stage}]" if self.stage else ""
        prog = f" {self.progress * 100:5.1f}%" if self.progress is not None else ""
        prefix = {"warn": "WARN ", "error": "ERROR", "success": "OK   "}.get(self.level, "")
        return f"{clock} {prefix}{tag}{prog} {self.message}".rstrip()


Sink = Callable[[Event], None]


class EventBus:
    """极简发布/订阅 + 环形缓冲（供 GUI 新连接回放历史）。"""

    def __init__(self, capacity: int = 2000):
        self._lock = threading.RLock()
        self._sinks: list[Sink] = []
        self._buffer: deque[Event] = deque(maxlen=capacity)
        self._seq = 0
        self._jsonl: Path | None = None
        self._jsonl_lock = threading.Lock()

    # --- 订阅 ---
    def subscribe(self, sink: Sink) -> Callable[[], None]:
        with self._lock:
            self._sinks.append(sink)

        def unsubscribe() -> None:
            with self._lock:
                if sink in self._sinks:
                    self._sinks.remove(sink)

        return unsubscribe

    # --- 持久化 ---
    def attach_jsonl(self, path: Path) -> None:
        """把事件同时追加写入 JSONL（断点续跑/事后排障）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._jsonl = path

    def history(self, since_seq: int = 0) -> list[Event]:
        with self._lock:
            return [e for e in self._buffer if e.seq > since_seq]

    # --- 发布 ---
    def emit(
        self,
        message: str,
        level: str = "info",
        stage: str = "",
        job_id: str = "",
        progress: float | None = None,
        **data: Any,
    ) -> Event:
        if level not in LEVELS:
            level = "info"
        with self._lock:
            self._seq += 1
            evt = Event(
                level=level,
                message=message,
                stage=stage,
                job_id=job_id,
                progress=progress,
                data=data,
                seq=self._seq,
            )
            self._buffer.append(evt)
            sinks = list(self._sinks)
        for sink in sinks:
            try:
                sink(evt)
            except Exception:  # 订阅者自己的错误不能拖垮流程
                pass
        if self._jsonl is not None:
            with self._jsonl_lock:
                try:
                    with self._jsonl.open("a", encoding="utf-8") as fh:
                        fh.write(evt.to_json() + "\n")
                except OSError:
                    pass
        return evt

    def log(self, message: str, **kw: Any) -> Event:
        return self.emit(message, **kw)

    def info(self, message: str, **kw: Any) -> Event:
        return self.emit(message, level="info", **kw)

    def warn(self, message: str, **kw: Any) -> Event:
        return self.emit(message, level="warn", **kw)

    def error(self, message: str, **kw: Any) -> Event:
        return self.emit(message, level="error", **kw)

    def success(self, message: str, **kw: Any) -> Event:
        return self.emit(message, level="success", **kw)


def console_sink(prefix: str = "") -> Sink:
    """CLI 用：直接打到 stdout，保证 CLI 与 GUI 看到同样的信息粒度。"""

    def _sink(evt: Event) -> None:
        line = (prefix + evt.human()) if prefix else evt.human()
        print(line, flush=True)

    return _sink


def tee_jsonl(path: Path) -> Sink:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _sink(evt: Event) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(evt.to_json() + "\n")

    return _sink


def read_jsonl(path: Path) -> Iterable[Event]:
    if not path.exists():
        return []
    events: list[Event] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(Event.from_json(line))
        except (json.JSONDecodeError, TypeError):
            continue
    return events
