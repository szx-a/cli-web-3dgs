"""GUI 后台任务管理：把每次 run 放进工作线程，并保留事件总线供 SSE 订阅。

单卡训练场景下刻意串行（runtime.max_parallel_jobs 默认 1），
避免两个作业抢显存互相 OOM。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings, load as load_settings
from ..events import EventBus
from ..pipeline.orchestrator import JobSpec, Orchestrator


@dataclass
class JobHandle:
    run_id: str
    job_id: str
    spec: dict[str, Any]
    status: str = "queued"           # queued | running | done | failed | cancelled
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    bus: EventBus = field(default_factory=EventBus)
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


class JobManager:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self._lock = threading.RLock()
        self._handles: dict[str, JobHandle] = {}
        self._queue: list[str] = []
        self._running: str | None = None
        self._max_parallel = int(self.settings.get("runtime.max_parallel_jobs", 1) or 1)

    # ------------------------------------------------------------------ 提交
    def submit(self, spec: JobSpec, dry_run: bool = False) -> JobHandle:
        run_id = uuid.uuid4().hex[:12]
        handle = JobHandle(run_id=run_id, job_id=spec.job_id or "", spec=spec.to_dict())
        with self._lock:
            self._handles[run_id] = handle
            self._queue.append(run_id)
        t = threading.Thread(target=self._worker, args=(run_id, spec, dry_run), name=f"gsflow-{run_id}", daemon=True)
        handle.thread = t
        t.start()
        return handle

    def _worker(self, run_id: str, spec: JobSpec, dry_run: bool) -> None:
        # 串行闸门：等前面的作业结束
        while True:
            with self._lock:
                if self._running is None or self._max_parallel > 1:
                    self._running = run_id
                    break
            time.sleep(0.2)
        handle = self.get(run_id)
        if handle is None:
            return
        handle.status = "running"
        try:
            orch = Orchestrator(
                spec,
                settings=self.settings,
                events=handle.bus,
                cancel_event=handle.cancel,
                dry_run=dry_run,
            )
            handle.job_id = orch.job_id
            result = orch.run()
            handle.result = result
            handle.status = {"done": "done", "failed": "failed", "cancelled": "cancelled"}.get(result.get("status", ""), "done")
        except Exception as exc:  # 兜底，避免线程静默死掉
            handle.error = f"{type(exc).__name__}: {exc}"
            handle.status = "failed"
            handle.bus.error(f"作业异常终止：{handle.error}", stage="orchestrator", job_id=handle.job_id)
        finally:
            handle.finished_at = time.time()
            with self._lock:
                if self._running == run_id:
                    self._running = None

    # ------------------------------------------------------------------ 查询
    def get(self, run_id: str) -> JobHandle | None:
        with self._lock:
            return self._handles.get(run_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            handles = list(self._handles.values())
        return [
            {
                "run_id": h.run_id,
                "job_id": h.job_id,
                "status": h.status,
                "title": h.spec.get("title", ""),
                "created_at": h.created_at,
                "finished_at": h.finished_at,
                "error": h.error,
                "stages": h.result.get("stages", {}),
            }
            for h in sorted(handles, key=lambda x: x.created_at, reverse=True)
        ]

    def events(self, run_id: str, since: int = 0) -> list[dict[str, Any]]:
        handle = self.get(run_id)
        if handle is None:
            return []
        from dataclasses import asdict

        return [asdict(e) for e in handle.bus.history(since)]

    def cancel(self, run_id: str) -> bool:
        handle = self.get(run_id)
        if handle is None:
            return False
        handle.cancel.set()
        handle.bus.warn("收到取消请求，正在安全停止（已完成的阶段产物会保留）", stage="orchestrator", job_id=handle.job_id)
        return True
