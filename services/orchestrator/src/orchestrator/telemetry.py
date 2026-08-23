import asyncio
from collections import deque
from copy import deepcopy
import math
import time
from typing import Any, Callable, Mapping


PROCESSING_FPS_WINDOW_SECONDS = 10.0
DECISION_JOURNAL_CAPACITY = 100


class RequestCounters:
    def __init__(self) -> None:
        self._received_requests = 0
        self._completed_requests = 0
        self._lock = asyncio.Lock()

    async def record_received(self) -> None:
        async with self._lock:
            self._received_requests += 1

    async def record_completed(self) -> None:
        async with self._lock:
            self._completed_requests += 1

    async def snapshot(self) -> dict[str, int]:
        async with self._lock:
            return {
                "received_requests": self._received_requests,
                "completed_requests": self._completed_requests,
            }


class FrameCounters:
    def __init__(
        self,
        monotonic_clock: Callable[[], float] = time.monotonic,
        processing_fps_window_seconds: float = PROCESSING_FPS_WINDOW_SECONDS,
    ) -> None:
        self._received_frames = 0
        self._processed_frames = 0
        self._skipped_frames = 0
        self._dropped_frames = 0
        self._clock = monotonic_clock
        self._window_seconds = processing_fps_window_seconds
        self._completion_timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    def _discard_stale_completions(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._completion_timestamps and self._completion_timestamps[0] < cutoff:
            self._completion_timestamps.popleft()

    def _current_processing_fps(self) -> float:
        if len(self._completion_timestamps) < 2:
            return 0.0
        elapsed = self._completion_timestamps[-1] - self._completion_timestamps[0]
        if elapsed <= 0:
            return 0.0
        processing_fps = (len(self._completion_timestamps) - 1) / elapsed
        return processing_fps if math.isfinite(processing_fps) else 0.0

    async def record_received(self) -> None:
        async with self._lock:
            self._received_frames += 1

    async def record_processed(self) -> tuple[float, int]:
        completed_at = self._clock()
        async with self._lock:
            self._discard_stale_completions(completed_at)
            self._processed_frames += 1
            self._completion_timestamps.append(completed_at)
            return self._current_processing_fps(), len(self._completion_timestamps)

    async def record_skipped(self) -> None:
        async with self._lock:
            self._skipped_frames += 1

    async def record_dropped(self) -> None:
        async with self._lock:
            self._dropped_frames += 1

    async def snapshot(self) -> dict[str, int | float]:
        now = self._clock()
        async with self._lock:
            self._discard_stale_completions(now)
            return {
                "received_frames": self._received_frames,
                "processed_frames": self._processed_frames,
                "skipped_frames": self._skipped_frames,
                "dropped_frames": self._dropped_frames,
                "processing_fps": self._current_processing_fps(),
            }


class DecisionJournal:
    def __init__(self, capacity: int = DECISION_JOURNAL_CAPACITY) -> None:
        if capacity <= 0:
            raise ValueError("decision journal capacity must be positive")
        self._events: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._next_sequence_id = 1
        self._lock = asyncio.Lock()

    async def record(
        self,
        frame: Mapping[str, Any],
        outcome: str,
        **details: Any,
    ) -> dict[str, Any]:
        async with self._lock:
            event = {
                "sequence_id": self._next_sequence_id,
                "outcome": outcome,
                "frame": {
                    "stream_id": frame["stream_id"],
                    "frame_id": frame["frame_id"],
                    "captured_at_ms": frame["captured_at_ms"],
                },
                "constraints": dict(frame["constraints"]),
                **details,
            }
            self._next_sequence_id += 1
            self._events.append(event)
            return deepcopy(event)

    async def snapshot(self) -> list[dict[str, Any]]:
        async with self._lock:
            return deepcopy(list(self._events))
