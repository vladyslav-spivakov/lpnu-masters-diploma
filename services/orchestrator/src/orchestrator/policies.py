import asyncio
import math
import time
from collections import deque
from collections.abc import Callable
from typing import Any

MODEL_LATENCY_WINDOW_SIZE = 5
PACING_EARLY_TOLERANCE_RATIO = 0.05


class FramePacer:
    def __init__(self, monotonic_clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = monotonic_clock
        self._next_deadline_by_stream: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def admit(self, stream_id: str, target_fps: float) -> tuple[bool, float]:
        interval = 1 / target_fps
        early_tolerance = interval * PACING_EARLY_TOLERANCE_RATIO
        async with self._lock:
            now = self._clock()
            deadline = self._next_deadline_by_stream.get(stream_id)
            if deadline is None:
                self._next_deadline_by_stream[stream_id] = now + interval
                return True, 0.0
            if now + early_tolerance < deadline:
                return False, deadline - now

            elapsed_after_deadline = max(0.0, now - deadline)
            elapsed_intervals = math.floor(elapsed_after_deadline / interval)
            self._next_deadline_by_stream[stream_id] = (
                deadline + (elapsed_intervals + 1) * interval
            )
            return True, 0.0


class OutputCadenceRegulator:
    def __init__(
        self,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self._clock = monotonic_clock
        self._sleep = sleep
        self._next_deadline_by_stream: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def regulate(self, stream_id: str, target_fps: float) -> dict[str, Any]:
        interval = 1 / target_fps
        async with self._lock:
            now = self._clock()
            deadline = self._next_deadline_by_stream.get(stream_id, now)
            release_at = max(now, deadline)
            self._next_deadline_by_stream[stream_id] = release_at + interval
        delay = max(0.0, release_at - self._clock())
        if delay:
            await self._sleep(delay)
        released_at = self._clock()
        lateness_ms = max(0, round((released_at - release_at) * 1000))
        return {
            "scheduled_release_ms": round(release_at * 1000),
            "release_delay_ms": round(delay * 1000),
            "deadline_missed": lateness_ms > 0,
            "deadline_lateness_ms": lateness_ms,
        }


class ModelLatencySelector:
    def __init__(
        self,
        model_ids: tuple[str, ...],
        default_model_id: str,
        window_size: int = MODEL_LATENCY_WINDOW_SIZE,
    ) -> None:
        self._model_ids = model_ids
        self._default_model_id = default_model_id
        self._observations = {
            model_id: deque(maxlen=window_size) for model_id in model_ids
        }
        self._lock = asyncio.Lock()

    def _average_latencies(self) -> dict[str, float]:
        return {
            model_id: sum(values) / len(values)
            for model_id, values in self._observations.items()
            if values
        }

    @staticmethod
    def _describe_selection(
        selected_model_id: str,
        reason: str,
        averages: dict[str, float],
        max_latency_ms: int,
    ) -> dict[str, Any]:
        observed_latency_ms = averages.get(selected_model_id)
        return {
            "selected_model_id": selected_model_id,
            "reason": reason,
            "observed_latency_ms": observed_latency_ms,
            "max_latency_ms": max_latency_ms,
            "latency_constraint_met": (
                None if observed_latency_ms is None else observed_latency_ms <= max_latency_ms
            ),
            "observed_model_ids": sorted(averages),
        }

    async def record(self, model_id: str, latency_ms: int) -> None:
        async with self._lock:
            self._observations[model_id].append(latency_ms)

    async def select(
        self,
        requested_model_id: str | None,
        max_latency_ms: int,
    ) -> dict[str, Any]:
        async with self._lock:
            averages = self._average_latencies()
            if requested_model_id is not None:
                return self._describe_selection(
                    requested_model_id, "explicit_model", averages, max_latency_ms
                )
            if len(averages) < len(self._model_ids):
                return self._describe_selection(
                    self._default_model_id,
                    "default_insufficient_observations",
                    averages,
                    max_latency_ms,
                )
            eligible = [
                model_id
                for model_id, average in averages.items()
                if average <= max_latency_ms
            ]
            candidates = eligible or list(self._model_ids)
            selected_model_id = min(
                candidates,
                key=lambda model_id: (averages[model_id], model_id),
            )
            reason = (
                "lowest_latency_within_constraint"
                if eligible
                else "lowest_latency_constraint_unmet"
            )
            return self._describe_selection(
                selected_model_id, reason, averages, max_latency_ms
            )

    async def snapshot(self) -> dict[str, dict[str, float | int]]:
        async with self._lock:
            return {
                model_id: {
                    "sample_count": len(values),
                    "average_latency_ms": sum(values) / len(values),
                }
                for model_id, values in self._observations.items()
                if values
            }


def evaluate_frame_constraints(
    *,
    processing_fps: float,
    fps_sample_count: int,
    target_fps: float,
    end_to_end_time_ms: int,
    max_latency_ms: int,
) -> dict[str, Any]:
    fps_status = (
        "insufficient_data"
        if fps_sample_count < 2
        else "met" if processing_fps >= target_fps else "violated"
    )
    latency_status = "met" if end_to_end_time_ms <= max_latency_ms else "violated"
    violations = []
    if fps_status == "violated":
        violations.append("target_fps")
    if latency_status == "violated":
        violations.append("max_latency_ms")
    overall_status = (
        "violated"
        if violations
        else "insufficient_data" if fps_status == "insufficient_data" else "met"
    )
    return {
        "status": overall_status,
        "fps": {
            "status": fps_status,
            "actual_fps": processing_fps,
            "target_fps": target_fps,
            "sample_count": fps_sample_count,
        },
        "latency": {
            "status": latency_status,
            "actual_ms": end_to_end_time_ms,
            "maximum_ms": max_latency_ms,
        },
        "violations": violations,
    }
