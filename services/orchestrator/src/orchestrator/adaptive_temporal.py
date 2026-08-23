import asyncio
from dataclasses import dataclass
from typing import Any

from .domain import FrameInput
from .sort_temporal import KalmanTrackedObject
from .temporal import TemporalContext, TemporalContextStore, TemporalCoordinator, TemporalDecision


@dataclass(frozen=True, slots=True)
class AdaptiveCadenceDecision:
    use_detector: bool
    reason: str


@dataclass(frozen=True, slots=True)
class DetectorUpdateQuality:
    match_ratio: float = 1.0
    new_tracks: int = 0
    lost_tracks: int = 0


class AdaptiveCadencePolicy:
    def __init__(
        self,
        max_anchor_age_ms: int,
        max_position_variance: float,
        min_match_ratio: float,
    ) -> None:
        self._max_anchor_age_ms = max_anchor_age_ms
        self._max_position_variance = max_position_variance
        self._min_match_ratio = min_match_ratio

    def decide(
        self,
        frame: FrameInput,
        context: TemporalContext,
        quality: DetectorUpdateQuality,
    ) -> AdaptiveCadenceDecision:
        if quality.new_tracks:
            return AdaptiveCadenceDecision(True, "adaptive_new_track")
        if quality.lost_tracks:
            return AdaptiveCadenceDecision(True, "adaptive_lost_track")
        if quality.match_ratio < self._min_match_ratio:
            return AdaptiveCadenceDecision(True, "adaptive_weak_matching")
        if not context.objects:
            return AdaptiveCadenceDecision(True, "adaptive_no_tracks")

        states = tuple(self._kalman_state(item) for item in context.objects)

        if any(state.hits < 2 for state in states):
            return AdaptiveCadenceDecision(True, "adaptive_unconfirmed_track")
        if any(state.missed_updates for state in states):
            return AdaptiveCadenceDecision(True, "adaptive_missed_track")
        if frame.captured_at_ms - context.anchor_captured_at_ms >= self._max_anchor_age_ms:
            return AdaptiveCadenceDecision(True, "adaptive_stale_anchor")
        if any(self._position_variance(state) > self._max_position_variance for state in states):
            return AdaptiveCadenceDecision(True, "adaptive_high_covariance")
        return AdaptiveCadenceDecision(False, "adaptive_quality_sufficient")

    @staticmethod
    def _kalman_state(item: object) -> Any:
        if not isinstance(item, KalmanTrackedObject):
            raise TypeError("adaptive cadence requires Kalman tracking context")
        return item.kalman_state

    @staticmethod
    def _position_variance(state: Any) -> float:
        return max(dimension[0] for dimension in state.covariance)


class AdaptiveTemporalCoordinator:
    def __init__(
        self,
        detector_every_n: int,
        store: TemporalContextStore,
        coordinator: TemporalCoordinator,
        policy: AdaptiveCadencePolicy,
    ) -> None:
        self._detector_every_n = detector_every_n
        self._store = store
        self._coordinator = coordinator
        self._policy = policy
        self._quality_by_stream: dict[str, DetectorUpdateQuality] = {}
        self._quality_lock = asyncio.Lock()

    async def estimate(self, frame: FrameInput) -> TemporalDecision:
        if frame.frame_id % self._detector_every_n == 0:
            return await self._coordinator.estimate(frame)
        context = await self._store.get(frame.stream_id)
        if context is None:
            return await self._coordinator.estimate(frame)
        async with self._quality_lock:
            quality = self._quality_by_stream.get(frame.stream_id, DetectorUpdateQuality())
        decision = self._policy.decide(frame, context, quality)
        if decision.use_detector:
            return TemporalDecision(None, decision.reason)
        return await self._coordinator.estimate(frame)

    async def record_detector_result(
        self, frame: FrameInput, prediction: dict[str, Any]
    ) -> TemporalContext:
        previous = await self._store.get(frame.stream_id)
        context = await self._coordinator.record_detector_result(frame, prediction)
        quality = self._measure_update(previous, context)

        async with self._quality_lock:
            self._quality_by_stream[frame.stream_id] = quality
        return context

    async def reset(self, stream_id: str) -> None:
        await self._coordinator.reset(stream_id)
        async with self._quality_lock:
            self._quality_by_stream.pop(stream_id, None)

    @staticmethod
    def _measure_update(
        previous: TemporalContext | None, current: TemporalContext
    ) -> DetectorUpdateQuality:
        previous_ids = {item.track_id for item in previous.objects} if previous else set()
        current_ids = {item.track_id for item in current.objects}
        matched = previous_ids & current_ids

        return DetectorUpdateQuality(
            match_ratio=len(matched) / len(previous_ids) if previous_ids else 1.0,
            new_tracks=len(current_ids - previous_ids),
            lost_tracks=len(previous_ids - current_ids),
        )
