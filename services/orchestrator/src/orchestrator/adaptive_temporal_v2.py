import asyncio
from dataclasses import dataclass
from typing import Any

from .adaptive_temporal import AdaptiveCadenceDecision
from .domain import FrameInput
from .sort_temporal import KalmanTrackedObject
from .temporal import TemporalContext, TemporalContextStore, TemporalCoordinator, TemporalDecision


@dataclass(frozen=True, slots=True)
class DetectorAssociationStatistics:
    previous_tracks: int = 0
    matched_previous_tracks: int = 0
    new_tracks: int = 0

    @property
    def association_ratio(self) -> float:
        if self.previous_tracks == 0:
            return 1.0

        return self.matched_previous_tracks / self.previous_tracks


class AdaptiveCadenceV2Policy:
    def __init__(
        self,
        max_anchor_age_ms: int,
        max_position_variance: float,
        min_association_ratio: float,
    ) -> None:
        self._max_anchor_age_ms = max_anchor_age_ms
        self._max_position_variance = max_position_variance
        self._min_association_ratio = min_association_ratio

    def decide(
        self,
        frame: FrameInput,
        context: TemporalContext,
        association: DetectorAssociationStatistics,
    ) -> AdaptiveCadenceDecision:
        states = tuple(self._kalman_state(item) for item in context.objects)

        if association.new_tracks or any(state.hits < 2 for state in states):
            return AdaptiveCadenceDecision(True, "track_confirmation_needed")
        if association.association_ratio < self._min_association_ratio:
            return AdaptiveCadenceDecision(True, "weak_association")
        if frame.captured_at_ms - context.anchor_captured_at_ms >= self._max_anchor_age_ms:
            return AdaptiveCadenceDecision(True, "stale_anchor")
        if any(self._position_variance(state) > self._max_position_variance for state in states):
            return AdaptiveCadenceDecision(True, "high_covariance")

        return AdaptiveCadenceDecision(False, "quality_sufficient")

    @staticmethod
    def _kalman_state(item: object) -> Any:
        if not isinstance(item, KalmanTrackedObject):
            raise TypeError("adaptive cadence requires Kalman tracking context")

        return item.kalman_state

    @staticmethod
    def _position_variance(state: Any) -> float:
        return max(dimension[0] for dimension in state.covariance)


class AdaptiveTemporalV2Coordinator:
    def __init__(
        self,
        detector_every_n: int,
        store: TemporalContextStore,
        coordinator: TemporalCoordinator,
        policy: AdaptiveCadenceV2Policy,
    ) -> None:
        self._detector_every_n = detector_every_n
        self._store = store
        self._coordinator = coordinator
        self._policy = policy
        self._association_by_stream: dict[str, DetectorAssociationStatistics] = {}
        self._association_lock = asyncio.Lock()

    async def estimate(self, frame: FrameInput) -> TemporalDecision:
        if frame.frame_id % self._detector_every_n == 0:
            return await self._coordinator.estimate(frame)

        context = await self._store.get(frame.stream_id)
        if context is None:
            return await self._coordinator.estimate(frame)

        async with self._association_lock:
            association = self._association_by_stream.get(
                frame.stream_id,
                DetectorAssociationStatistics(),
            )

        decision = self._policy.decide(frame, context, association)
        if decision.use_detector:
            return TemporalDecision(None, decision.reason)

        return await self._coordinator.estimate(frame)

    async def record_detector_result(
        self, frame: FrameInput, prediction: dict[str, Any]
    ) -> TemporalContext:
        previous = await self._store.get(frame.stream_id)
        context = await self._coordinator.record_detector_result(frame, prediction)
        association = self.measure_association(previous, context)

        async with self._association_lock:
            self._association_by_stream[frame.stream_id] = association

        return context

    async def reset(self, stream_id: str) -> None:
        await self._coordinator.reset(stream_id)

        async with self._association_lock:
            self._association_by_stream.pop(stream_id, None)

    @staticmethod
    def measure_association(
        previous: TemporalContext | None,
        current: TemporalContext,
    ) -> DetectorAssociationStatistics:
        previous_ids = {item.track_id for item in previous.objects} if previous else set()
        current_by_id = {item.track_id: item for item in current.objects}
        matched_previous = sum(
            1
            for track_id in previous_ids
            if track_id in current_by_id
            and AdaptiveCadenceV2Policy._kalman_state(current_by_id[track_id]).missed_updates == 0
        )

        return DetectorAssociationStatistics(
            previous_tracks=len(previous_ids),
            matched_previous_tracks=matched_previous,
            new_tracks=len(set(current_by_id) - previous_ids),
        )
