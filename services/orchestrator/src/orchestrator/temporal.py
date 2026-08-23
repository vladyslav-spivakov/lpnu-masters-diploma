import asyncio
from dataclasses import dataclass
from itertools import permutations
from math import hypot, isfinite
from typing import Any, Protocol

from .domain import FrameInput


@dataclass(frozen=True, slots=True)
class TrackedObject:
    track_id: str
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    observed_at_ms: int
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def as_prediction(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
        }


@dataclass(frozen=True, slots=True)
class TemporalContext:
    stream_id: str
    generation: int
    anchor_frame_id: int
    anchor_captured_at_ms: int
    updated_at_ms: int
    expires_at_ms: int
    objects: tuple[TrackedObject, ...]
    image: dict[str, int]


@dataclass(frozen=True, slots=True)
class TemporalEstimate:
    context: TemporalContext
    prediction: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TemporalDecision:
    estimate: TemporalEstimate | None
    reason: str


class TemporalEstimator(Protocol):
    def estimate(self, frame: FrameInput, context: TemporalContext) -> TemporalEstimate: ...


class HoldLastEstimator:
    def estimate(self, frame: FrameInput, context: TemporalContext) -> TemporalEstimate:
        objects = tuple(
            TrackedObject(
                track_id=item.track_id,
                label=item.label,
                bbox=item.bbox,
                confidence=item.confidence,
                observed_at_ms=item.observed_at_ms,
                velocity=item.velocity,
            )
            for item in context.objects
        )
        next_context = TemporalContext(
            stream_id=context.stream_id,
            generation=context.generation,
            anchor_frame_id=context.anchor_frame_id,
            anchor_captured_at_ms=context.anchor_captured_at_ms,
            updated_at_ms=frame.captured_at_ms,
            expires_at_ms=context.expires_at_ms,
            objects=objects,
            image=context.image,
        )
        return TemporalEstimate(
            context=next_context,
            prediction={
                "model": "temporal-hold-last",
                "model_version": "1",
                "image": dict(context.image),
                "predictions": [item.as_prediction() for item in objects],
                "inference_time_ms": 0,
                "processing_time_ms": 0,
            },
        )


class LinearMotionEstimator:
    def estimate(self, frame: FrameInput, context: TemporalContext) -> TemporalEstimate:
        elapsed_ms = frame.captured_at_ms - context.anchor_captured_at_ms
        objects = tuple(
            self._project(item, elapsed_ms, context.image) for item in context.objects
        )
        next_context = TemporalContext(
            stream_id=context.stream_id,
            generation=context.generation,
            anchor_frame_id=context.anchor_frame_id,
            anchor_captured_at_ms=context.anchor_captured_at_ms,
            updated_at_ms=frame.captured_at_ms,
            expires_at_ms=context.expires_at_ms,
            objects=context.objects,
            image=context.image,
        )
        return TemporalEstimate(
            context=next_context,
            prediction={
                "model": "temporal-linear-motion",
                "model_version": "1",
                "image": dict(context.image),
                "predictions": [item.as_prediction() for item in objects],
                "inference_time_ms": 0,
                "processing_time_ms": 0,
            },
        )

    @staticmethod
    def _project(
        item: TrackedObject, elapsed_ms: int, image: dict[str, int]
    ) -> TrackedObject:
        projected = tuple(
            value + velocity * elapsed_ms
            for value, velocity in zip(item.bbox, item.velocity, strict=True)
        )
        if not all(isfinite(value) for value in projected):
            raise ValueError("motion projection must be finite")
        x, y, width, height = projected
        if width <= 0 or height <= 0:
            raise ValueError("motion projection must preserve positive box dimensions")
        bounded_width = min(width, float(image["width"]))
        bounded_height = min(height, float(image["height"]))
        bounded_x = min(max(0.0, x), image["width"] - bounded_width)
        bounded_y = min(max(0.0, y), image["height"] - bounded_height)
        return TrackedObject(
            track_id=item.track_id,
            label=item.label,
            bbox=(bounded_x, bounded_y, bounded_width, bounded_height),
            confidence=item.confidence,
            observed_at_ms=item.observed_at_ms,
            velocity=item.velocity,
        )


def create_temporal_estimator(name: str) -> TemporalEstimator:
    if name == "hold-last":
        return HoldLastEstimator()
    if name == "linear-motion":
        return LinearMotionEstimator()
    raise ValueError(f"Unsupported temporal estimator: {name}")


class DetectorAnchorTracker:
    def track(
        self,
        frame: FrameInput,
        prediction: dict[str, Any],
        previous: TemporalContext | None,
        generation: int,
    ) -> tuple[TrackedObject, ...]:
        previous_objects = list(previous.objects) if previous is not None else []
        elapsed_ms = (
            frame.captured_at_ms - previous.anchor_captured_at_ms
            if previous is not None
            else 0
        )
        detected = list(prediction.get("predictions", ()))
        matches = self._match(detected, previous_objects, prediction["image"])
        return tuple(
            self._tracked_object(
                item,
                index,
                matches.get(index),
                generation,
                frame.captured_at_ms,
                elapsed_ms,
            )
            for index, item in enumerate(detected)
        )

    @staticmethod
    def _tracked_object(
        item: dict[str, Any],
        index: int,
        matched: TrackedObject | None,
        generation: int,
        observed_at_ms: int,
        elapsed_ms: int,
    ) -> TrackedObject:
        bbox = tuple(float(value) for value in item["bbox"])
        return TrackedObject(
            track_id=matched.track_id if matched else f"{generation}:{index}",
            label=item["label"],
            bbox=bbox,
            confidence=float(item["confidence"]),
            observed_at_ms=observed_at_ms,
            velocity=DetectorAnchorTracker._velocity(matched, bbox, elapsed_ms),
        )

    @staticmethod
    def _match(
        detected: list[dict[str, Any]],
        previous: list[TrackedObject],
        image: dict[str, int],
    ) -> dict[int, TrackedObject]:
        diagonal = hypot(image["width"], image["height"])
        matches: dict[int, TrackedObject] = {}
        for label in {item["label"] for item in detected}:
            current_indexes = [
                index for index, item in enumerate(detected) if item["label"] == label
            ]
            previous_objects = [item for item in previous if item.label == label]
            for current_index, previous_index in DetectorAnchorTracker._minimum_assignment(
                current_indexes, detected, previous_objects
            ):
                current_bbox = detected[current_index]["bbox"]
                previous_object = previous_objects[previous_index]
                if DetectorAnchorTracker._center_distance(
                    current_bbox, previous_object.bbox
                ) <= diagonal * 0.25:
                    matches[current_index] = previous_object
        return matches

    @staticmethod
    def _minimum_assignment(
        current_indexes: list[int],
        detected: list[dict[str, Any]],
        previous: list[TrackedObject],
    ) -> tuple[tuple[int, int], ...]:
        if not current_indexes or not previous:
            return ()
        if max(len(current_indexes), len(previous)) > 8:
            return DetectorAnchorTracker._greedy_assignment(
                current_indexes, detected, previous
            )
        if len(current_indexes) <= len(previous):
            candidates = (
                tuple(zip(current_indexes, assignment, strict=True))
                for assignment in permutations(range(len(previous)), len(current_indexes))
            )
        else:
            candidates = (
                tuple(zip(assignment, range(len(previous)), strict=True))
                for assignment in permutations(current_indexes, len(previous))
            )
        return min(
            candidates,
            key=lambda assignment: DetectorAnchorTracker._assignment_cost(
                assignment, detected, previous
            ),
        )

    @staticmethod
    def _greedy_assignment(
        current_indexes: list[int],
        detected: list[dict[str, Any]],
        previous: list[TrackedObject],
    ) -> tuple[tuple[int, int], ...]:
        pairs = sorted(
            (
                DetectorAnchorTracker._center_distance(
                    detected[current_index]["bbox"], previous_object.bbox
                ),
                current_index,
                previous_index,
            )
            for current_index in current_indexes
            for previous_index, previous_object in enumerate(previous)
        )
        selected = []
        used_current: set[int] = set()
        used_previous: set[int] = set()
        for _, current_index, previous_index in pairs:
            if current_index not in used_current and previous_index not in used_previous:
                selected.append((current_index, previous_index))
                used_current.add(current_index)
                used_previous.add(previous_index)
        return tuple(selected)

    @staticmethod
    def _assignment_cost(
        assignment: tuple[tuple[int, int], ...],
        detected: list[dict[str, Any]],
        previous: list[TrackedObject],
    ) -> float:
        return sum(
            DetectorAnchorTracker._center_distance(
                detected[current_index]["bbox"], previous[previous_index].bbox
            )
            for current_index, previous_index in assignment
        )

    @staticmethod
    def _center_distance(
        left: tuple[float, float, float, float] | list[float],
        right: tuple[float, float, float, float] | list[float],
    ) -> float:
        return hypot(
            left[0] + left[2] / 2 - (right[0] + right[2] / 2),
            left[1] + left[3] / 2 - (right[1] + right[3] / 2),
        )

    @staticmethod
    def _velocity(
        previous: TrackedObject | None,
        bbox: tuple[float, float, float, float],
        elapsed_ms: int,
    ) -> tuple[float, float, float, float]:
        if previous is None or elapsed_ms <= 0:
            return (0.0, 0.0, 0.0, 0.0)
        return tuple(
            (current - old) / elapsed_ms
            for current, old in zip(bbox, previous.bbox, strict=True)
        )


class TemporalContextStore:
    def __init__(self) -> None:
        self._contexts: dict[str, TemporalContext] = {}
        self._lock = asyncio.Lock()

    async def get(self, stream_id: str) -> TemporalContext | None:
        async with self._lock:
            return self._contexts.get(stream_id)

    async def put(self, context: TemporalContext) -> None:
        async with self._lock:
            self._contexts[context.stream_id] = context

    async def reset(self, stream_id: str) -> None:
        async with self._lock:
            self._contexts.pop(stream_id, None)


class TemporalCoordinator:
    def __init__(
        self,
        detector_every_n: int,
        context_max_age_ms: int,
        store: TemporalContextStore,
        estimator: TemporalEstimator,
        anchor_tracker: DetectorAnchorTracker | None = None,
    ) -> None:
        self._detector_every_n = detector_every_n
        self._context_max_age_ms = context_max_age_ms
        self._store = store
        self._estimator = estimator
        self._anchor_tracker = anchor_tracker or DetectorAnchorTracker()

    async def estimate(self, frame: FrameInput) -> TemporalDecision:
        if frame.frame_id % self._detector_every_n == 0:
            return TemporalDecision(None, "detector_due")
        context = await self._store.get(frame.stream_id)
        invalid_reason = self._invalid_context_reason(frame, context)
        if invalid_reason is not None:
            return TemporalDecision(None, invalid_reason)
        try:
            estimate = self._estimator.estimate(frame, context)
        except Exception:
            return TemporalDecision(None, "estimator_failed")
        await self._store.put(estimate.context)
        return TemporalDecision(estimate, "context_available")

    async def record_detector_result(
        self, frame: FrameInput, prediction: dict[str, Any]
    ) -> TemporalContext:
        previous = await self._store.get(frame.stream_id)
        generation = 1 if previous is None else previous.generation + 1
        objects = self._anchor_tracker.track(frame, prediction, previous, generation)
        context = TemporalContext(
            stream_id=frame.stream_id,
            generation=generation,
            anchor_frame_id=frame.frame_id,
            anchor_captured_at_ms=frame.captured_at_ms,
            updated_at_ms=frame.captured_at_ms,
            expires_at_ms=frame.captured_at_ms + self._context_max_age_ms,
            objects=objects,
            image=dict(prediction["image"]),
        )
        await self._store.put(context)
        return context

    async def reset(self, stream_id: str) -> None:
        await self._store.reset(stream_id)

    @staticmethod
    def _invalid_context_reason(
        frame: FrameInput, context: TemporalContext | None
    ) -> str | None:
        if context is None:
            return "context_missing"
        if frame.frame_id <= context.anchor_frame_id or frame.captured_at_ms <= context.updated_at_ms:
            return "out_of_order_frame"
        if frame.captured_at_ms >= context.expires_at_ms:
            return "context_stale"
        return None
