from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .domain import FrameInput
from .sort_tracker import Box, KalmanTrack, Observation, SortTracker
from .temporal import TemporalContext, TemporalEstimate, TrackedObject


@dataclass(frozen=True, slots=True)
class KalmanTrackedObject:
    track_id: str
    label: str
    bbox: Box
    confidence: float
    observed_at_ms: int
    velocity: Box
    kalman_state: KalmanTrack

    def as_prediction(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
        }


class KalmanSortEstimator:
    def __init__(self, tracker: SortTracker | None = None) -> None:
        self._tracker = tracker or SortTracker()

    def estimate(self, frame: FrameInput, context: TemporalContext) -> TemporalEstimate:
        objects = tuple(
            self._predict_object(item, frame.captured_at_ms, context.image)
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
                "model": "temporal-kalman-sort",
                "model_version": "1",
                "image": dict(context.image),
                "predictions": [item.as_prediction() for item in objects],
                "inference_time_ms": 0,
                "processing_time_ms": 0,
            },
        )

    def _predict_object(
        self,
        item: TrackedObject | KalmanTrackedObject,
        captured_at_ms: int,
        image: dict[str, int],
    ) -> KalmanTrackedObject:
        if not isinstance(item, KalmanTrackedObject):
            raise TypeError("kalman-sort context contains a non-Kalman track")
        state = self._tracker.predict(item.kalman_state, captured_at_ms)
        state = replace(state, bbox=_bounded_box(state.bbox, image))
        return _tracked_object(state, item.observed_at_ms)


class KalmanSortAnchorTracker:
    def __init__(self, tracker: SortTracker | None = None) -> None:
        self._tracker = tracker or SortTracker()

    def track(
        self,
        frame: FrameInput,
        prediction: dict[str, Any],
        previous: TemporalContext | None,
        generation: int,
    ) -> tuple[KalmanTrackedObject, ...]:
        tracks = (
            tuple(
                item.kalman_state
                for item in previous.objects
                if isinstance(item, KalmanTrackedObject)
            )
            if previous is not None
            else ()
        )
        observations = tuple(
            Observation(
                label=item["label"],
                bbox=tuple(float(value) for value in item["bbox"]),
                confidence=float(item["confidence"]),
            )
            for item in prediction.get("predictions", ())
        )
        states = self._tracker.update(
            tracks, observations, frame.captured_at_ms, generation
        )
        return tuple(_tracked_object(state, frame.captured_at_ms) for state in states)


def _tracked_object(state: KalmanTrack, observed_at_ms: int) -> KalmanTrackedObject:
    return KalmanTrackedObject(
        track_id=state.track_id,
        label=state.label,
        bbox=state.bbox,
        confidence=state.confidence,
        observed_at_ms=observed_at_ms,
        velocity=state.velocity,
        kalman_state=state,
    )


def _bounded_box(box: Box, image: dict[str, int]) -> Box:
    x, y, width, height = box
    bounded_width = min(width, float(image["width"]))
    bounded_height = min(height, float(image["height"]))
    bounded_x = min(max(0.0, x), image["width"] - bounded_width)
    bounded_y = min(max(0.0, y), image["height"] - bounded_height)
    return bounded_x, bounded_y, bounded_width, bounded_height
