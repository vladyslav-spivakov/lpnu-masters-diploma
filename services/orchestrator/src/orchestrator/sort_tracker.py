from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

Box = tuple[float, float, float, float]
Covariance = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Observation:
    label: str
    bbox: Box
    confidence: float


@dataclass(frozen=True, slots=True)
class KalmanTrack:
    track_id: str
    label: str
    bbox: Box
    velocity: Box
    covariance: tuple[Covariance, Covariance, Covariance, Covariance]
    confidence: float
    updated_at_ms: int
    hits: int = 1
    missed_updates: int = 0

    @property
    def confirmed(self) -> bool:
        return self.hits >= 2


class SortTracker:
    def __init__(self, iou_threshold: float = 0.2, max_missed_updates: int = 2) -> None:
        self._iou_threshold = iou_threshold
        self._max_missed_updates = max_missed_updates

    def predict(self, track: KalmanTrack, captured_at_ms: int) -> KalmanTrack:
        elapsed_seconds = max(0, captured_at_ms - track.updated_at_ms) / 1000
        bbox: list[float] = []
        covariance: list[Covariance] = []
        for value, velocity, uncertainty in zip(
            track.bbox, track.velocity, track.covariance, strict=True
        ):
            projected, projected_uncertainty = _predict_dimension(
                value, velocity, uncertainty, elapsed_seconds
            )
            bbox.append(projected)
            covariance.append(projected_uncertainty)
        predicted_box = _valid_projected_box(tuple(bbox))
        return replace(
            track,
            bbox=predicted_box,
            covariance=tuple(covariance),
            updated_at_ms=captured_at_ms,
        )

    def update(
        self,
        tracks: tuple[KalmanTrack, ...],
        observations: tuple[Observation, ...],
        captured_at_ms: int,
        generation: int,
    ) -> tuple[KalmanTrack, ...]:
        predicted = tuple(self.predict(track, captured_at_ms) for track in tracks)
        matched, unmatched_tracks, unmatched_observations = self._associate(
            predicted, observations
        )
        updated = [
            self._correct(predicted[track_index], observations[observation_index])
            for track_index, observation_index in matched
        ]
        updated.extend(
            replace(
                predicted[index],
                missed_updates=predicted[index].missed_updates + 1,
            )
            for index in unmatched_tracks
            if predicted[index].missed_updates < self._max_missed_updates
        )
        updated.extend(
            self._new_track(observations[index], generation, index, captured_at_ms)
            for index in unmatched_observations
        )
        return tuple(updated)

    def _associate(
        self,
        tracks: tuple[KalmanTrack, ...],
        observations: tuple[Observation, ...],
    ) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...], tuple[int, ...]]:
        if not tracks or not observations:
            return (), tuple(range(len(tracks))), tuple(range(len(observations)))
        costs = [
            [
                1 - intersection_over_union(track.bbox, observation.bbox)
                if track.label == observation.label
                else 2.0
                for observation in observations
            ]
            for track in tracks
        ]
        candidates = minimum_cost_assignment(costs)
        matched = tuple(
            (track_index, observation_index)
            for track_index, observation_index in candidates
            if costs[track_index][observation_index] <= 1 - self._iou_threshold
        )
        matched_tracks = {track_index for track_index, _ in matched}
        matched_observations = {observation_index for _, observation_index in matched}
        return (
            matched,
            tuple(index for index in range(len(tracks)) if index not in matched_tracks),
            tuple(
                index
                for index in range(len(observations))
                if index not in matched_observations
            ),
        )

    @staticmethod
    def _correct(track: KalmanTrack, observation: Observation) -> KalmanTrack:
        bbox: list[float] = []
        velocity: list[float] = []
        covariance: list[Covariance] = []
        for value, speed, uncertainty, measured in zip(
            track.bbox,
            track.velocity,
            track.covariance,
            observation.bbox,
            strict=True,
        ):
            corrected, corrected_speed, corrected_uncertainty = _update_dimension(
                value, speed, uncertainty, measured
            )
            bbox.append(corrected)
            velocity.append(corrected_speed)
            covariance.append(corrected_uncertainty)
        return replace(
            track,
            bbox=_valid_box(tuple(bbox)),
            velocity=tuple(velocity),
            covariance=tuple(covariance),
            confidence=observation.confidence,
            hits=track.hits + 1,
            missed_updates=0,
        )

    @staticmethod
    def _new_track(
        observation: Observation,
        generation: int,
        observation_index: int,
        captured_at_ms: int,
    ) -> KalmanTrack:
        initial_covariance: Covariance = (25.0, 0.0, 0.0, 100.0)
        return KalmanTrack(
            track_id=f"{generation}:{observation_index}",
            label=observation.label,
            bbox=_valid_box(observation.bbox),
            velocity=(0.0, 0.0, 0.0, 0.0),
            covariance=(initial_covariance,) * 4,
            confidence=observation.confidence,
            updated_at_ms=captured_at_ms,
        )


def _predict_dimension(
    value: float,
    velocity: float,
    covariance: Covariance,
    elapsed_seconds: float,
) -> tuple[float, Covariance]:
    p00, p01, p10, p11 = covariance
    process_noise = 1.0
    return (
        value + velocity * elapsed_seconds,
        (
            p00 + elapsed_seconds * (p10 + p01) + elapsed_seconds**2 * p11 + process_noise,
            p01 + elapsed_seconds * p11,
            p10 + elapsed_seconds * p11,
            p11 + process_noise,
        ),
    )


def _update_dimension(
    value: float,
    velocity: float,
    covariance: Covariance,
    measurement: float,
) -> tuple[float, float, Covariance]:
    p00, p01, p10, p11 = covariance
    innovation_variance = p00 + 4.0
    position_gain = p00 / innovation_variance
    velocity_gain = p10 / innovation_variance
    residual = measurement - value
    return (
        value + position_gain * residual,
        velocity + velocity_gain * residual,
        (
            (1 - position_gain) * p00,
            (1 - position_gain) * p01,
            p10 - velocity_gain * p00,
            p11 - velocity_gain * p01,
        ),
    )


def intersection_over_union(left: Box, right: Box) -> float:
    left_x, left_y, left_width, left_height = left
    right_x, right_y, right_width, right_height = right
    intersection_width = max(
        0.0, min(left_x + left_width, right_x + right_width) - max(left_x, right_x)
    )
    intersection_height = max(
        0.0, min(left_y + left_height, right_y + right_height) - max(left_y, right_y)
    )
    intersection = intersection_width * intersection_height
    union = left_width * left_height + right_width * right_height - intersection
    return intersection / union if union > 0 else 0.0


def minimum_cost_assignment(costs: list[list[float]]) -> tuple[tuple[int, int], ...]:

    if not costs or not costs[0]:
        return ()
    transposed = len(costs) > len(costs[0])
    matrix = [list(row) for row in zip(*costs, strict=True)] if transposed else costs
    row_count, column_count = len(matrix), len(matrix[0])
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    column_match = [0] * (column_count + 1)
    path = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        column_match[0] = row
        minimum = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        current_column = 0
        while True:
            used[current_column] = True
            current_row = column_match[current_column]
            delta = float("inf")
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = (
                    matrix[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    path[column] = current_column
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            for column in range(column_count + 1):
                if used[column]:
                    row_potential[column_match[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if column_match[current_column] == 0:
                break
        while True:
            previous_column = path[current_column]
            column_match[current_column] = column_match[previous_column]
            current_column = previous_column
            if current_column == 0:
                break
    assignment = tuple(
        (row - 1, column - 1)
        for column, row in enumerate(column_match[1:], start=1)
        if row != 0
    )
    return tuple((column, row) for row, column in assignment) if transposed else assignment


def _valid_box(values: tuple[float, ...]) -> Box:
    if len(values) != 4 or not all(isfinite(value) for value in values):
        raise ValueError("tracking box must contain four finite values")
    x, y, width, height = values
    if width <= 0 or height <= 0:
        raise ValueError("tracking box dimensions must be positive")
    return x, y, width, height


def _valid_projected_box(values: tuple[float, ...]) -> Box:
    if len(values) != 4 or not all(isfinite(value) for value in values):
        raise ValueError("tracking box must contain four finite values")

    x, y, width, height = values

    return x, y, max(1.0, width), max(1.0, height)
