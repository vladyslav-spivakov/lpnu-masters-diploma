import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from os import getenv
from types import MappingProxyType
from urllib.parse import urlsplit

SUPPORTED_MEDIA_TYPES = ("image/jpeg", "image/png", "image/webp")
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_TARGET_FPS = 1.0
DEFAULT_MAX_LATENCY_MS = 1000
DEFAULT_FRAME_QUEUE_CAPACITY = 8
DEFAULT_FRAME_SAMPLE_EVERY_N = 1
DEFAULT_TEMPORAL_ENABLED = False
DEFAULT_TEMPORAL_DETECTOR_EVERY_N = 5
DEFAULT_TEMPORAL_CONTEXT_MAX_AGE_MS = 2000
DEFAULT_TEMPORAL_ESTIMATOR = "hold-last"
DEFAULT_ADAPTIVE_TEMPORAL_ENABLED = False
DEFAULT_ADAPTIVE_MAX_ANCHOR_AGE_MS = 250
DEFAULT_ADAPTIVE_MAX_POSITION_VARIANCE = 100.0
DEFAULT_ADAPTIVE_MIN_MATCH_RATIO = 0.75
TEMPORAL_ESTIMATORS = ("hold-last", "linear-motion", "kalman-sort")
MODEL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


def _normalize_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"Invalid model container URL: {value!r}")
    return base_url


@dataclass(frozen=True, slots=True)
class ModelContainerEndpoint:
    model_id: str
    base_url: str

    def __post_init__(self) -> None:
        if not MODEL_ID_PATTERN.fullmatch(self.model_id):
            raise ValueError(f"Invalid model id: {self.model_id!r}")
        object.__setattr__(self, "base_url", _normalize_base_url(self.base_url))


@dataclass(frozen=True, slots=True)
class ModelContainerRegistry:
    containers: tuple[ModelContainerEndpoint, ...]
    default_model_id: str

    def __post_init__(self) -> None:
        if not self.containers:
            raise ValueError("At least one model container must be configured")
        model_ids = [container.model_id for container in self.containers]
        base_urls = [container.base_url for container in self.containers]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("Model container ids must be unique")
        if len(base_urls) != len(set(base_urls)):
            raise ValueError("Model container URLs must be unique")
        if self.default_model_id not in model_ids:
            raise ValueError(
                f"DEFAULT_MODEL_ID {self.default_model_id!r} is not present in MODEL_CONTAINERS"
            )

    @property
    def by_id(self) -> Mapping[str, ModelContainerEndpoint]:
        return MappingProxyType({container.model_id: container for container in self.containers})

    @property
    def default(self) -> ModelContainerEndpoint:
        return self.by_id[self.default_model_id]


DEFAULT_MODEL_REGISTRY = ModelContainerRegistry(
    containers=(ModelContainerEndpoint("yolo", "http://127.0.0.1:8001"),),
    default_model_id="yolo",
)


def parse_model_registry(raw_containers: str, default_model_id: str) -> ModelContainerRegistry:
    if not raw_containers.strip():
        raise ValueError("MODEL_CONTAINERS must not be empty")

    endpoints: list[ModelContainerEndpoint] = []
    for raw_entry in raw_containers.split(","):
        entry = raw_entry.strip()
        if not entry or "=" not in entry:
            raise ValueError("MODEL_CONTAINERS entries must use model_id=http://host:port")
        model_id, base_url = (part.strip() for part in entry.split("=", 1))
        if not model_id or not base_url:
            raise ValueError("MODEL_CONTAINERS ids and URLs must not be empty")
        endpoints.append(ModelContainerEndpoint(model_id=model_id, base_url=base_url))

    return ModelContainerRegistry(
        containers=tuple(endpoints),
        default_model_id=default_model_id.strip(),
    )


def _positive_float_from_environment(name: str, default: float) -> float:
    raw_value = getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _positive_int_from_environment(name: str, default: int) -> int:
    raw_value = getenv(name)
    if raw_value is None:
        return default
    if not raw_value.isdigit() or int(raw_value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(raw_value)


def _boolean_from_environment(name: str, default: bool) -> bool:
    raw_value = getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return normalized == "true"


@dataclass(frozen=True, slots=True)
class FrameConstraints:
    target_fps: float = DEFAULT_TARGET_FPS
    max_latency_ms: int = DEFAULT_MAX_LATENCY_MS

    def __post_init__(self) -> None:
        if not isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("target_fps must be a positive finite number")
        if (
            not isinstance(self.max_latency_ms, int)
            or isinstance(self.max_latency_ms, bool)
            or self.max_latency_ms <= 0
        ):
            raise ValueError("max_latency_ms must be a positive integer")


@dataclass(frozen=True, slots=True)
class Settings:
    model_registry: ModelContainerRegistry = DEFAULT_MODEL_REGISTRY
    model_request_timeout_seconds: float = DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS
    frame_constraints: FrameConstraints = FrameConstraints()
    frame_queue_capacity: int = DEFAULT_FRAME_QUEUE_CAPACITY
    frame_sample_every_n: int = DEFAULT_FRAME_SAMPLE_EVERY_N
    temporal_enabled: bool = DEFAULT_TEMPORAL_ENABLED
    temporal_detector_every_n: int = DEFAULT_TEMPORAL_DETECTOR_EVERY_N
    temporal_context_max_age_ms: int = DEFAULT_TEMPORAL_CONTEXT_MAX_AGE_MS
    temporal_estimator: str = DEFAULT_TEMPORAL_ESTIMATOR
    adaptive_temporal_enabled: bool = DEFAULT_ADAPTIVE_TEMPORAL_ENABLED
    adaptive_max_anchor_age_ms: int = DEFAULT_ADAPTIVE_MAX_ANCHOR_AGE_MS
    adaptive_max_position_variance: float = DEFAULT_ADAPTIVE_MAX_POSITION_VARIANCE
    adaptive_min_match_ratio: float = DEFAULT_ADAPTIVE_MIN_MATCH_RATIO
    output_cadence_enabled: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("frame_queue_capacity", self.frame_queue_capacity),
            ("frame_sample_every_n", self.frame_sample_every_n),
            ("temporal_detector_every_n", self.temporal_detector_every_n),
            ("temporal_context_max_age_ms", self.temporal_context_max_age_ms),
            ("adaptive_max_anchor_age_ms", self.adaptive_max_anchor_age_ms),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.temporal_enabled, bool):
            raise ValueError("temporal_enabled must be a boolean")
        if not isinstance(self.output_cadence_enabled, bool):
            raise ValueError("output_cadence_enabled must be a boolean")
        if not isinstance(self.adaptive_temporal_enabled, bool):
            raise ValueError("adaptive_temporal_enabled must be a boolean")
        if self.adaptive_temporal_enabled and (
            not self.temporal_enabled or self.temporal_estimator != "kalman-sort"
        ):
            raise ValueError(
                "adaptive temporal cadence requires enabled kalman-sort estimation"
            )
        if (
            isinstance(self.adaptive_min_match_ratio, bool)
            or not isfinite(self.adaptive_min_match_ratio)
            or not 0 <= self.adaptive_min_match_ratio <= 1
        ):
            raise ValueError("adaptive_min_match_ratio must be between 0 and 1")
        if (
            isinstance(self.adaptive_max_position_variance, bool)
            or not isfinite(self.adaptive_max_position_variance)
            or self.adaptive_max_position_variance <= 0
        ):
            raise ValueError("adaptive_max_position_variance must be a positive number")
        if self.temporal_estimator not in TEMPORAL_ESTIMATORS:
            raise ValueError(
                f"temporal_estimator must be one of {', '.join(TEMPORAL_ESTIMATORS)}"
            )

    @classmethod
    def from_environment(cls) -> "Settings":
        defaults = cls()
        default_registry = defaults.model_registry
        default_containers = ",".join(
            f"{container.model_id}={container.base_url}"
            for container in default_registry.containers
        )
        return cls(
            model_registry=parse_model_registry(
                getenv("MODEL_CONTAINERS", default_containers),
                getenv("DEFAULT_MODEL_ID", default_registry.default_model_id),
            ),
            model_request_timeout_seconds=_positive_float_from_environment(
                "MODEL_REQUEST_TIMEOUT_SECONDS",
                defaults.model_request_timeout_seconds,
            ),
            frame_constraints=FrameConstraints(
                target_fps=_positive_float_from_environment(
                    "TARGET_FPS", defaults.frame_constraints.target_fps
                ),
                max_latency_ms=_positive_int_from_environment(
                    "MAX_LATENCY_MS", defaults.frame_constraints.max_latency_ms
                ),
            ),
            frame_queue_capacity=_positive_int_from_environment(
                "FRAME_QUEUE_CAPACITY", defaults.frame_queue_capacity
            ),
            frame_sample_every_n=_positive_int_from_environment(
                "FRAME_SAMPLE_EVERY_N", defaults.frame_sample_every_n
            ),
            output_cadence_enabled=_boolean_from_environment(
                "OUTPUT_CADENCE_ENABLED", defaults.output_cadence_enabled
            ),
            temporal_enabled=_boolean_from_environment(
                "TEMPORAL_ENABLED", defaults.temporal_enabled
            ),
            temporal_detector_every_n=_positive_int_from_environment(
                "TEMPORAL_DETECTOR_EVERY_N", defaults.temporal_detector_every_n
            ),
            temporal_context_max_age_ms=_positive_int_from_environment(
                "TEMPORAL_CONTEXT_MAX_AGE_MS", defaults.temporal_context_max_age_ms
            ),
            temporal_estimator=getenv(
                "TEMPORAL_ESTIMATOR", defaults.temporal_estimator
            ).strip(),
            adaptive_temporal_enabled=_boolean_from_environment(
                "ADAPTIVE_TEMPORAL_ENABLED", defaults.adaptive_temporal_enabled
            ),
            adaptive_max_anchor_age_ms=_positive_int_from_environment(
                "ADAPTIVE_MAX_ANCHOR_AGE_MS", defaults.adaptive_max_anchor_age_ms
            ),
            adaptive_max_position_variance=_positive_float_from_environment(
                "ADAPTIVE_MAX_POSITION_VARIANCE",
                defaults.adaptive_max_position_variance,
            ),
            adaptive_min_match_ratio=float(
                getenv(
                    "ADAPTIVE_MIN_MATCH_RATIO",
                    str(defaults.adaptive_min_match_ratio),
                )
            ),
        )
