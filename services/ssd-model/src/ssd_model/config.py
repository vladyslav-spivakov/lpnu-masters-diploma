from dataclasses import dataclass
from math import isfinite
from os import getenv


API_VERSION = "1.0.0"
SUPPORTED_MEDIA_TYPES = ("image/jpeg", "image/png", "image/webp")
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


def _score_threshold() -> float:
    raw_value = getenv("SSD_SCORE_THRESHOLD", "0.5")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError("SSD_SCORE_THRESHOLD must be a number within [0, 1]") from exc
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("SSD_SCORE_THRESHOLD must be a number within [0, 1]")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    model_name: str = "ssdlite320-mobilenet-v3-large"
    model_version: str = "torchvision-coco-v1"
    score_threshold: float = 0.5

    @classmethod
    def from_environment(cls) -> "Settings":
        defaults = cls()
        return cls(
            model_name=getenv("SSD_MODEL_NAME", defaults.model_name),
            model_version=getenv("SSD_MODEL_VERSION", defaults.model_version),
            score_threshold=_score_threshold(),
        )
