from dataclasses import dataclass, field
from os import getenv
from pathlib import Path
from tempfile import gettempdir


API_VERSION = "1.0.0"
SUPPORTED_MEDIA_TYPES = ("image/jpeg", "image/png", "image/webp")
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
SUPPORTED_DEVICES = ("cpu", "cuda", "auto")


def _default_weights_path() -> str:
    cache_root = getenv("LOCALAPPDATA") or getenv("XDG_CACHE_HOME") or gettempdir()
    return str(Path(cache_root) / "adaptive-neural-platform" / "models" / "yolo11n.pt")


@dataclass(frozen=True, slots=True)
class Settings:
    model_name: str = "yolo11n"
    model_version: str = "yolo11n"
    model_weights: str = field(default_factory=_default_weights_path)
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.device not in SUPPORTED_DEVICES:
            supported = ", ".join(SUPPORTED_DEVICES)
            raise ValueError(f"YOLO_DEVICE must be one of: {supported}")

    @classmethod
    def from_environment(cls) -> "Settings":
        defaults = cls()
        return cls(
            model_name=getenv("YOLO_MODEL_NAME", defaults.model_name),
            model_version=getenv("YOLO_MODEL_VERSION", defaults.model_version),
            model_weights=getenv("YOLO_MODEL_WEIGHTS", defaults.model_weights),
            device=getenv("YOLO_DEVICE", defaults.device).strip().lower(),
        )
