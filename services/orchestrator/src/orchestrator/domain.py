import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ImageInput:
    filename: str
    content_type: str
    content: bytes


@dataclass(frozen=True, slots=True)
class FrameInput:
    stream_id: str
    frame_id: int
    captured_at_ms: int
    target_fps: float
    max_latency_ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "frame_id": self.frame_id,
            "captured_at_ms": self.captured_at_ms,
            "constraints": {
                "target_fps": self.target_fps,
                "max_latency_ms": self.max_latency_ms,
            },
        }


@dataclass(frozen=True, slots=True)
class ApplicationResponse:
    status_code: int
    payload: dict[str, Any]

    @property
    def successful(self) -> bool:
        return 200 <= self.status_code < 300


@dataclass(slots=True)
class FrameJob:
    image: ImageInput
    model_id: str | None
    frame: FrameInput
    enqueued_at: float
    result: asyncio.Future[ApplicationResponse]
