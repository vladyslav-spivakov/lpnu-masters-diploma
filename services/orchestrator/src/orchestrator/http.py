import math
import re
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from .config import MAX_FILE_SIZE_BYTES, SUPPORTED_MEDIA_TYPES, Settings
from .domain import ApplicationResponse, FrameInput, ImageInput
from .prediction import error_response


STREAM_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def to_json_response(response: ApplicationResponse) -> JSONResponse:
    return JSONResponse(status_code=response.status_code, content=response.payload)


async def _close_uploads(items: list[tuple[str, Any]]) -> None:
    for _, value in items:
        if isinstance(value, UploadFile):
            await value.close()


async def read_image(request: Request) -> ImageInput | ApplicationResponse:
    media_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if media_type != "multipart/form-data":
        return error_response(
            415, "unsupported_media_type", "Content-Type must be multipart/form-data."
        )
    try:
        form = await request.form()
    except Exception:
        return error_response(400, "invalid_request", "The multipart request is malformed.")

    items = list(form.multi_items())
    if len(items) != 1 or items[0][0] != "image" or not isinstance(items[0][1], UploadFile):
        await _close_uploads(items)
        return error_response(
            400, "invalid_request", "Exactly one image file field is required."
        )

    upload = items[0][1]
    content_type = (upload.content_type or "").lower()
    if content_type not in SUPPORTED_MEDIA_TYPES:
        await upload.close()
        return error_response(
            415, "unsupported_media_type", "Only JPEG, PNG, and WebP are supported."
        )
    try:
        content = await upload.read(MAX_FILE_SIZE_BYTES + 1)
    finally:
        await upload.close()
    if len(content) > MAX_FILE_SIZE_BYTES:
        return error_response(
            413,
            "payload_too_large",
            f"Image exceeds {MAX_FILE_SIZE_BYTES} bytes.",
        )
    return ImageInput(upload.filename or "image", content_type, content)


def read_frame(
    request: Request,
    settings: Settings,
    *,
    stream_id: str | None,
    frame_id: str | None,
    captured_at_ms: str | None,
    target_fps: str | None,
    max_latency_ms: str | None,
) -> FrameInput | ApplicationResponse:
    metadata_values = {
        name: request.query_params.getlist(name)
        for name in ("stream_id", "frame_id", "captured_at_ms")
    }
    if (
        any(len(values) != 1 for values in metadata_values.values())
        or stream_id is None
        or not STREAM_ID_PATTERN.fullmatch(stream_id)
        or frame_id is None
        or not frame_id.isdigit()
        or captured_at_ms is None
        or not captured_at_ms.isdigit()
    ):
        return error_response(
            400,
            "invalid_frame_metadata",
            "stream_id, non-negative frame_id, and captured_at_ms are required.",
        )

    constraint_values = {
        name: request.query_params.getlist(name)
        for name in ("target_fps", "max_latency_ms")
    }
    if any(len(values) > 1 for values in constraint_values.values()):
        return error_response(
            400, "invalid_frame_constraints", "Frame constraints must be singular."
        )
    try:
        effective_target_fps = (
            settings.frame_constraints.target_fps
            if target_fps is None
            else float(target_fps)
        )
        effective_max_latency_ms = (
            settings.frame_constraints.max_latency_ms
            if max_latency_ms is None
            else int(max_latency_ms)
        )
    except ValueError:
        return error_response(
            400, "invalid_frame_constraints", "Frame constraints are invalid."
        )
    if (
        not math.isfinite(effective_target_fps)
        or effective_target_fps <= 0
        or effective_max_latency_ms <= 0
        or (max_latency_ms is not None and not max_latency_ms.isdigit())
    ):
        return error_response(
            400, "invalid_frame_constraints", "Frame constraints are invalid."
        )
    return FrameInput(
        stream_id=stream_id,
        frame_id=int(frame_id),
        captured_at_ms=int(captured_at_ms),
        target_fps=effective_target_fps,
        max_latency_ms=effective_max_latency_ms,
    )
