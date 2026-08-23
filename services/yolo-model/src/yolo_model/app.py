import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from .config import MAX_FILE_SIZE_BYTES, SUPPORTED_MEDIA_TYPES
from .schemas import ErrorBody, ErrorResponse, HealthResponse, InfoResponse, PredictionResponse
from .errors import InvalidImageError, InferenceFailedError, ModelNotReadyError
from .service import YoloModelService


LOGGER = logging.getLogger(__name__)
ERROR_RESPONSES = {
    status: {"model": ErrorResponse}
    for status in (400, 413, 415, 422, 500, 503)
}
PREDICT_REQUEST_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["image"],
                    "properties": {
                        "image": {
                            "type": "string",
                            "format": "binary",
                            "description": f"One image file, at most {MAX_FILE_SIZE_BYTES} bytes.",
                        }
                    },
                },
                "encoding": {"image": {"contentType": ", ".join(SUPPORTED_MEDIA_TYPES)}},
            }
        },
    }
}


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    payload = ErrorResponse(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


async def _close_uploads(items: list[tuple[str, Any]]) -> None:
    for _, value in items:
        if isinstance(value, UploadFile):
            await value.close()


def create_app(model_service: Any | None = None) -> FastAPI:
    service = model_service or YoloModelService()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if not service.ready:
            try:
                await run_in_threadpool(service.load)
            except Exception:
                LOGGER.exception("YOLO model failed to load; service remains unavailable")
        yield

    app = FastAPI(title="YOLO Model Service", version="1.0.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse, responses={503: {"model": ErrorResponse}})
    async def health() -> HealthResponse | JSONResponse:
        if not service.ready:
            return _error(503, "model_not_ready", "The model is not ready.")
        return HealthResponse()

    @app.get("/info", response_model=InfoResponse, responses={503: {"model": ErrorResponse}})
    async def info() -> InfoResponse | JSONResponse:
        if not service.ready:
            return _error(503, "model_not_ready", "The model is not ready.")
        return service.info()

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        responses=ERROR_RESPONSES,
        openapi_extra=PREDICT_REQUEST_BODY,
    )
    async def predict(request: Request) -> PredictionResponse | JSONResponse:
        if not service.ready:
            return _error(503, "model_not_ready", "The model is not ready.")

        request_media_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
        if request_media_type != "multipart/form-data":
            return _error(415, "unsupported_media_type", "Content-Type must be multipart/form-data.")

        try:
            form = await request.form()
        except Exception:
            return _error(400, "invalid_request", "The multipart request is malformed.")

        items = list(form.multi_items())
        if len(items) != 1 or items[0][0] != "image" or not isinstance(items[0][1], UploadFile):
            await _close_uploads(items)
            return _error(400, "invalid_request", "Exactly one image file field is required.")

        upload = items[0][1]
        if (upload.content_type or "").lower() not in SUPPORTED_MEDIA_TYPES:
            await upload.close()
            return _error(415, "unsupported_media_type", "Only JPEG, PNG, and WebP are supported.")

        try:
            image_bytes = await upload.read(MAX_FILE_SIZE_BYTES + 1)
        finally:
            await upload.close()
        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            return _error(413, "payload_too_large", f"Image exceeds {MAX_FILE_SIZE_BYTES} bytes.")

        try:
            return await run_in_threadpool(service.predict, image_bytes)
        except InvalidImageError as exc:
            return _error(422, "invalid_image", str(exc))
        except ModelNotReadyError:
            return _error(503, "model_not_ready", "The model is not ready.")
        except InferenceFailedError:
            return _error(500, "inference_failed", "Inference could not be completed.")

    return app


app = create_app()
