import time
from typing import Callable, Mapping

import httpx

from .client import ModelContainerClient, response_json
from .config import MODEL_ID_PATTERN, ModelContainerRegistry
from .domain import ApplicationResponse, ImageInput
from .policies import ModelLatencySelector
from .telemetry import RequestCounters


def error_response(status_code: int, code: str, message: str) -> ApplicationResponse:
    return ApplicationResponse(
        status_code=status_code,
        payload={"error": {"code": code, "message": message}},
    )


class PredictionService:
    def __init__(
        self,
        registry: ModelContainerRegistry,
        clients: Mapping[str, ModelContainerClient],
        counters: RequestCounters,
        latency_selector: ModelLatencySelector,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._clients = clients
        self._counters = counters
        self._latency_selector = latency_selector
        self._clock = monotonic_clock

    async def record_rejected_request(self) -> None:
        await self._counters.record_received()

    async def predict(
        self,
        image: ImageInput,
        model_id: str | None,
        *,
        started_at: float | None = None,
    ) -> ApplicationResponse:
        request_started_at = self._clock() if started_at is None else started_at
        await self._counters.record_received()

        if model_id is not None and not MODEL_ID_PATTERN.fullmatch(model_id):
            return error_response(
                400, "invalid_model_id", "model_id must be one lowercase identifier."
            )
        resolved_model_id = model_id or self._registry.default_model_id
        client = self._clients.get(resolved_model_id)
        if client is None:
            return error_response(
                404,
                "unknown_model",
                f"Model {resolved_model_id!r} is not registered.",
            )

        try:
            model_response = await client.predict(
                filename=image.filename,
                content_type=image.content_type,
                image=image.content,
            )
        except httpx.TimeoutException:
            return error_response(
                504,
                "model_timeout",
                "The model container did not respond within the configured timeout.",
            )
        except httpx.RequestError:
            return error_response(
                503, "model_unavailable", "The model container is unavailable."
            )

        try:
            payload = response_json(model_response)
        except ValueError:
            return error_response(
                502,
                "invalid_model_response",
                "The model container returned invalid JSON.",
            )

        if model_response.is_success:
            elapsed_ms = max(0, round((self._clock() - request_started_at) * 1000))
            payload["end_to_end_time_ms"] = elapsed_ms
            await self._counters.record_completed()
            await self._latency_selector.record(resolved_model_id, elapsed_ms)

        return ApplicationResponse(model_response.status_code, payload)

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()
