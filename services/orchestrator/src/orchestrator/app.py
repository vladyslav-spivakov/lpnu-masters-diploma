import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from types import MappingProxyType

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .adaptive_temporal import AdaptiveCadencePolicy, AdaptiveTemporalCoordinator
from .client import ModelContainerClient
from .config import MODEL_ID_PATTERN, Settings
from .domain import ApplicationResponse
from .frames import FrameExecutor
from .http import read_frame, read_image, to_json_response
from .policies import FramePacer, ModelLatencySelector, OutputCadenceRegulator
from .prediction import PredictionService, error_response
from .sort_temporal import KalmanSortAnchorTracker, KalmanSortEstimator
from .telemetry import DecisionJournal, FrameCounters, RequestCounters
from .temporal import (
    DetectorAnchorTracker,
    TemporalContextStore,
    TemporalCoordinator,
    create_temporal_estimator,
)


def create_app(
    model_clients: Mapping[str, ModelContainerClient] | None = None,
    settings: Settings | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    resolved_settings = settings or Settings.from_environment()
    registry = resolved_settings.model_registry
    clients = dict(model_clients) if model_clients is not None else {
        endpoint.model_id: ModelContainerClient(
            endpoint.base_url,
            timeout_seconds=resolved_settings.model_request_timeout_seconds,
        )
        for endpoint in registry.containers
    }
    if set(clients) != set(registry.by_id):
        raise ValueError("Model clients must match the configured model registry")

    request_counters = RequestCounters()
    frame_counters = FrameCounters(monotonic_clock)
    frame_pacer = FramePacer(monotonic_clock)
    output_regulator = OutputCadenceRegulator(monotonic_clock)
    model_selector = ModelLatencySelector(
        tuple(registry.by_id), registry.default_model_id
    )
    decision_journal = DecisionJournal()
    temporal_store = TemporalContextStore()
    if resolved_settings.temporal_estimator == "kalman-sort":
        temporal_estimator = KalmanSortEstimator()
        detector_result_tracker = KalmanSortAnchorTracker()
    else:
        temporal_estimator = create_temporal_estimator(
            resolved_settings.temporal_estimator
        )
        detector_result_tracker = DetectorAnchorTracker()
    fixed_temporal_coordinator = TemporalCoordinator(
        resolved_settings.temporal_detector_every_n,
        resolved_settings.temporal_context_max_age_ms,
        temporal_store,
        temporal_estimator,
        detector_result_tracker,
    )
    temporal_coordinator = (
        AdaptiveTemporalCoordinator(
            resolved_settings.temporal_detector_every_n,
            temporal_store,
            fixed_temporal_coordinator,
            AdaptiveCadencePolicy(
                resolved_settings.adaptive_max_anchor_age_ms,
                resolved_settings.adaptive_max_position_variance,
                resolved_settings.adaptive_min_match_ratio,
            ),
        )
        if resolved_settings.adaptive_temporal_enabled
        else fixed_temporal_coordinator
    )
    prediction_service = PredictionService(
        registry,
        clients,
        request_counters,
        model_selector,
        monotonic_clock,
    )
    frame_executor = FrameExecutor(
        resolved_settings,
        prediction_service,
        frame_counters,
        frame_pacer,
        model_selector,
        decision_journal,
        temporal_coordinator,
        output_regulator,
        monotonic_clock,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        app.state.frame_worker_task = frame_executor.start()
        try:
            yield
        finally:
            await frame_executor.close()
            app.state.frame_worker_task = None
            await prediction_service.close()

    app = FastAPI(
        title="Adaptive Neural Platform Orchestrator",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.model_registry = registry
    app.state.model_clients = MappingProxyType(clients)
    app.state.frame_queue = frame_executor.queue
    app.state.frame_worker_task = None
    app.state.request_counters = request_counters
    app.state.frame_counters = frame_counters
    app.state.frame_pacer = frame_pacer
    app.state.output_cadence_regulator = output_regulator
    app.state.model_latency_selector = model_selector
    app.state.decision_journal = decision_journal
    app.state.temporal_context_store = temporal_store
    app.state.temporal_coordinator = temporal_coordinator

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/metrics")
    async def metrics() -> dict[str, int | float]:
        request_snapshot = await request_counters.snapshot()
        frame_snapshot = await frame_counters.snapshot()
        return {**request_snapshot, **frame_snapshot}

    @app.get("/decisions")
    async def decisions() -> dict[str, list[dict[str, object]]]:
        return {"events": await decision_journal.snapshot()}

    @app.post("/predict")
    async def predict(request: Request, model_id: str | None = None) -> JSONResponse:
        started_at = monotonic_clock()
        model_ids = request.query_params.getlist("model_id")
        if len(model_ids) > 1 or (
            model_id is not None and not MODEL_ID_PATTERN.fullmatch(model_id)
        ):
            await prediction_service.record_rejected_request()
            return to_json_response(
                error_response(
                    400,
                    "invalid_model_id",
                    "model_id must be one lowercase identifier.",
                )
            )

        image = await read_image(request)
        if isinstance(image, ApplicationResponse):
            await prediction_service.record_rejected_request()
            return to_json_response(image)
        return to_json_response(
            await prediction_service.predict(image, model_id, started_at=started_at)
        )

    @app.post("/frames")
    async def submit_frame(
        request: Request,
        stream_id: str | None = None,
        frame_id: str | None = None,
        captured_at_ms: str | None = None,
        model_id: str | None = None,
        target_fps: str | None = None,
        max_latency_ms: str | None = None,
    ) -> JSONResponse:
        frame = read_frame(
            request,
            resolved_settings,
            stream_id=stream_id,
            frame_id=frame_id,
            captured_at_ms=captured_at_ms,
            target_fps=target_fps,
            max_latency_ms=max_latency_ms,
        )
        if isinstance(frame, ApplicationResponse):
            return to_json_response(frame)
        admission_error = await frame_executor.admit(frame)
        if admission_error is not None:
            return to_json_response(admission_error)
        image = await read_image(request)
        if isinstance(image, ApplicationResponse):
            await prediction_service.record_rejected_request()
            await frame_executor.record_input_failure(frame, model_id, image)
            return to_json_response(image)
        return to_json_response(
            await frame_executor.submit_admitted(frame, image, model_id)
        )

    return app


app = create_app()
