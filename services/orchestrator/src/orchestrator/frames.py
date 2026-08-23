import asyncio
import math
import time
from typing import Any, Callable

from .config import Settings
from .domain import ApplicationResponse, FrameInput, FrameJob, ImageInput
from .policies import (
    FramePacer,
    ModelLatencySelector,
    OutputCadenceRegulator,
    evaluate_frame_constraints,
)
from .prediction import PredictionService, error_response
from .telemetry import DecisionJournal, FrameCounters
from .temporal import TemporalCoordinator


class FrameExecutor:
    def __init__(
        self,
        settings: Settings,
        prediction_service: PredictionService,
        counters: FrameCounters,
        pacer: FramePacer,
        model_selector: ModelLatencySelector,
        journal: DecisionJournal,
        temporal: TemporalCoordinator,
        output_regulator: OutputCadenceRegulator,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._prediction_service = prediction_service
        self._counters = counters
        self._pacer = pacer
        self._model_selector = model_selector
        self._journal = journal
        self._temporal = temporal
        self._output_regulator = output_regulator
        self._clock = monotonic_clock
        self.queue: asyncio.Queue[FrameJob] = asyncio.Queue(
            maxsize=settings.frame_queue_capacity
        )
        self.worker_task: asyncio.Task[None] | None = None

    def start(self) -> asyncio.Task[None]:
        self.worker_task = asyncio.create_task(self._run_worker(), name="frame-queue-worker")
        return self.worker_task

    async def close(self) -> None:
        if self.worker_task is not None:
            self.worker_task.cancel()
            await asyncio.gather(self.worker_task, return_exceptions=True)
        while not self.queue.empty():
            job = self.queue.get_nowait()
            if not job.result.done():
                job.result.cancel()
            self.queue.task_done()
        self.worker_task = None

    async def submit(
        self,
        frame: FrameInput,
        image: ImageInput,
        model_id: str | None,
    ) -> ApplicationResponse:
        admission_error = await self.admit(frame)
        if admission_error is not None:
            return admission_error
        return await self.submit_admitted(frame, image, model_id)

    async def admit(self, frame: FrameInput) -> ApplicationResponse | None:
        await self._counters.record_received()
        frame_payload = frame.as_dict()
        if frame.frame_id % self._settings.frame_sample_every_n != 0:
            await self._counters.record_skipped()
            await self._journal.record(frame_payload, "every_n_skipped")
            return error_response(
                409,
                "frame_skipped",
                "The frame was skipped by the configured every-N policy.",
            )

        admitted, retry_after_seconds = await self._pacer.admit(
            frame.stream_id, frame.target_fps
        )
        if not admitted:
            await self._counters.record_skipped()
            retry_after_ms = max(1, math.ceil(retry_after_seconds * 1000))
            await self._journal.record(
                frame_payload, "pacing_skipped", retry_after_ms=retry_after_ms
            )
            return ApplicationResponse(
                409,
                {
                    "error": {
                        "code": "frame_pacing_skipped",
                        "message": "The frame arrived before the target-FPS interval elapsed.",
                        "retry_after_ms": retry_after_ms,
                    }
                },
            )

        return None

    async def submit_admitted(
        self,
        frame: FrameInput,
        image: ImageInput,
        model_id: str | None,
    ) -> ApplicationResponse:
        result = asyncio.get_running_loop().create_future()
        job = FrameJob(
            image=image,
            model_id=model_id,
            frame=frame,
            enqueued_at=self._clock(),
            result=result,
        )

        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull:
            await self._replace_oldest_waiting_job(job)
        response = await result
        if response.successful and self._settings.output_cadence_enabled:
            cadence = await self._output_regulator.regulate(
                frame.stream_id, frame.target_fps
            )
            response.payload["frame"]["output_cadence"] = cadence
        return response

    async def record_input_failure(
        self,
        frame: FrameInput,
        model_id: str | None,
        response: ApplicationResponse,
    ) -> None:
        model_selection = await self._model_selector.select(
            model_id, frame.max_latency_ms
        )
        await self._record_failure(frame.as_dict(), model_selection, response)

    async def _replace_oldest_waiting_job(self, new_job: FrameJob) -> None:
        dropped_job = self.queue.get_nowait()
        self.queue.task_done()
        self.queue.put_nowait(new_job)
        await self._counters.record_dropped()
        await self._journal.record(dropped_job.frame.as_dict(), "dropped")
        if not dropped_job.result.done():
            dropped_job.result.set_result(
                error_response(
                    409,
                    "frame_dropped",
                    "The frame was replaced by a newer waiting frame.",
                )
            )

    async def _run_worker(self) -> None:
        while True:
            job = await self.queue.get()
            try:
                response = await self._execute(job)
                if not job.result.cancelled():
                    job.result.set_result(response)
            except asyncio.CancelledError:
                if not job.result.done():
                    job.result.cancel()
                raise
            except Exception as exc:
                await self._journal.record(
                    job.frame.as_dict(),
                    "failed",
                    http_status=500,
                    error_code="internal_error",
                )
                if not job.result.done():
                    job.result.set_exception(exc)
            finally:
                self.queue.task_done()

    async def _execute(self, job: FrameJob) -> ApplicationResponse:
        frame_payload = job.frame.as_dict()
        queue_wait_time_ms = max(0, round((self._clock() - job.enqueued_at) * 1000))
        model_selection = await self._model_selector.select(
            job.model_id, job.frame.max_latency_ms
        )
        prediction, temporal_metadata = await self._predict_frame(job, model_selection)

        if not prediction.successful:
            await self._record_failure(frame_payload, model_selection, prediction)
            return prediction

        prediction_payload = dict(prediction.payload)
        prediction_payload["end_to_end_time_ms"] = max(
            0, round((self._clock() - job.enqueued_at) * 1000)
        )
        processing_fps, sample_count = await self._counters.record_processed()
        constraint_status = evaluate_frame_constraints(
            processing_fps=processing_fps,
            fps_sample_count=sample_count,
            target_fps=job.frame.target_fps,
            end_to_end_time_ms=prediction_payload["end_to_end_time_ms"],
            max_latency_ms=job.frame.max_latency_ms,
        )
        journal_details = {
            "model_selection": {
                "selected_model_id": model_selection["selected_model_id"],
                "reason": model_selection["reason"],
            },
            "constraint_status": {
                "status": constraint_status["status"],
                "violations": constraint_status["violations"],
            },
        }
        if temporal_metadata is not None:
            journal_details["temporal"] = temporal_metadata
        await self._journal.record(frame_payload, "processed", **journal_details)
        frame_metadata = (
            {"temporal": temporal_metadata} if temporal_metadata is not None else {}
        )
        return ApplicationResponse(
            200,
            {
                "frame": {
                    **frame_payload,
                    "queue_wait_time_ms": queue_wait_time_ms,
                    "model_selection": model_selection,
                    "constraint_status": constraint_status,
                    **frame_metadata,
                },
                "prediction": prediction_payload,
            },
        )

    async def _predict_frame(
        self,
        job: FrameJob,
        model_selection: dict[str, Any],
    ) -> tuple[ApplicationResponse, dict[str, Any] | None]:
        if not self._settings.temporal_enabled:
            return await self._predict_with_detector(job, model_selection), None
        if model_selection["selected_model_id"] not in self._settings.model_registry.by_id:
            return await self._predict_with_detector(job, model_selection), None

        decision = await self._temporal.estimate(job.frame)
        if decision.estimate is not None:
            return ApplicationResponse(200, decision.estimate.prediction), {
                "source": "temporal",
                "reason": decision.reason,
                "generation": decision.estimate.context.generation,
            }

        prediction = await self._predict_with_detector(job, model_selection)
        if not prediction.successful:
            return prediction, None
        context = await self._temporal.record_detector_result(job.frame, prediction.payload)
        return prediction, {
            "source": "detector",
            "reason": decision.reason,
            "generation": context.generation,
        }

    async def _predict_with_detector(
        self,
        job: FrameJob,
        model_selection: dict[str, Any],
    ) -> ApplicationResponse:
        return await self._prediction_service.predict(
            job.image, model_selection["selected_model_id"]
        )

    async def _record_failure(
        self,
        frame: dict[str, Any],
        model_selection: dict[str, Any],
        response: ApplicationResponse,
    ) -> None:
        error = response.payload.get("error", {})
        error_code = error.get("code") if isinstance(error, dict) else None
        await self._journal.record(
            frame,
            "failed",
            model_selection={
                "selected_model_id": model_selection["selected_model_id"],
                "reason": model_selection["reason"],
            },
            http_status=response.status_code,
            error_code=error_code if isinstance(error_code, str) else None,
        )
