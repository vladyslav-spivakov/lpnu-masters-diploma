from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable

from PIL import Image

from .config import API_VERSION, MAX_FILE_SIZE_BYTES, SUPPORTED_MEDIA_TYPES, Settings
from .device import CudaRuntime, resolve_device, synchronize_device
from .errors import InferenceFailedError, ModelNotReadyError
from .image import decode_rgb_image
from .normalization import normalize_yolo_result
from .schemas import InfoResponse, InputCapabilities, PredictionResponse


class YoloModelService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        clock_ns: Callable[[], int] = perf_counter_ns,
        cuda: CudaRuntime | None = None,
    ) -> None:
        self.settings = settings or Settings.from_environment()
        self._model: Any | None = None
        self._device: str | None = None
        self._clock_ns = clock_ns
        self._cuda = cuda

    @property
    def ready(self) -> bool:
        return self._model is not None and self._device is not None

    @property
    def device(self) -> str | None:
        return self._device

    def load(self) -> None:
        import torch
        from ultralytics import YOLO

        cuda = self._cuda or torch.cuda
        device = resolve_device(self.settings.device, cuda)
        weights_path = Path(self.settings.model_weights).expanduser()
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(weights_path))
        self._cuda = cuda
        self._device = device
        self._model = model

    def info(self) -> InfoResponse:
        return InfoResponse(
            api_version=API_VERSION,
            model=self.settings.model_name,
            model_version=self.settings.model_version,
            input=InputCapabilities(
                media_types=list(SUPPORTED_MEDIA_TYPES),
                max_file_size_bytes=MAX_FILE_SIZE_BYTES,
            ),
        )

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        if self._model is None:
            raise ModelNotReadyError("The model is not ready.")

        started_ns = self._clock_ns()
        image = decode_rgb_image(image_bytes)
        try:
            result, inference_time_ms = self._run_inference(image)
            response = normalize_yolo_result(
                result,
                model_name=self.settings.model_name,
                model_version=self.settings.model_version,
                image_width=image.width,
                image_height=image.height,
                processing_time_ms=0,
                inference_time_ms=inference_time_ms,
            )
            response.processing_time_ms = self._elapsed_ms(started_ns)
            return response
        except Exception as exc:
            raise InferenceFailedError("Inference could not be completed.") from exc

    def _run_inference(self, image: Image.Image) -> tuple[Any, int]:
        if self._device is None or self._cuda is None:
            raise RuntimeError("The execution device is not initialized")
        synchronize_device(self._device, self._cuda)
        inference_started_ns = self._clock_ns()
        results = self._model.predict(source=image, verbose=False, device=self._device)
        synchronize_device(self._device, self._cuda)
        inference_time_ms = self._elapsed_ms(inference_started_ns)

        if len(results) != 1:
            raise ValueError("YOLO must return exactly one result")
        return results[0], inference_time_ms

    def _elapsed_ms(self, started_ns: int) -> int:
        return max(0, round((self._clock_ns() - started_ns) / 1_000_000))
