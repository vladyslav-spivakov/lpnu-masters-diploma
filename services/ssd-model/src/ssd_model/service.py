from time import perf_counter_ns
from typing import Any, Callable

from PIL import Image

from .config import API_VERSION, MAX_FILE_SIZE_BYTES, SUPPORTED_MEDIA_TYPES, Settings
from .errors import InferenceFailedError, ModelNotReadyError
from .image import decode_rgb_image
from .normalization import normalize_ssd_output


class SsdLiteModelService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        clock_ns: Callable[[], int] = perf_counter_ns,
    ) -> None:
        self.settings = settings or Settings.from_environment()
        self._model: Any | None = None
        self._transform: Any | None = None
        self._categories: list[str] = []
        self._clock_ns = clock_ns

    @property
    def ready(self) -> bool:
        return self._model is not None and self._transform is not None and bool(self._categories)

    def load(self) -> None:
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )

        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        model = ssdlite320_mobilenet_v3_large(weights=weights)
        model.eval()
        self._model = model
        self._transform = weights.transforms()
        self._categories = [str(value) for value in weights.meta["categories"]]

    def info(self) -> dict[str, Any]:
        return {
            "api_version": API_VERSION,
            "model": self.settings.model_name,
            "model_version": self.settings.model_version,
            "task": "object-detection",
            "input": {
                "media_types": list(SUPPORTED_MEDIA_TYPES),
                "max_file_size_bytes": MAX_FILE_SIZE_BYTES,
            },
        }

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        if not self.ready:
            raise ModelNotReadyError("The model is not ready.")

        started_ns = self._clock_ns()
        image = decode_rgb_image(image_bytes)
        try:
            output, inference_time_ms = self._run_inference(image)
            response = normalize_ssd_output(
                output,
                categories=self._categories,
                model_name=self.settings.model_name,
                model_version=self.settings.model_version,
                score_threshold=self.settings.score_threshold,
                image_width=image.width,
                image_height=image.height,
                processing_time_ms=0,
                inference_time_ms=inference_time_ms,
            )
            response["processing_time_ms"] = self._elapsed_ms(started_ns)
            return response
        except Exception as exc:
            raise InferenceFailedError("Inference could not be completed.") from exc

    def _run_inference(self, image: Image.Image) -> tuple[dict[str, Any], int]:
        import torch

        transformed = self._transform(image)
        with torch.inference_mode():
            inference_started_ns = self._clock_ns()
            outputs = self._model([transformed])
            inference_time_ms = self._elapsed_ms(inference_started_ns)

        if len(outputs) != 1:
            raise ValueError("SSDLite must return exactly one result")
        return outputs[0], inference_time_ms

    def _elapsed_ms(self, started_ns: int) -> int:
        return max(0, round((self._clock_ns() - started_ns) / 1_000_000))
