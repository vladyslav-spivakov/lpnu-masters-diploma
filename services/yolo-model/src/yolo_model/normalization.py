from typing import Any

from .schemas import Detection, ImageMetadata, PredictionResponse


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _class_label(names: Any, class_id: int) -> str:
    try:
        label = names[class_id]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Missing label for class id {class_id}") from exc
    return str(label)


def normalize_yolo_result(
    result: Any,
    *,
    model_name: str,
    model_version: str,
    image_width: int,
    image_height: int,
    processing_time_ms: int,
    inference_time_ms: int,
) -> PredictionResponse:
    boxes = result.boxes
    xyxy_rows = _as_list(boxes.xyxy)
    confidences = _as_list(boxes.conf)
    class_ids = _as_list(boxes.cls)
    if not (len(xyxy_rows) == len(confidences) == len(class_ids)):
        raise ValueError("YOLO result arrays have inconsistent lengths")

    predictions: list[Detection] = []
    for xyxy, confidence, class_id in zip(xyxy_rows, confidences, class_ids, strict=True):
        if len(xyxy) != 4:
            raise ValueError("YOLO bounding box must contain four coordinates")
        x_min, y_min, x_max, y_max = (float(value) for value in xyxy)
        x_min = min(max(x_min, 0.0), float(image_width))
        y_min = min(max(y_min, 0.0), float(image_height))
        x_max = min(max(x_max, x_min), float(image_width))
        y_max = min(max(y_max, y_min), float(image_height))
        predictions.append(
            Detection(
                label=_class_label(result.names, int(class_id)),
                confidence=min(max(float(confidence), 0.0), 1.0),
                bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
            )
        )

    return PredictionResponse(
        model=model_name,
        model_version=model_version,
        image=ImageMetadata(width=image_width, height=image_height),
        predictions=predictions,
        processing_time_ms=max(0, processing_time_ms),
        inference_time_ms=max(0, inference_time_ms),
    )
