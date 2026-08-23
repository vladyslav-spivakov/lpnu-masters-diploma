from typing import Any


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def normalize_ssd_output(
    output: dict[str, Any],
    *,
    categories: list[str],
    model_name: str,
    model_version: str,
    score_threshold: float,
    image_width: int,
    image_height: int,
    processing_time_ms: int,
    inference_time_ms: int,
) -> dict[str, Any]:
    try:
        boxes = _as_list(output["boxes"])
        scores = _as_list(output["scores"])
        labels = _as_list(output["labels"])
    except (KeyError, TypeError) as exc:
        raise ValueError("SSDLite output is missing boxes, scores, or labels") from exc
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("SSDLite output arrays have inconsistent lengths")

    predictions: list[dict[str, Any]] = []
    for box, score, label_id in zip(boxes, scores, labels, strict=True):
        confidence = float(score)
        if confidence < score_threshold:
            continue
        if len(box) != 4:
            raise ValueError("SSDLite bounding box must contain four coordinates")
        category_index = int(label_id)
        if not 0 <= category_index < len(categories):
            raise ValueError(f"Missing category for class id {category_index}")

        x_min, y_min, x_max, y_max = (float(value) for value in box)
        x_min = min(max(x_min, 0.0), float(image_width))
        y_min = min(max(y_min, 0.0), float(image_height))
        x_max = min(max(x_max, x_min), float(image_width))
        y_max = min(max(y_max, y_min), float(image_height))
        predictions.append(
            {
                "label": str(categories[category_index]),
                "confidence": min(max(confidence, 0.0), 1.0),
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            }
        )

    return {
        "model": model_name,
        "model_version": model_version,
        "image": {"width": image_width, "height": image_height},
        "predictions": predictions,
        "processing_time_ms": max(0, processing_time_ms),
        "inference_time_ms": max(0, inference_time_ms),
    }
