from __future__ import annotations

import argparse
from io import BytesIO
import json
import statistics
from time import perf_counter_ns
from typing import Any

from PIL import Image

from .service import YoloModelService


def _elapsed_ms(started_ns: int) -> float:
    return (perf_counter_ns() - started_ns) / 1_000_000


def _input_png(size: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (size, size), (20, 40, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(statistics.fmean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "minimum_ms": round(min(values), 3),
        "maximum_ms": round(max(values), 3),
    }


def _event_value(event: Any, cuda_name: str, device_name: str) -> float:
    return float(getattr(event, cuda_name, getattr(event, device_name, 0.0)))


def _profile_events(service: YoloModelService, image_bytes: bytes) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as recording:
        service.predict(image_bytes)

    events = recording.key_averages()
    rows = [
        {
            "name": event.key,
            "calls": event.count,
            "self_cpu_time_us": round(float(event.self_cpu_time_total), 3),
            "self_cuda_time_us": round(
                _event_value(event, "self_cuda_time_total", "self_device_time_total"), 3
            ),
            "cuda_memory_bytes": int(
                _event_value(event, "cuda_memory_usage", "device_memory_usage")
            ),
        }
        for event in events
    ]
    total_cuda_us = sum(row["self_cuda_time_us"] for row in rows)
    top_cuda = sorted(rows, key=lambda row: row["self_cuda_time_us"], reverse=True)[:15]
    copies = [
        row
        for row in rows
        if "copy" in row["name"].lower() or "memcpy" in row["name"].lower()
    ]
    return {
        "total_self_cuda_time_us": round(total_cuda_us, 3),
        "top_cuda_events": top_cuda,
        "copy_events": copies,
        "warning": "Profiler instrumentation inflates CPU time; use CUDA event shares, not profiled wall latency.",
    }


def run_profile(*, image_size: int, warmups: int, repetitions: int) -> dict[str, Any]:
    image_bytes = _input_png(image_size)
    service = YoloModelService()
    service.load()
    for _ in range(warmups):
        service.predict(image_bytes)

    wall_ms = []
    processing_ms = []
    ultralytics_ms = []
    for _ in range(repetitions):
        started_ns = perf_counter_ns()
        response = service.predict(image_bytes)
        wall_ms.append(_elapsed_ms(started_ns))
        processing_ms.append(float(response.processing_time_ms))
        ultralytics_ms.append(float(response.inference_time_ms))

    return {
        "schema_version": "cuda-model-profile/v1",
        "device": service.device,
        "protocol": {
            "image_size": [image_size, image_size],
            "encoded_bytes": len(image_bytes),
            "decoded_rgb_bytes": image_size * image_size * 3,
            "warmups": warmups,
            "repetitions": repetitions,
        },
        "unprofiled": {
            "service_wall": _summary(wall_ms),
            "service_processing": _summary(processing_ms),
            "ultralytics_predict": _summary(ultralytics_ms),
            "outside_ultralytics": _summary(
                [processing - inference for processing, inference in zip(processing_ms, ultralytics_ms)]
            ),
        },
        "profiler": _profile_events(service, image_bytes),
        "buffer_path": [
            "multipart bytes",
            "BytesIO view over encoded bytes",
            "PIL decode",
            "PIL RGB conversion",
            "Ultralytics tensor preprocessing",
            "one observed host-to-device transfer",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    if args.image_size <= 0 or args.warmups < 0 or args.repetitions <= 0:
        parser.error("image size/repetitions must be positive and warmups non-negative")
    print(
        json.dumps(
            run_profile(
                image_size=args.image_size,
                warmups=args.warmups,
                repetitions=args.repetitions,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
