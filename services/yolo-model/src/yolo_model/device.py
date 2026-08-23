from typing import Protocol


class CudaRuntime(Protocol):
    def is_available(self) -> bool: ...

    def synchronize(self) -> None: ...


def resolve_device(requested_device: str, cuda: CudaRuntime) -> str:
    if requested_device == "cpu":
        return "cpu"
    if cuda.is_available():
        return "cuda"
    if requested_device == "auto":
        return "cpu"
    raise RuntimeError("YOLO_DEVICE=cuda was requested, but CUDA is unavailable")


def synchronize_device(device: str, cuda: CudaRuntime) -> None:
    if device == "cuda":
        cuda.synchronize()
