from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorBody(ContractModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ErrorResponse(ContractModel):
    error: ErrorBody


class HealthResponse(ContractModel):
    status: Literal["ready"] = "ready"


class InputCapabilities(ContractModel):
    media_types: list[str]
    max_file_size_bytes: int = Field(ge=1)


class InfoResponse(ContractModel):
    api_version: Literal["1.0.0"] = "1.0.0"
    model: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    task: Literal["object-detection"] = "object-detection"
    input: InputCapabilities


class ImageMetadata(ContractModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class Detection(ContractModel):
    label: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bbox: tuple[float, float, float, float]


class PredictionResponse(ContractModel):
    model: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    image: ImageMetadata
    predictions: list[Detection]
    processing_time_ms: int = Field(ge=0)
    inference_time_ms: int = Field(ge=0)
