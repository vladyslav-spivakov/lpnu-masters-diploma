from typing import Any

import httpx


class ModelContainerClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )

    async def predict(self, *, filename: str, content_type: str, image: bytes) -> httpx.Response:
        return await self._client.post(
            "/predict",
            files={"image": (filename, image, content_type)},
        )

    async def close(self) -> None:
        await self._client.aclose()


def response_json(response: httpx.Response) -> dict[str, Any]:

    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Model container response must be a JSON object")
    return payload
