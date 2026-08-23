from io import BytesIO

from PIL import Image, UnidentifiedImageError

from .errors import InvalidImageError


def decode_rgb_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            "The uploaded file cannot be decoded as an image."
        ) from exc
