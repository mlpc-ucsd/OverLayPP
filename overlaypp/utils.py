"""Small shared helpers used across pipeline stages."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")


def iter_image_paths(image_dir: Path) -> Iterator[Path]:
    """Yield every supported image under ``image_dir`` (case-insensitive)."""
    exts = {ext.lower() for ext in IMAGE_EXTENSIONS}
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def find_image_for_stem(image_dir: Path, stem: str) -> Path | None:
    """Return the image path matching ``stem`` regardless of casing/extension."""
    for ext in IMAGE_EXTENSIONS:
        for candidate_ext in (ext, ext.upper()):
            candidate = image_dir / f"{stem}{candidate_ext}"
            if candidate.exists():
                return candidate
    return None


def load_image_rgb(path: Path | str) -> Image.Image:
    """Load an image from disk and convert it to RGB."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def center_crop(image: Image.Image, target_size: int) -> Image.Image | None:
    """Return a centre-cropped square if the image is large enough, else ``None``."""
    width, height = image.size
    if width < target_size or height < target_size:
        return None
    left = (width - target_size) // 2
    top = (height - target_size) // 2
    return image.crop((left, top, left + target_size, top + target_size))


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from a VLM response.

    Accepts either a bare JSON object or one wrapped in a ```json ...``` fence.
    Returns an empty dict on failure.
    """
    fenced = re.search(r"```json\s*([\s\S]+?)\s*```", text)
    candidate = fenced.group(1) if fenced else text
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def clamp_bbox(
    bbox: list[float] | tuple[float, ...],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """Clamp a ``[x1, y1, x2, y2]`` bbox to image bounds, returning ``None`` if degenerate."""
    if len(bbox) != 4:
        return None
    x1 = max(0, min(int(bbox[0]), image_width))
    y1 = max(0, min(int(bbox[1]), image_height))
    x2 = max(0, min(int(bbox[2]), image_width))
    y2 = max(0, min(int(bbox[3]), image_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def is_non_empty_str(value: Any) -> bool:
    return value is not None and bool(str(value).strip())
