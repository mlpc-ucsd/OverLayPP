"""Download and crop images for the published OverLay++ dataset.

Annotations are streamed from Hugging Face. Each source URL is downloaded and
centre-cropped, while the corresponding published annotation is saved as JSON.
Annotated previews are saved separately with object bounding boxes and labels.
This utility is independent of the four-stage dataset-generation pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from overlaypp.config import TARGET_RESOLUTION
from overlaypp.utils import center_crop, clamp_bbox, read_json

DEFAULT_DATASET_NAME = "mlpcucsd/OverLayPP"
DEFAULT_DOWNLOAD_TIMEOUT_S = 15.0
DEFAULT_JPEG_QUALITY = 95
ANNOTATION_FIELDS = (
    "url",
    "objects",
    "long_global_caption",
    "short_global_caption",
    "image_hash",
)


@dataclass(frozen=True)
class DownloadResult:
    status: str
    identifier: str
    error: str | None = None


def _identifier(sample: dict[str, Any]) -> str:
    """Return a safe, stable filename stem for a dataset example."""
    image_hash = str(sample.get("image_hash") or "").strip()
    if re.fullmatch(r"[0-9a-fA-F]{64}", image_hash):
        return image_hash.lower()

    url = str(sample.get("url") or "").strip()
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _metadata(sample: dict[str, Any]) -> dict[str, Any]:
    """Keep the fields defined by the published OverLay++ dataset card."""
    return {key: sample[key] for key in ANNOTATION_FIELDS if key in sample}


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    temporary_path.replace(path)


def _write_overlay(image_path: Path, output_path: Path, objects: list[Any]) -> None:
    """Draw published pixel-coordinate boxes on a separate copy of the image."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, image.width // 64))
    colors = ("#e63946", "#0077b6", "#2a9d8f", "#9b5de5", "#f4a261")
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        try:
            bbox = clamp_bbox(obj.get("bbox", []), image.width, image.height)
        except (TypeError, ValueError, OverflowError):
            continue
        if bbox is None:
            continue
        color = colors[index % len(colors)]
        draw.rectangle(bbox, outline=color, width=max(2, image.width // 256))
        label = str(obj.get("category") or f"Object {index + 1}")[:48]
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        x, y = bbox[0], max(0, bbox[1] - (bottom - top) - 4)
        draw.rectangle(
            (x, y, min(image.width - 1, x + right - left + 4), y + bottom - top + 4),
            fill=color,
        )
        draw.text((x + 2 - left, y + 2 - top), label, fill="white", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".jpg.tmp")
    image.save(temporary_path, format="JPEG", quality=DEFAULT_JPEG_QUALITY)
    temporary_path.replace(output_path)


def _download_example(
    sample: dict[str, Any],
    *,
    images_dir: Path,
    metadata_dir: Path,
    target_resolution: int,
    timeout: float,
    images_overlayed_dir: Path | None = None,
) -> DownloadResult:
    identifier = _identifier(sample)
    image_path = images_dir / f"{identifier}.jpg"
    metadata_path = metadata_dir / f"{identifier}_metadata.json"
    images_overlayed_dir = (
        images_overlayed_dir or images_dir.parent / "images_overlayed"
    )
    overlay_path = images_overlayed_dir / image_path.name

    if image_path.exists() and metadata_path.exists() and overlay_path.exists():
        return DownloadResult("skipped", identifier)

    url = str(sample.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return DownloadResult("failed", identifier, "missing or invalid URL")

    try:
        if not image_path.exists():
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))
            image.load()
            if image.mode != "RGB":
                image = image.convert("RGB")

            cropped = center_crop(image, target_size=target_resolution)
            if cropped is None:
                return DownloadResult(
                    "failed",
                    identifier,
                    f"source image is smaller than {target_resolution}x{target_resolution}",
                )

            temporary_image = image_path.with_suffix(".jpg.tmp")
            cropped.save(
                temporary_image,
                format="JPEG",
                quality=DEFAULT_JPEG_QUALITY,
            )
            temporary_image.replace(image_path)

        if metadata_path.exists():
            metadata = read_json(metadata_path)
        else:
            metadata = _metadata(sample)
            _write_json_atomic(metadata_path, metadata)
        if not overlay_path.exists():
            _write_overlay(image_path, overlay_path, metadata.get("objects") or [])
        return DownloadResult("downloaded", identifier)
    except Exception as exc:
        return DownloadResult("failed", identifier, str(exc))


def _default_num_workers() -> int:
    return min(32, max(8, (os.cpu_count() or 4) * 2))


def download_dataset(
    *,
    dataset_name: str,
    split: str,
    images_dir: Path,
    metadata_dir: Path,
    target_resolution: int,
    num_workers: int,
    timeout: float,
    images_overlayed_dir: Path | None = None,
    max_samples: int | None = None,
    use_token: bool = False,
) -> None:
    """Stream annotations and concurrently download their source images."""
    images_overlayed_dir = (
        images_overlayed_dir or images_dir.parent / "images_overlayed"
    )
    if images_overlayed_dir.resolve() == images_dir.resolve():
        raise ValueError("--images-overlayed-dir must differ from --images-dir")
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    images_overlayed_dir.mkdir(parents=True, exist_ok=True)

    dataset: Iterable[dict[str, Any]] = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        token=True if use_token else None,
    )

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    seen_identifiers: set[str] = set()

    def record(future: Future[DownloadResult], progress: tqdm) -> None:
        result = future.result()
        counts[result.status] += 1
        progress.update(1)

    with (
        ThreadPoolExecutor(max_workers=num_workers) as executor,
        tqdm(total=max_samples, desc="Preparing OverLay++ images") as progress,
    ):
        pending: set[Future[DownloadResult]] = set()
        submitted = 0

        for sample in dataset:
            if max_samples is not None and submitted >= max_samples:
                break

            identifier = _identifier(sample)
            if identifier in seen_identifiers:
                continue
            seen_identifiers.add(identifier)

            pending.add(
                executor.submit(
                    _download_example,
                    sample,
                    images_dir=images_dir,
                    metadata_dir=metadata_dir,
                    target_resolution=target_resolution,
                    timeout=timeout,
                    images_overlayed_dir=images_overlayed_dir,
                )
            )
            submitted += 1

            if len(pending) >= num_workers * 2:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    record(future, progress)

        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                record(future, progress)

    print(
        "[download_dataset] "
        f"{counts['downloaded']} downloaded, "
        f"{counts['skipped']} already present, "
        f"{counts['failed']} failed"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/images"))
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("dataset/metadata"),
    )
    parser.add_argument(
        "--images-overlayed-dir",
        type=Path,
        default=None,
        help="Annotated previews (default: images_overlayed beside --images-dir).",
    )
    parser.add_argument(
        "--target-resolution",
        type=int,
        default=TARGET_RESOLUTION,
    )
    parser.add_argument("--num-workers", type=int, default=_default_num_workers())
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=DEFAULT_DOWNLOAD_TIMEOUT_S,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional number of unique examples to prepare.",
    )
    parser.add_argument(
        "--token",
        action="store_true",
        help="Use the token saved by `hf auth login`.",
    )
    args = parser.parse_args()

    if args.target_resolution <= 0:
        parser.error("--target-resolution must be positive")
    if args.num_workers <= 0:
        parser.error("--num-workers must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")

    download_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        images_dir=args.images_dir,
        metadata_dir=args.metadata_dir,
        images_overlayed_dir=args.images_overlayed_dir,
        target_resolution=args.target_resolution,
        num_workers=args.num_workers,
        timeout=args.download_timeout,
        max_samples=args.max_samples,
        use_token=args.token,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
