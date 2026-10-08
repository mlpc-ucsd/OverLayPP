"""Stage I: Image filtering.

Reads the upstream LAION-Aesthetics v2 parquet shards, keeps only rows whose
aesthetic score >= ``MIN_AESTHETIC_SCORE`` and whose source resolution is at
least ``MIN_RESOLUTION`` on both sides, downloads the image, centre-crops it
to ``TARGET_RESOLUTION``, and writes:

- ``<images-dir>/<stem>.jpg`` -- the cropped image.
- ``<metadata-dir>/<stem>_metadata.json`` -- per-image metadata seeded with
  the source ``url``. Later stages add ``detections``, captions, etc. to the
  same file.

This corresponds to stage I in the OverLay++ pipeline figure.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from dataclasses import dataclass
from functools import partial
from multiprocessing import Manager, Pool
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pyarrow.parquet as pq
import requests
from PIL import Image
from tqdm import tqdm

from overlaypp.config import (
    MIN_AESTHETIC_SCORE,
    MIN_RESOLUTION,
    TARGET_RESOLUTION,
)
from overlaypp.utils import center_crop, write_json

DEFAULT_DOWNLOAD_TIMEOUT_S: float = 10.0
DEFAULT_JPEG_QUALITY: int = 95

# Most filesystems cap a single path component at 255 bytes. We reserve some
# headroom for the "_metadata.json" sibling and an occasional "_N" disambiguation
# suffix, so 200 bytes is a safe upper bound for the image filename itself.
MAX_FILENAME_BYTES: int = 200


@dataclass
class ShardStats:
    extracted: int
    failed: int


def _download_image(url: str, timeout: float) -> Image.Image | None:
    """Download a URL and return a fully-decoded RGB image, or ``None`` on failure.

    Calls ``Image.load()`` eagerly so truncated/corrupt downloads are caught
    here rather than blowing up later inside ``crop`` / ``save``.
    """
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        img.load()  # forces decoding so truncated bytes raise here
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        return None


def _row_meets_filters(row: pd.Series, min_score: float, min_resolution: int) -> bool:
    """Return True iff the parquet row passes Stage I's filters.

    Mirrors the original ``extract_aesthetics_dataset.py`` exactly:
    rows are kept when ``AESTHETIC_SCORE > min_score`` (strict greater) and
    both ``WIDTH`` and ``HEIGHT`` are at least ``min_resolution``.
    """
    width = row.get("WIDTH")
    height = row.get("HEIGHT")
    score = row.get("AESTHETIC_SCORE")
    url = row.get("URL")

    if pd.isna(width) or pd.isna(height) or pd.isna(score) or pd.isna(url):
        return False
    if float(width) < min_resolution or float(height) < min_resolution:
        return False
    if float(score) <= min_score:
        return False
    return str(url).strip().startswith("http")


def _next_image_path(
    url: str, counter_lock, counter_dict, images_dir: Path
) -> Path | None:
    """Pick a unique filename for an image to be saved under ``images_dir``.

    Returns ``None`` when the URL-derived basename is too long for the
    filesystem to handle (errno 36 / ENAMETOOLONG); the caller should treat
    that as a soft failure and skip the row.
    """
    with counter_lock:
        counter_dict["count"] = counter_dict.get("count", 0) + 1
        file_idx = counter_dict["count"]

    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    if not filename or "." not in filename:
        filename = f"{file_idx:08d}.jpg"

    if len(filename.encode("utf-8", errors="replace")) > MAX_FILENAME_BYTES:
        return None

    base = Path(filename).stem
    ext = Path(filename).suffix or ".jpg"
    candidate = images_dir / filename
    bump = 0
    while candidate.exists():
        bump += 1
        candidate = images_dir / f"{base}_{bump}{ext}"
    return candidate


def process_shard(
    parquet_file: Path,
    *,
    images_dir: Path,
    metadata_dir: Path,
    min_score: float,
    min_resolution: int,
    target_resolution: int,
    download_timeout: float,
    counter_lock,
    counter_dict,
) -> ShardStats:
    """Filter, download, crop, and write per-image metadata for one parquet shard."""
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    failed = 0

    try:
        df = pq.read_table(parquet_file).to_pandas()
    except Exception as exc:
        print(f"\n[filter_images] error reading {parquet_file.name}: {exc}")
        return ShardStats(0, 0)

    seen_urls: dict[str, Path] = {}

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"  {parquet_file.name}",
        leave=False,
    ):
        if not _row_meets_filters(row, min_score, min_resolution):
            continue
        url = str(row["URL"]).strip()

        try:
            if url in seen_urls:
                image_path = seen_urls[url]
            else:
                downloaded = _download_image(url, timeout=download_timeout)
                if downloaded is None:
                    failed += 1
                    continue
                cropped = center_crop(downloaded, target_size=target_resolution)
                if cropped is None:
                    failed += 1
                    continue

                image_path = _next_image_path(
                    url, counter_lock, counter_dict, images_dir
                )
                if image_path is None:
                    # URL-derived basename is too long for the filesystem.
                    failed += 1
                    continue

                cropped.save(image_path, "JPEG", quality=DEFAULT_JPEG_QUALITY)
                seen_urls[url] = image_path

            metadata_path = metadata_dir / f"{image_path.stem}_metadata.json"
            if metadata_path.exists():
                # Preserve any fields added by later stages on a re-run.
                try:
                    with open(metadata_path, "r", encoding="utf-8") as fh:
                        existing = json.load(fh)
                except Exception:
                    existing = {}
            else:
                existing = {}

            existing["url"] = url
            write_json(metadata_path, existing)
            extracted += 1

        except Exception:
            # Catches truncated/corrupt images, filename-too-long, transient
            # disk errors, and any other per-row hiccup. The worker keeps going.
            failed += 1
            continue

    return ShardStats(extracted, failed)


def filter_images(
    parquet_dir: Path,
    images_dir: Path,
    metadata_dir: Path,
    *,
    min_score: float = MIN_AESTHETIC_SCORE,
    min_resolution: int = MIN_RESOLUTION,
    target_resolution: int = TARGET_RESOLUTION,
    num_workers: int = 16,
    download_timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_S,
) -> None:
    """Run Stage I across every parquet shard in ``parquet_dir``."""
    images_dir = Path(images_dir)
    metadata_dir = Path(metadata_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(Path(parquet_dir).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    print(
        f"[filter_images] {len(parquet_files)} parquet shards, "
        f"{num_workers} workers, "
        f"score>={min_score}, resolution>={min_resolution}, crop={target_resolution}"
    )

    manager = Manager()
    counter_lock = manager.Lock()
    counter_dict = manager.dict({"count": 0})

    worker = partial(
        process_shard,
        images_dir=images_dir,
        metadata_dir=metadata_dir,
        min_score=min_score,
        min_resolution=min_resolution,
        target_resolution=target_resolution,
        download_timeout=download_timeout,
        counter_lock=counter_lock,
        counter_dict=counter_dict,
    )

    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(worker, parquet_files),
                total=len(parquet_files),
                desc="Processing shards",
            )
        )

    total_extracted = sum(r.extracted for r in results)
    total_failed = sum(r.failed for r in results)
    print(
        f"[filter_images] saved {total_extracted} images "
        f"({total_failed} failed) -> {images_dir} (metadata: {metadata_dir})"
    )


def _default_num_workers() -> int:
    cpu_count = os.cpu_count() or 8
    return min(max(cpu_count * 2, 16), 32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=MIN_AESTHETIC_SCORE)
    parser.add_argument("--min-resolution", type=int, default=MIN_RESOLUTION)
    parser.add_argument("--target-resolution", type=int, default=TARGET_RESOLUTION)
    parser.add_argument("--num-workers", type=int, default=_default_num_workers())
    parser.add_argument(
        "--download-timeout", type=float, default=DEFAULT_DOWNLOAD_TIMEOUT_S
    )
    args = parser.parse_args()

    filter_images(
        parquet_dir=args.parquet_dir,
        images_dir=args.images_dir,
        metadata_dir=args.metadata_dir,
        min_score=args.min_score,
        min_resolution=args.min_resolution,
        target_resolution=args.target_resolution,
        num_workers=args.num_workers,
        download_timeout=args.download_timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
