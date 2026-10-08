"""Stage IV: Hierarchical captioning.

For every surviving scene we add three fields to the metadata file:

- ``global_caption``        -- long scene description (1-2 sentences).
- ``short_global_caption``  -- short scene description (< 20 words).
- ``short_local_prompt``    -- short object description, added to each detection.

The long ``local_prompt`` for each detection is produced earlier in Stage II,
so this stage only fills in the missing pieces.

After captioning, a cleanup pass removes any record that is missing one of the
fields the final dataset requires (``url``, ``detections`` with full per-object
metadata, ``global_caption``, ``short_global_caption``).

This corresponds to stage IV in the OverLay++ pipeline figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from overlaypp.config import DEFAULT_MODEL_NAME, VLLMConfig
from overlaypp.detector import VLLMDetector
from overlaypp.utils import (
    find_image_for_stem,
    is_non_empty_str,
    load_image_rgb,
    read_json,
    write_json,
)


def _missing_global(metadata: dict[str, Any]) -> bool:
    return not is_non_empty_str(metadata.get("global_caption"))


def _missing_short_global(metadata: dict[str, Any]) -> bool:
    return not is_non_empty_str(metadata.get("short_global_caption"))


def _missing_short_local(metadata: dict[str, Any]) -> bool:
    detections = metadata.get("detections") or []
    return any(
        isinstance(det, dict) and not is_non_empty_str(det.get("short_local_prompt"))
        for det in detections
    )


def _batched(items: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def add_long_global_captions(
    metadata_files: list[Path],
    images_dir: Path,
    detector: VLLMDetector,
    batch_size: int,
) -> int:
    pending: list[tuple[Path, dict, Path]] = []
    for meta_path in metadata_files:
        try:
            metadata = read_json(meta_path)
        except Exception:
            continue
        if not _missing_global(metadata):
            continue
        image_path = find_image_for_stem(
            images_dir, meta_path.stem.replace("_metadata", "")
        )
        if image_path is None:
            continue
        pending.append((meta_path, metadata, image_path))

    if not pending:
        return 0

    updated = 0
    pbar = tqdm(
        _batched(pending, batch_size),
        total=(len(pending) + batch_size - 1) // batch_size,
        desc="Long global captions",
    )
    for batch in pbar:
        images = [load_image_rgb(image_path) for _, _, image_path in batch]
        captions = detector.long_global_captions(images)
        for (meta_path, metadata, _), caption in zip(batch, captions):
            if not is_non_empty_str(caption):
                continue
            metadata["global_caption"] = caption.strip()
            write_json(meta_path, metadata)
            updated += 1
        pbar.set_postfix({"updated": updated})
    return updated


def add_short_global_captions(
    metadata_files: list[Path],
    images_dir: Path,
    detector: VLLMDetector,
    batch_size: int,
) -> int:
    pending: list[tuple[Path, dict, Path]] = []
    for meta_path in metadata_files:
        try:
            metadata = read_json(meta_path)
        except Exception:
            continue
        if not _missing_short_global(metadata):
            continue
        image_path = find_image_for_stem(
            images_dir, meta_path.stem.replace("_metadata", "")
        )
        if image_path is None:
            continue
        pending.append((meta_path, metadata, image_path))

    if not pending:
        return 0

    updated = 0
    pbar = tqdm(
        _batched(pending, batch_size),
        total=(len(pending) + batch_size - 1) // batch_size,
        desc="Short global captions",
    )
    for batch in pbar:
        images = [load_image_rgb(image_path) for _, _, image_path in batch]
        captions = detector.short_global_captions(images)
        for (meta_path, metadata, _), caption in zip(batch, captions):
            if not is_non_empty_str(caption):
                continue
            metadata["short_global_caption"] = caption.strip()
            write_json(meta_path, metadata)
            updated += 1
        pbar.set_postfix({"updated": updated})
    return updated


def add_short_local_prompts(
    metadata_files: list[Path],
    images_dir: Path,
    detector: VLLMDetector,
    batch_size: int,
) -> int:
    pending: list[tuple[Path, dict, Path]] = []
    for meta_path in metadata_files:
        try:
            metadata = read_json(meta_path)
        except Exception:
            continue
        if not _missing_short_local(metadata):
            continue
        detections = metadata.get("detections") or []
        if not detections:
            continue
        image_path = find_image_for_stem(
            images_dir, meta_path.stem.replace("_metadata", "")
        )
        if image_path is None:
            continue
        pending.append((meta_path, metadata, image_path))

    if not pending:
        return 0

    updated = 0
    pbar = tqdm(
        _batched(pending, batch_size),
        total=(len(pending) + batch_size - 1) // batch_size,
        desc="Short local prompts",
    )
    for batch in pbar:
        images = [load_image_rgb(image_path) for _, _, image_path in batch]
        detections_batch = [metadata["detections"] for _, metadata, _ in batch]
        results = detector.short_local_prompts(images, detections_batch)

        for (meta_path, metadata, _), short_prompts in zip(batch, results):
            detections = metadata.get("detections") or []
            for det, short_prompt in zip(detections, short_prompts):
                if isinstance(det, dict) and is_non_empty_str(short_prompt):
                    det["short_local_prompt"] = short_prompt
            metadata["detections"] = detections
            write_json(meta_path, metadata)
            updated += 1
        pbar.set_postfix({"updated": updated})
    return updated


def _is_complete(metadata: dict[str, Any]) -> bool:
    """Return True iff the record has every field required by the final dataset."""
    if not is_non_empty_str(metadata.get("url")):
        return False
    if not is_non_empty_str(metadata.get("global_caption")):
        return False
    if not is_non_empty_str(metadata.get("short_global_caption")):
        return False
    detections = metadata.get("detections")
    if not isinstance(detections, list) or not detections:
        return False
    for det in detections:
        if not isinstance(det, dict):
            return False
        if not is_non_empty_str(det.get("category")):
            return False
        bbox = det.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        if not is_non_empty_str(det.get("local_prompt")):
            return False
        if not is_non_empty_str(det.get("short_local_prompt")):
            return False
    return True


def cleanup_incomplete_records(
    images_dir: Path,
    metadata_dir: Path,
    *,
    keep_images: bool = False,
) -> int:
    """Delete metadata files (and images) for any record missing required fields.

    Returns the number of records removed.
    """
    images_dir = Path(images_dir)
    metadata_dir = Path(metadata_dir)

    removed = 0
    for meta_path in sorted(metadata_dir.glob("*_metadata.json")):
        try:
            metadata = read_json(meta_path)
        except Exception:
            metadata = {}

        if _is_complete(metadata):
            continue

        stem = meta_path.stem.replace("_metadata", "")
        image_path = find_image_for_stem(images_dir, stem)

        try:
            meta_path.unlink()
        except OSError:
            pass
        if not keep_images and image_path is not None:
            try:
                image_path.unlink()
            except OSError:
                pass
        removed += 1
    return removed


def caption_scenes(
    images_dir: Path,
    metadata_dir: Path,
    *,
    detector: VLLMDetector,
    batch_size: int = 16,
    cleanup: bool = True,
    keep_images: bool = False,
) -> None:
    """Add long/short global captions and short local prompts to every scene.

    When ``cleanup`` is True (the default), any record still missing a
    required field after captioning is removed -- the corresponding image is
    deleted too unless ``keep_images`` is set. After this call the working
    directory contains only fully-formed records.
    """
    images_dir = Path(images_dir)
    metadata_dir = Path(metadata_dir)

    metadata_files = sorted(metadata_dir.glob("*_metadata.json"))
    if not metadata_files:
        print("[caption] no metadata files to process")
        return

    print(f"[caption] {len(metadata_files)} scenes, batch_size={batch_size}")

    long_updated = add_long_global_captions(
        metadata_files, images_dir, detector, batch_size
    )
    short_updated = add_short_global_captions(
        metadata_files, images_dir, detector, batch_size
    )
    local_updated = add_short_local_prompts(
        metadata_files, images_dir, detector, batch_size
    )

    print(
        f"[caption] global_caption updated: {long_updated}, "
        f"short_global_caption updated: {short_updated}, "
        f"short_local_prompt updated: {local_updated}"
    )

    if cleanup:
        removed = cleanup_incomplete_records(
            images_dir=images_dir,
            metadata_dir=metadata_dir,
            keep_images=keep_images,
        )
        print(f"[caption] cleanup removed {removed} incomplete record(s)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help=(
            "Skip the final cleanup pass that removes records missing any "
            "required field after captioning."
        ),
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help=(
            "During the cleanup pass, keep image files for removed records "
            "(useful for inspecting what was discarded). By default the image "
            "is deleted along with its metadata."
        ),
    )
    args = parser.parse_args()

    detector = VLLMDetector(
        VLLMConfig(
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            download_dir=args.cache_dir,
        )
    )

    caption_scenes(
        images_dir=args.images_dir,
        metadata_dir=args.metadata_dir,
        detector=detector,
        batch_size=args.batch_size,
        cleanup=not args.no_cleanup,
        keep_images=args.keep_images,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
