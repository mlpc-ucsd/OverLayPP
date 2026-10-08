"""Stage III: Scene filtering.

For every image with raw detections from Stage II, ask the VLM a binary yes/no
question on each cropped bounding box. Keep only detections that answer "yes",
and discard the whole scene if fewer than ``MIN_VALIDATED_OBJECTS`` survive.

This corresponds to stage III in the OverLay++ pipeline figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from overlaypp.config import (
    DEFAULT_MODEL_NAME,
    MIN_VALIDATED_OBJECTS,
    VLLMConfig,
)
from overlaypp.detector import VLLMDetector
from overlaypp.utils import (
    find_image_for_stem,
    load_image_rgb,
    read_json,
    write_json,
)


def filter_scenes(
    images_dir: Path,
    metadata_dir: Path,
    *,
    detector: VLLMDetector,
    min_validated_objects: int = MIN_VALIDATED_OBJECTS,
    keep_images: bool = False,
) -> None:
    """Validate every metadata file in place and drop scenes with too few detections.

    When a scene is discarded (fewer than ``min_validated_objects`` survive the
    per-object yes/no check), both its metadata file and its image are removed
    so the on-disk dataset never contains orphans. Pass ``keep_images=True``
    to keep the rejected image file (the metadata is still removed).
    """
    images_dir = Path(images_dir)
    metadata_dir = Path(metadata_dir)

    metadata_files = sorted(metadata_dir.glob("*_metadata.json"))
    if not metadata_files:
        print("[filter_scenes] no metadata files to process")
        return

    print(
        f"[filter_scenes] {len(metadata_files)} scenes, "
        f"keeping those with >= {min_validated_objects} validated objects"
    )

    kept = 0
    discarded = 0
    errors = 0

    pbar = tqdm(metadata_files, desc="Validating")
    for meta_path in pbar:
        stem = meta_path.stem.replace("_metadata", "")
        image_path = find_image_for_stem(images_dir, stem)
        if image_path is None:
            errors += 1
            continue

        try:
            metadata = read_json(meta_path)
            image = load_image_rgb(image_path)
            validated = detector.validate(image, metadata)
        except Exception as exc:
            print(f"\n[filter_scenes] error on {meta_path.name}: {exc}")
            errors += 1
            continue

        kept_count = len(validated.get("detections", []))
        if kept_count < min_validated_objects:
            discarded += 1
            try:
                meta_path.unlink()
            except OSError:
                pass
            if not keep_images:
                try:
                    image_path.unlink()
                except OSError:
                    pass
            continue

        metadata["detections"] = validated["detections"]
        write_json(meta_path, metadata)
        kept += 1

        pbar.set_postfix({"kept": kept, "discarded": discarded, "errors": errors})

    print(
        f"[filter_scenes] kept {kept}, discarded {discarded}, "
        f"errors {errors} (metadata dir: {metadata_dir})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--min-validated-objects",
        type=int,
        default=MIN_VALIDATED_OBJECTS,
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help=(
            "Keep the image file on disk when a scene is discarded "
            "(useful for inspecting rejected images). By default both the "
            "metadata file and the image are deleted together."
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

    filter_scenes(
        images_dir=args.images_dir,
        metadata_dir=args.metadata_dir,
        detector=detector,
        min_validated_objects=args.min_validated_objects,
        keep_images=args.keep_images,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
