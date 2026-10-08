"""Stage II: Object detection.

Run the Qwen3-VL detector on every cropped image produced by Stage I and merge
the raw detections (category + bounding box + long ``local_prompt``) into the
per-image metadata file written by Stage I (which already contains the
source ``url``).

Scenes whose initial detection count falls below ``min_detections`` are
discarded immediately.

This corresponds to stage II in the OverLay++ pipeline figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from overlaypp.config import (
    DEFAULT_MODEL_NAME,
    MIN_VALIDATED_OBJECTS,
    TARGET_RESOLUTION,
    VLLMConfig,
)
from overlaypp.detector import VLLMDetector
from overlaypp.utils import iter_image_paths, load_image_rgb, read_json, write_json


def _metadata_path(metadata_dir: Path, image_path: Path) -> Path:
    return metadata_dir / f"{image_path.stem}_metadata.json"


def _load_metadata(metadata_dir: Path, image_path: Path) -> dict:
    """Load the metadata file written by Stage I, or start a fresh dict."""
    meta_path = _metadata_path(metadata_dir, image_path)
    if not meta_path.exists():
        return {}
    try:
        return read_json(meta_path)
    except Exception:
        return {}


def _discard(image_path: Path, meta_path: Path, *, keep_images: bool) -> None:
    """Delete the metadata file and (optionally) the image of a discarded scene."""
    try:
        meta_path.unlink()
    except OSError:
        pass
    if not keep_images:
        try:
            image_path.unlink()
        except OSError:
            pass


def detect_objects(
    images_dir: Path,
    metadata_dir: Path,
    *,
    detector: VLLMDetector,
    resolution: int = TARGET_RESOLUTION,
    batch_size: int = 16,
    min_detections: int = MIN_VALIDATED_OBJECTS,
    skip_existing: bool = True,
    keep_images: bool = False,
) -> None:
    """Detect objects on every image under ``images_dir``.

    For each image we either:
    - add a ``detections`` field to its metadata file (when the detector
      returns ``>= min_detections`` boxes), or
    - delete both the metadata file and the image (when it returns fewer),
      matching the original pipeline's early-exit behaviour.

    Pass ``keep_images=True`` to retain the rejected image file for
    inspection; the metadata file is still removed.
    """
    images_dir = Path(images_dir)
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(iter_image_paths(images_dir))
    if skip_existing:
        image_paths = [
            p
            for p in image_paths
            if "detections" not in _load_metadata(metadata_dir, p)
        ]

    if not image_paths:
        print("[detect_objects] no images to process")
        return

    print(
        f"[detect_objects] {len(image_paths)} images, "
        f"batch_size={batch_size}, resolution={resolution}, "
        f"min_detections={min_detections}"
    )

    saved = 0
    discarded = 0
    pbar = tqdm(range(0, len(image_paths), batch_size), desc="Detecting")
    for batch_start in pbar:
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        batch_images = []
        valid_paths: list[Path] = []
        for path in batch_paths:
            try:
                batch_images.append(load_image_rgb(path))
                valid_paths.append(path)
            except Exception as exc:
                print(f"\n[detect_objects] error loading {path.name}: {exc}")

        if not batch_images:
            continue

        try:
            detections_batch = detector.detect(batch_images, resolution=resolution)
        except Exception as exc:
            print(f"\n[detect_objects] batch detection failed: {exc}")
            continue

        for path, detections in zip(valid_paths, detections_batch):
            detection_list = detections.get("detections", []) or []
            meta_path = _metadata_path(metadata_dir, path)

            if len(detection_list) < min_detections:
                _discard(path, meta_path, keep_images=keep_images)
                discarded += 1
                continue

            metadata = _load_metadata(metadata_dir, path)
            metadata["detections"] = detection_list
            write_json(meta_path, metadata)
            saved += 1

        pbar.set_postfix({"saved": saved, "discarded": discarded})

    print(
        f"[detect_objects] saved {saved}, discarded {discarded} "
        f"(metadata dir: {metadata_dir})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--resolution", type=int, default=TARGET_RESOLUTION)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--min-detections",
        type=int,
        default=MIN_VALIDATED_OBJECTS,
        help=(
            "Discard scenes whose initial detection count is below this "
            "threshold (default matches --min-validated-objects in Stage III)."
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--cache-dir", default=None, help="HuggingFace cache directory."
    )
    parser.add_argument("--no-skip-existing", action="store_true")
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

    detect_objects(
        images_dir=args.images_dir,
        metadata_dir=args.metadata_dir,
        detector=detector,
        resolution=args.resolution,
        batch_size=args.batch_size,
        min_detections=args.min_detections,
        skip_existing=not args.no_skip_existing,
        keep_images=args.keep_images,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
