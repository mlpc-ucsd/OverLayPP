"""Default configuration values for the OverLay++ pipeline.

All thresholds, model names, and tunable knobs live here so that the pipeline
stages can share a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODEL_NAME: str = "Qwen/Qwen3-VL-32B-Instruct"

# Stage I -- image filtering
MIN_AESTHETIC_SCORE: float = 5.98
MIN_RESOLUTION: int = 1024
TARGET_RESOLUTION: int = 1024

# Stage III -- scene filtering
MIN_VALIDATED_OBJECTS: int = 5

# Detection constraints
MIN_BBOX_FRACTION: float = 0.05

# Captioning
MAX_SHORT_CAPTION_WORDS: int = 20


@dataclass(frozen=True)
class VLLMConfig:
    """Runtime configuration for the underlying vLLM inference engine."""

    model_name: str = DEFAULT_MODEL_NAME
    tensor_parallel_size: int | None = None  # None -> use all visible GPUs
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    max_num_seqs: int = 256
    trust_remote_code: bool = True
    download_dir: str | None = None
