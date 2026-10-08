"""Thin wrapper around vLLM for OverLay++ detection and captioning.

The wrapper exposes the operations the pipeline actually needs:
- ``detect``: Stage II object detection on a 1024x1024 image.
- ``validate``: Stage III per-object binary check on cropped regions.
- ``generate``: Stage IV free-form text generation (used for captions).

Each method works on batches so vLLM's scheduler can saturate the GPUs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import encode_image_base64

from overlaypp.config import VLLMConfig
from overlaypp.prompts import (
    detection_prompt,
    long_global_caption_prompt,
    short_global_caption_prompt,
    short_local_prompts_prompt,
    validation_prompt,
)
from overlaypp.utils import clamp_bbox, extract_json

ImageLike = Image.Image | str | Path


def _to_rgb(image: ImageLike) -> Image.Image:
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    else:
        img = image
    return img if img.mode == "RGB" else img.convert("RGB")


def _image_message(prompt: str, image: Image.Image) -> list[dict[str, Any]]:
    """Build a single chat message containing a text prompt and an image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image_base64(image)}"
                    },
                },
            ],
        }
    ]


class VLLMDetector:
    """Detection + captioning over a Qwen-VL style model served by vLLM."""

    def __init__(self, config: VLLMConfig | None = None) -> None:
        self.config = config or VLLMConfig()
        tensor_parallel_size = self.config.tensor_parallel_size
        if tensor_parallel_size is None:
            tensor_parallel_size = max(1, torch.cuda.device_count())

        print(
            f"[VLLMDetector] loading {self.config.model_name} on "
            f"{tensor_parallel_size} GPU(s)..."
        )
        self.llm = LLM(
            model=self.config.model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            max_num_seqs=self.config.max_num_seqs,
            trust_remote_code=self.config.trust_remote_code,
            download_dir=self.config.download_dir,
        )

    def _chat(
        self,
        messages: Sequence[list[dict[str, Any]]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        if not messages:
            return []
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.llm.chat(list(messages), sampling_params=params)
        return [out.outputs[0].text for out in outputs]

    def detect(
        self,
        images: Sequence[ImageLike],
        *,
        resolution: int = 1024,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> list[dict[str, Any]]:
        """Stage II: detect objects on each image, returning parsed JSON dicts."""
        rgb_images = [_to_rgb(img) for img in images]
        messages = []
        for img in rgb_images:
            width, height = img.size
            prompt = detection_prompt(width, height, resolution)
            messages.append(_image_message(prompt, img))
        texts = self._chat(messages, temperature=temperature, max_tokens=max_tokens)
        return [extract_json(text) for text in texts]

    def validate(
        self,
        image: ImageLike,
        detections: dict[str, Any],
        *,
        temperature: float = 0.0,
        max_tokens: int = 10,
    ) -> dict[str, Any]:
        """Stage III: keep only detections whose cropped region answers 'yes'."""
        img = _to_rgb(image)
        objects = detections.get("detections", []) or []
        if not objects:
            return {"detections": []}

        width, height = img.size
        messages: list[list[dict[str, Any]]] = []
        kept_indices: list[int] = []

        for idx, obj in enumerate(objects):
            if not isinstance(obj, dict):
                continue
            clamped = clamp_bbox(obj.get("bbox", []), width, height)
            if clamped is None:
                continue
            x1, y1, x2, y2 = clamped
            crop = img.crop((x1, y1, x2, y2))
            prompt = validation_prompt(
                category=str(obj.get("category", f"Object_{idx + 1}")),
                local_prompt=str(obj.get("local_prompt", "")),
            )
            messages.append(_image_message(prompt, crop))
            kept_indices.append(idx)

        responses = self._chat(messages, temperature=temperature, max_tokens=max_tokens)
        validated: list[dict[str, Any]] = []
        for idx, response in zip(kept_indices, responses):
            if response.strip().lower().startswith("yes"):
                validated.append(objects[idx])
        return {"detections": validated}

    def generate(
        self,
        images: Sequence[ImageLike],
        prompts: Sequence[str],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> list[str]:
        """Generic batched generation: one prompt + one image per request."""
        if len(images) != len(prompts):
            raise ValueError("images and prompts must be the same length")
        messages = [
            _image_message(prompt, _to_rgb(img)) for img, prompt in zip(images, prompts)
        ]
        texts = self._chat(messages, temperature=temperature, max_tokens=max_tokens)
        return [text.strip() for text in texts]

    def long_global_captions(
        self,
        images: Sequence[ImageLike],
        *,
        max_tokens: int = 256,
    ) -> list[str]:
        prompt = long_global_caption_prompt()
        return self.generate(
            images,
            [prompt] * len(images),
            max_tokens=max_tokens,
        )

    def short_global_captions(
        self,
        images: Sequence[ImageLike],
        *,
        max_tokens: int = 128,
    ) -> list[str]:
        prompt = short_global_caption_prompt()
        return self.generate(
            images,
            [prompt] * len(images),
            max_tokens=max_tokens,
        )

    def short_local_prompts(
        self,
        images: Sequence[ImageLike],
        detections_per_image: Sequence[list[dict[str, Any]]],
        *,
        max_tokens: int = 512,
    ) -> list[list[str]]:
        """Return, for each input image, a list of short local prompts (one per detection)."""
        if len(images) != len(detections_per_image):
            raise ValueError("images and detections_per_image must be the same length")
        prompts = [short_local_prompts_prompt(d) for d in detections_per_image]
        raw_texts = self.generate(images, prompts, max_tokens=max_tokens)

        results: list[list[str]] = []
        for raw, detections in zip(raw_texts, detections_per_image):
            parsed = extract_json(raw).get("detections", [])
            row: list[str] = []
            for idx in range(len(detections)):
                if idx < len(parsed) and isinstance(parsed[idx], dict):
                    row.append(str(parsed[idx].get("short_local_prompt", "")).strip())
                else:
                    row.append("")
            results.append(row)
        return results
