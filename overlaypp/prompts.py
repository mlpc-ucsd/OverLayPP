"""Prompt templates used by the OverLay++ pipeline.

Centralising every prompt in one place makes it easy to audit what the VLM is
being asked to do and to keep wording consistent across the pipeline stages.
"""

from __future__ import annotations

import json
from typing import Any

from overlaypp.config import MAX_SHORT_CAPTION_WORDS, MIN_BBOX_FRACTION


def detection_prompt(image_width: int, image_height: int, resolution: int) -> str:
    """Prompt for Stage II: detect all objects with overlap and size constraints.

    Character-for-character identical to the original
    ``VLLMObjectDetector._create_detection_prompt`` (with ``MIN_BBOX_FRACTION``
    factored out so the threshold is configurable in one place).
    """
    return (
        f"Analyze this image and detect ALL objects that are present in the image. "
        "For each detected object, provide:\n"
        "1. The object name/category\n"
        "2. A bounding box [x1, y1, x2, y2] indicating where the object is located in the image\n"
        "3. A detailed local_prompt description that describes the object's appearance, attributes, style, and visual characteristics\n\n"
        "IMPORTANT REQUIREMENTS:\n"
        "- Detect objects that are ACTUALLY present in the image\n"
        "- CRITICAL: Each object's bounding box MUST have some overlap area with at least one other object's bounding box\n"
        "- Bounding boxes must NOT be very small - each bounding box should have a minimum width and height of at least "
        f"{int(MIN_BBOX_FRACTION * 100)}% of the image resolution\n"
        f"- Minimum bounding box dimensions: width >= {resolution * MIN_BBOX_FRACTION:.0f} pixels, "
        f"height >= {resolution * MIN_BBOX_FRACTION:.0f} pixels\n"
        "- Overlapping means the bounding boxes should share at least some common area (intersection over union > 0 or at least partial overlap)\n"
        "- Include various types of objects: foreground objects, background objects, parts of larger objects, etc.\n"
        "- The local_prompt should be a detailed text description that includes:\n"
        "  * Visual appearance\n"
        "  * Size and proportions\n"
        "  * Style and design details\n"
        "  * Any specific attributes or features\n"
        "  * Contextual details relevant to the scene\n\n"
        "Response in JSON format:\n"
        "{\n"
        '  "detections": [\n'
        "    {\n"
        '      "category": "object_category",\n'
        '      "bbox": [x1, y1, x2, y2],\n'
        '      "local_prompt": "detailed description of the object\'s appearance, style, and visual characteristics"\n'
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n\n"
        f"The bounding box coordinates should be in the range [0, {resolution}] where (0,0) is the top-left corner.\n"
        f"The image dimensions are: width={image_width}, height={image_height}.\n"
        "CRITICAL: Ensure all bounding boxes have overlapping areas with at least one other bounding box.\n"
        "CRITICAL: Do not create very small bounding boxes.\n"
        "Strictly follow the JSON format without any additional text."
    )


def validation_prompt(category: str, local_prompt: str) -> str:
    """Prompt for Stage III: binary yes/no check on a cropped bounding-box region.

    Matches the original ``VLLMObjectDetector.validate_detections_batch``
    construction: ``"{category}. {local_prompt}"`` when ``local_prompt`` is
    truthy, otherwise just ``category``.
    """
    object_description = f"{category}. {local_prompt}" if local_prompt else category
    return (
        "Look at this cropped image region. Is the following object present in this image?\n\n"
        f"Object: {object_description}\n\n"
        "Answer with only 'yes' or 'no'. No additional explanation."
    )


def long_global_caption_prompt() -> str:
    """Prompt for Stage IV: long scene-level caption."""
    return (
        "Describe this image in one or two sentences, including the main objects, "
        "activities, and setting."
    )


def short_global_caption_prompt() -> str:
    """Prompt for Stage IV: short (<20 words) scene-level caption."""
    return (
        "Describe this image with a short global caption.\n"
        "Requirements:\n"
        "- The caption must summarize the overall scene.\n"
        "- Include the key object(s) and setting.\n"
        f"- Keep it under {MAX_SHORT_CAPTION_WORDS} words.\n"
        "- Return only the caption text (no quotes, labels, or extra explanation)."
    )


def short_local_prompts_prompt(detections: list[dict[str, Any]]) -> str:
    """Prompt for Stage IV: produce short (<20 words) object-level captions.

    The VLM sees the full image plus the existing per-object metadata, and emits
    one short_local_prompt for each detection in the same order.
    """
    return (
        "You are given a full image and object detections.\n"
        "Use the image content to answer (not detections text alone).\n"
        "For each detection, return:\n"
        f"1) short_local_prompt: less than {MAX_SHORT_CAPTION_WORDS} words\n\n"
        "IMPORTANT:\n"
        "- Keep the same order and number of detections.\n"
        "- Each short_local_prompt must include color and any other visual attributes.\n"
        f"- Each short_local_prompt must be less than {MAX_SHORT_CAPTION_WORDS} words.\n"
        "- Return only JSON in this format:\n"
        "{\n"
        '  "detections": [\n'
        '    {"short_local_prompt": "..."},\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        f"Detections input:\n{json.dumps(detections, ensure_ascii=True)}"
    )
