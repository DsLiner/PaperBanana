from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) <= 2:
        return stripped

    if lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_json_object(text: str) -> dict:
    cleaned = strip_markdown_fence(text)
    try:
        loaded = json.loads(cleaned)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    snippet = cleaned[start : end + 1]
    try:
        loaded = json.loads(snippet)
    except json.JSONDecodeError:
        return {}

    return loaded if isinstance(loaded, dict) else {}


def save_placeholder_diagram(description: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1920, 1080), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    try:
        header_font = ImageFont.truetype("Arial.ttf", 36)
        body_font = ImageFont.truetype("Arial.ttf", 24)
    except OSError:
        header_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    draw.rectangle(((40, 30), (1880, 1050)), outline=(220, 220, 220), width=4)
    draw.rectangle(
        ((40, 30), (1880, 110)), fill=(241, 247, 255), outline=(220, 220, 220), width=2
    )
    draw.text(
        (70, 55),
        "PaperBanana Visualizer Output (mock)",
        fill=(25, 25, 25),
        font=header_font,
    )

    wrapped_lines = textwrap.wrap(description, width=105)
    y = 150
    for line in wrapped_lines[:30]:
        draw.text((70, y), line, fill=(40, 40, 40), font=body_font)
        y += 30

    if len(wrapped_lines) > 30:
        draw.text((70, y + 10), "... (truncated)", fill=(120, 120, 120), font=body_font)

    image.save(output_path)
