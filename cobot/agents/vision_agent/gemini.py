"""Gemini Robotics-ER helpers for object detection and gripper localisation."""

from __future__ import annotations

import json

from google import genai
from google.genai import types as gtypes


def gemini_locate_box(client: genai.Client, jpeg: bytes, description: str) -> dict | None:
    """Ask Gemini ER for a bounding box around an object.

    Returns {"box_2d": [ymin, xmin, ymax, xmax], "point": [cy, cx], "label": "..."}
    with coords normalised 0-1000, or None if not found.
    The "point" is the box centre — drop-in replacement for gemini_locate output.
    """
    prompt = (
        f"Locate: {description}\n"
        "Return ONLY JSON (no code fences):\n"
        "{\"box_2d\": [ymin, xmin, ymax, xmax], \"label\": \"...\"}\n"
        "Values normalised 0-1000 (integers). If not found return {\"found\": false}."
    )
    resp = client.models.generate_content(
        model="gemini-robotics-er-1.6-preview",
        contents=[
            gtypes.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
            prompt,
        ],
        config=gtypes.GenerateContentConfig(
            temperature=1.0,
            thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = resp.text.strip().strip("```json").strip("```").strip()
    try:
        data = json.loads(raw)
        if data.get("found") is False:
            return None
        ymin, xmin, ymax, xmax = data["box_2d"]
        cy = (ymin + ymax) / 2
        cx = (xmin + xmax) / 2
        return {
            "point":  [cy, cx],
            "box_2d": data["box_2d"],
            "label":  data.get("label", description),
        }
    except Exception:
        return None


def gemini_locate(client: genai.Client, jpeg: bytes, description: str) -> dict | None:
    """Ask Gemini ER to locate an object by natural-language description.

    Tries bounding-box detection first (more accurate centre + depth region);
    falls back to single-point detection if the box request fails.

    Returns {"point": [y, x], "box_2d": [...] | None, "label": "..."}
    with coords normalised 0-1000, or None if the object is not found.
    """
    det = gemini_locate_box(client, jpeg, description)
    if det is not None:
        return det

    # Fallback: single-point prompt
    prompt = (
        f"Locate: {description}\n"
        "Return ONLY JSON: {\"point\": [y, x], \"label\": \"...\"}\n"
        "Points normalised 0-1000. If not found return {\"found\": false}."
    )
    resp = client.models.generate_content(
        model="gemini-robotics-er-1.6-preview",
        contents=[
            gtypes.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
            prompt,
        ],
        config=gtypes.GenerateContentConfig(
            temperature=1.0,
            thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = resp.text.strip().strip("```json").strip("```").strip()
    try:
        data = json.loads(raw)
        if data.get("found") is False:
            return None
        return {**data, "box_2d": None}
    except Exception:
        return None


def gemini_locate_gripper(client: genai.Client, jpeg: bytes) -> tuple[int, int] | None:
    """Ask Gemini ER to locate the robot gripper tip (pixel coords for calibration).

    Returns (px, py) in image coordinates, or None if not visible.
    """
    prompt = (
        "Locate the robot gripper tip (the very end of the robot arm / end-effector).\n"
        "Return ONLY JSON: {\"point\": [y, x]}\n"
        "Points normalised 0-1000. If not visible return {\"found\": false}."
    )
    resp = client.models.generate_content(
        model="gemini-robotics-er-1.6-preview",
        contents=[
            gtypes.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
            prompt,
        ],
        config=gtypes.GenerateContentConfig(
            temperature=1.0,
            thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = resp.text.strip().strip("```json").strip("```").strip()
    try:
        data = json.loads(raw)
        if data.get("found") is False:
            return None
        ny, nx = data["point"]
        return int(nx / 1000 * 640), int(ny / 1000 * 480)
    except Exception:
        return None
