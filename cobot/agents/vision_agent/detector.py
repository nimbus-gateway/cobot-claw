"""Vision detector abstraction.

Defines the VisionDetector protocol and two implementations:
  - GeminiDetector   : Google Gemini Robotics-ER (cloud)
  - OpenAIVisionDetector : any OpenAI-compatible vision endpoint (Ollama, vLLM, etc.)

Switch via cobot.yaml:
    agents.vision.detector: gemini          # default
    agents.vision.detector: openai_vision   # local model via llm.base_url
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable


@runtime_checkable
class VisionDetector(Protocol):
    """Minimal interface every vision backend must satisfy."""

    def locate(self, jpeg: bytes, description: str) -> dict | None:
        """Locate an object in a JPEG image.

        Returns {"point": [y, x], "box_2d": [ymin,xmin,ymax,xmax] | None, "label": str}
        with coordinates normalised 0-1000, or None if not found.
        """
        ...

    def locate_gripper(self, jpeg: bytes) -> tuple[int, int] | None:
        """Locate the robot gripper tip in a JPEG image.

        Returns (px, py) in raw image pixel coordinates, or None if not visible.
        Image dimensions must be passed separately; implementors should default to
        640×480 when unknown.
        """
        ...


# ---------------------------------------------------------------------------
# Gemini ER implementation
# ---------------------------------------------------------------------------

class GeminiDetector:
    """Wraps Google Gemini Robotics-ER for spatial object detection."""

    def __init__(self, api_key: str, model: str = "gemini-robotics-er-1.6-preview") -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model  = model

    def locate(self, jpeg: bytes, description: str) -> dict | None:
        from cobot.agents.vision_agent.gemini import gemini_locate
        return gemini_locate(self._client, jpeg, description)

    def locate_gripper(self, jpeg: bytes) -> tuple[int, int] | None:
        from cobot.agents.vision_agent.gemini import gemini_locate_gripper
        return gemini_locate_gripper(self._client, jpeg)


# ---------------------------------------------------------------------------
# OpenAI-compatible vision implementation (Ollama, vLLM, LMStudio, etc.)
# ---------------------------------------------------------------------------

class OpenAIVisionDetector:
    """Uses any OpenAI-compatible vision endpoint for spatial detection.

    The model receives the image as a base64 data URL and a structured prompt
    asking for JSON with bounding box / point coordinates.  Works with:
      - Ollama (llava, llama3.2-vision, cosmos-reason2, etc.)
      - vLLM / LMStudio serving OpenAI-compatible APIs
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key=api_key or "local")
        self._model  = model

    def _ask(self, jpeg: bytes, prompt: str) -> str | None:
        import base64
        b64 = base64.b64encode(jpeg).decode()
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=0.0,
            max_tokens=256,
        )
        return resp.choices[0].message.content if resp.choices else None

    def locate(self, jpeg: bytes, description: str) -> dict | None:
        prompt = (
            f"Locate: {description}\n"
            "Return ONLY JSON (no code fences):\n"
            "{\"box_2d\": [ymin, xmin, ymax, xmax], \"label\": \"...\"}\n"
            "Values normalised 0-1000 (integers). If not found return {\"found\": false}."
        )
        raw = self._ask(jpeg, prompt)
        if not raw:
            return None
        try:
            data = json.loads(raw.strip().strip("```json").strip("```").strip())
            if data.get("found") is False:
                return None
            if "box_2d" in data:
                ymin, xmin, ymax, xmax = data["box_2d"]
                return {
                    "point":  [(ymin + ymax) / 2, (xmin + xmax) / 2],
                    "box_2d": data["box_2d"],
                    "label":  data.get("label", description),
                }
            if "point" in data:
                return {"point": data["point"], "box_2d": None, "label": data.get("label", description)}
        except Exception:
            pass
        return None

    def locate_gripper(self, jpeg: bytes) -> tuple[int, int] | None:
        prompt = (
            "Locate the robot gripper tip (the very end of the robot arm / end-effector).\n"
            "Return ONLY JSON: {\"point\": [y, x]}\n"
            "Points normalised 0-1000. If not visible return {\"found\": false}."
        )
        raw = self._ask(jpeg, prompt)
        if not raw:
            return None
        try:
            data = json.loads(raw.strip().strip("```json").strip("```").strip())
            if data.get("found") is False:
                return None
            ny, nx = data["point"]
            return int(nx / 1000 * 640), int(ny / 1000 * 480)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_detector(cfg) -> VisionDetector | None:
    """Build a VisionDetector from VisionAgentConfig. Returns None if not configured."""
    vcfg = cfg.agents.vision
    backend = getattr(vcfg, "detector", "gemini")

    if backend == "gemini":
        import os
        key = getattr(vcfg, "google_api_key", "") or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            return None
        return GeminiDetector(api_key=key, model=getattr(vcfg, "model", "gemini-robotics-er-1.6-preview"))

    if backend == "openai_vision":
        import os
        api_key = cfg.llm.api_key or os.environ.get("OPENAI_API_KEY", "local")
        return OpenAIVisionDetector(
            base_url=cfg.llm.base_url,
            api_key=api_key,
            model=vcfg.model,
        )

    raise ValueError(f"Unknown vision detector backend: {backend!r}. Use 'gemini' or 'openai_vision'.")
