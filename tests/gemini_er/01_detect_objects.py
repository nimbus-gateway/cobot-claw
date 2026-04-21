"""
Test 1 — Detect all objects on the robot table.

Asks Gemini ER to find and label everything it sees.
No ArUco markers needed — pure vision.

Output: result_01_detect.jpg  (annotated with dots + labels)
"""
import json, os, cv2
from google import genai
from google.genai import types
from _camera import capture_frame, encode_jpeg, norm_to_pixel

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

PROMPT = """
You are looking at a robot lab workbench. Point to ALL objects you can see,
especially on the black table surface.
The label should be a short identifying name (e.g. "water bottle", "blue block", "gripper").
Return ONLY a JSON array, no other text:
[{"point": [y, x], "label": "..."}]
Points are normalised 0-1000. y is vertical (0=top), x is horizontal (0=left).
"""

print("Capturing frame...")
color, depth, ci, depth_scale = capture_frame()
h, w = color.shape[:2]

print("Sending to Gemini ER...")
response = client.models.generate_content(
    model="gemini-robotics-er-1.6-preview",
    contents=[
        types.Part.from_bytes(data=encode_jpeg(color), mime_type="image/jpeg"),
        PROMPT,
    ],
    config=types.GenerateContentConfig(
        temperature=1.0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
)

raw = response.text.strip().strip("```json").strip("```").strip()
print("\nRaw response:", raw)
detections = json.loads(raw)
print(f"\nDetected {len(detections)} objects:")

annotated = color.copy()
for det in detections:
    ny, nx = det["point"]
    px, py = norm_to_pixel(ny, nx, h, w)
    label = det["label"]
    d = depth[py, px] * depth_scale
    print(f"  {label:30s}  pixel=({px},{py})  depth={d:.3f}m")
    cv2.circle(annotated, (px, py), 8, (0, 255, 0), -1)
    cv2.putText(annotated, f"{label} ({d:.2f}m)", (px + 10, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

cv2.imwrite("result_01_detect.jpg", annotated)
print("\nSaved result_01_detect.jpg")
