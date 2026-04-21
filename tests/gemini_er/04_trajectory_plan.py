"""
Test 4 — Ask Gemini ER to plan a pick trajectory for an object.

Gemini returns a sequence of 2D waypoints (normalised).
The D455 depth at each waypoint gives a 3D path in camera frame.

Usage:
    python 04_trajectory_plan.py "water bottle" "drop zone marker"

Output: result_04_trajectory.jpg (waypoints drawn as numbered path)
"""
import json, os, sys, cv2
from google import genai
from google.genai import types
from _camera import capture_frame, encode_jpeg, norm_to_pixel, deproject_pixel

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "object on the right"
TARGET = sys.argv[2] if len(sys.argv) > 2 else "left side of the table"
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

PROMPT = f"""
Plan a pick-and-place trajectory:
- Pick up: {SOURCE}
- Place at: {TARGET}

Return 10-15 waypoints as a numbered path from start (label "0") to end (label "N").
The gripper starts at the source object. Plan approach, grasp, lift, move, lower, release.

Return ONLY JSON (no other text):
[{{"point": [y, x], "label": "0"}}, ...]
Points normalised 0-1000.
"""

print(f"Planning trajectory: '{SOURCE}' → '{TARGET}'")
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
    config=types.GenerateContentConfig(temperature=1.0),
)

raw = response.text.strip().strip("```json").strip("```").strip()
waypoints = json.loads(raw)
print(f"\n{len(waypoints)} waypoints:")

annotated = color.copy()
prev_px = None
for wp in waypoints:
    ny, nx = wp["point"]
    px, py = norm_to_pixel(ny, nx, h, w)
    xyz = deproject_pixel(ci, depth, px, py, depth_scale)
    print(f"  Step {wp['label']:>3s}: pixel=({px},{py})  3D={xyz}")
    if prev_px:
        cv2.line(annotated, prev_px, (px, py), (0, 200, 255), 2)
    cv2.circle(annotated, (px, py), 6, (0, 200, 255), -1)
    cv2.putText(annotated, wp["label"], (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    prev_px = (px, py)

cv2.imwrite("result_04_trajectory.jpg", annotated)
print("\nSaved result_04_trajectory.jpg")
