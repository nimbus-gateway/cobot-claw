"""
Test 2 — Point to a specific object by natural language description.

Usage:
    python 02_point_to_object.py "water bottle"
    python 02_point_to_object.py "the object on the right"
    python 02_point_to_object.py "blue lotion bottle"

Output: result_02_point.jpg + 3D position in camera frame
"""
import json, os, sys, cv2
from google import genai
from google.genai import types
from _camera import capture_frame, encode_jpeg, norm_to_pixel, deproject_pixel

TARGET = sys.argv[1] if len(sys.argv) > 1 else "water bottle"
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

PROMPT = f"""
Point to: {TARGET}
Return ONLY a JSON array (no other text):
[{{"point": [y, x], "label": "..."}}]
Points normalised 0-1000. If not found return [].
"""

print(f"Looking for: '{TARGET}'")
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
detections = json.loads(raw)
annotated = color.copy()

if not detections:
    print(f"'{TARGET}' not found in scene.")
else:
    for det in detections:
        ny, nx = det["point"]
        px, py = norm_to_pixel(ny, nx, h, w)
        xyz = deproject_pixel(ci, depth, px, py, depth_scale)
        print(f"\nFound: {det['label']}")
        print(f"  Pixel     : ({px}, {py})")
        print(f"  Camera XYZ: {xyz}  (metres)")
        cv2.circle(annotated, (px, py), 12, (0, 0, 255), -1)
        cv2.circle(annotated, (px, py), 14, (255, 255, 255), 2)
        depth_m = depth[py, px] * depth_scale
        cv2.putText(annotated, f"{det['label']} ({depth_m:.2f}m)",
                    (px + 15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imwrite("result_02_point.jpg", annotated)
print("\nSaved result_02_point.jpg")
