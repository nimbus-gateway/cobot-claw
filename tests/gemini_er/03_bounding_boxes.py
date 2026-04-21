"""
Test 3 — Bounding boxes for all objects on the table.

More precise than points — gives the full extent of each object.
Depth is sampled at the box centre.

Output: result_03_boxes.jpg
"""
import json, os, cv2
from google import genai
from google.genai import types
from _camera import capture_frame, encode_jpeg, deproject_pixel

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

PROMPT = """
Return bounding boxes for every graspable object on the black table surface.
Include the robot arm only if its gripper/end-effector is visible.
Return ONLY a JSON array (no other text, no code fences):
[{"box_2d": [ymin, xmin, ymax, xmax], "label": "..."}]
Values normalised 0-1000 (integers only).
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
print("\nRaw:", raw)
boxes = json.loads(raw)
print(f"\nDetected {len(boxes)} objects:")

annotated = color.copy()
colors = [(0,255,0),(0,128,255),(255,0,128),(255,255,0),(0,255,255),(255,0,255)]
for i, box in enumerate(boxes):
    ymin, xmin, ymax, xmax = box["box_2d"]
    x1 = int(xmin / 1000 * w)
    y1 = int(ymin / 1000 * h)
    x2 = int(xmax / 1000 * w)
    y2 = int(ymax / 1000 * h)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    d = depth[min(cy, h-1), min(cx, w-1)] * depth_scale
    col = colors[i % len(colors)]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
    cv2.putText(annotated, f"{box['label']} {d:.2f}m",
                (x1, max(y1-8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    print(f"  {box['label']:30s}  depth={d:.3f}m  box=({x1},{y1})-({x2},{y2})")

cv2.imwrite("result_03_boxes.jpg", annotated)
print("\nSaved result_03_boxes.jpg")
