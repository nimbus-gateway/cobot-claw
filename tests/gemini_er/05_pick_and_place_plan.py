"""
Test 5 — Full pick-and-place function call plan (closest to real robot use).

Gemini ER:
1. Finds the source and target objects → gets their pixel coords
2. Converts to normalised robot coords (relative to robot origin)
3. Returns a JSON sequence of robot function calls

This is the pattern closest to how you'd wire Gemini ER into cobot-claw.
No actual robot movement — just the plan printed to console.

Usage:
    python 05_pick_and_place_plan.py "water bottle" "left side of table"
"""
import json, os, sys, cv2
from google import genai
from google.genai import types
from _camera import capture_frame, encode_jpeg, norm_to_pixel, deproject_pixel

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "object on the right"
TARGET = sys.argv[2] if len(sys.argv) > 2 else "object on the left"
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# ---- Step 1: locate source and target ----
LOCATE_PROMPT = f"""
Locate these two items and return their pixel coordinates:
1. Source (pick): {SOURCE}
2. Target (place): {TARGET}

Return ONLY JSON (no other text):
[{{"point": [y, x], "label": "source"}}, {{"point": [y, x], "label": "target"}}]
Points normalised 0-1000.
"""

print(f"Locating objects: '{SOURCE}' → '{TARGET}'")
print("Capturing frame...")
color, depth, ci, depth_scale = capture_frame()
h, w = color.shape[:2]

print("Step 1 — Locating objects...")
loc_response = client.models.generate_content(
    model="gemini-robotics-er-1.6-preview",
    contents=[
        types.Part.from_bytes(data=encode_jpeg(color), mime_type="image/jpeg"),
        LOCATE_PROMPT,
    ],
    config=types.GenerateContentConfig(
        temperature=1.0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
)

raw = loc_response.text.strip().strip("```json").strip("```").strip()
locations = json.loads(raw)

# Extract source / target
source_pt = next((l for l in locations if l["label"] == "source"), None)
target_pt = next((l for l in locations if l["label"] == "target"), None)

if not source_pt or not target_pt:
    print("Could not locate both objects. Raw response:", raw)
    raise SystemExit(1)

src_px, src_py = norm_to_pixel(*source_pt["point"], h, w)
tgt_px, tgt_py = norm_to_pixel(*target_pt["point"], h, w)
src_xyz = deproject_pixel(ci, depth, src_px, src_py, depth_scale)
tgt_xyz = deproject_pixel(ci, depth, tgt_px, tgt_py, depth_scale)

print(f"\n  Source '{SOURCE}': pixel=({src_px},{src_py})  3D={src_xyz}")
print(f"  Target '{TARGET}': pixel=({tgt_px},{tgt_py})  3D={tgt_xyz}")

# ---- Step 2: generate function call plan ----
PLAN_PROMPT = f"""
You are a 6-DOF robot arm controller. Generate a pick-and-place function call plan.

Available functions:
  move_tcp(x, y, z, speed)   — move TCP to XYZ in camera frame (metres), speed 0-1
  open_gripper()              — open the gripper
  close_gripper()             — close the gripper
  move_home()                 — return to home position

Object positions in camera frame (metres):
  Source ({SOURCE}): x={src_xyz['x']}, y={src_xyz['y']}, z={src_xyz['z']}
  Target ({TARGET}): x={tgt_xyz['x']}, y={tgt_xyz['y']}, z={tgt_xyz['z']}

Plan a safe pick-and-place:
- Approach 10cm above source, descend, grasp, lift, move above target, descend, release, return home.
- Return ONLY a JSON array of function calls (no other text):
[{{"function": "move_tcp", "args": [x, y, z, speed], "comment": "..."}}, ...]
"""

print("\nStep 2 — Generating function call plan...")
plan_response = client.models.generate_content(
    model="gemini-robotics-er-1.6-preview",
    contents=[PLAN_PROMPT],
    config=types.GenerateContentConfig(temperature=1.0),
)

raw_plan = plan_response.text.strip().strip("```json").strip("```").strip()
plan = json.loads(raw_plan)

print(f"\nPick-and-place plan ({len(plan)} steps):")
for i, step in enumerate(plan):
    fn = step["function"]
    args = step.get("args", [])
    comment = step.get("comment", "")
    print(f"  {i+1:2d}. {fn}({', '.join(str(a) for a in args)})  # {comment}")

# ---- Visualise ----
annotated = color.copy()
cv2.circle(annotated, (src_px, src_py), 12, (0, 255, 0), -1)
cv2.putText(annotated, f"PICK: {SOURCE}", (src_px+12, src_py),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
cv2.circle(annotated, (tgt_px, tgt_py), 12, (0, 0, 255), -1)
cv2.putText(annotated, f"PLACE: {TARGET}", (tgt_px+12, tgt_py),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
cv2.arrowedLine(annotated, (src_px, src_py), (tgt_px, tgt_py), (255, 200, 0), 2)

cv2.imwrite("result_05_plan.jpg", annotated)
print("\nSaved result_05_plan.jpg")
