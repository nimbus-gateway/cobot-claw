"""
Test 6 — Spatial reasoning queries.

Ask Gemini ER questions about the scene that require understanding
spatial relationships — not just "where is X" but "which X is closest",
"what is blocking Y", "is there space for Z".

Output: result_06_spatial.jpg
"""
import json, os, cv2
from google import genai
from google.genai import types
from _camera import capture_frame, encode_jpeg, norm_to_pixel, deproject_pixel

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

QUERIES = [
    "Which object is closest to the robot base?",
    "Which object is easiest for the robot gripper to pick up?",
    "Is there clear space between the two bottles?",
    "Which object should I move first to clear a path from left to right?",
]

print("Capturing frame...")
color, depth, ci, depth_scale = capture_frame()
h, w = color.shape[:2]

annotated = color.copy()
print("\nSpatial reasoning results:\n")

for i, query in enumerate(QUERIES):
    prompt = f"""
    Analyse this robot workspace image and answer:
    {query}

    Then point to the most relevant object.
    Return JSON:
    {{
        "answer": "...",
        "object": "...",
        "point": [y, x]
    }}
    Points normalised 0-1000. Return ONLY the JSON.
    """
    response = client.models.generate_content(
        model="gemini-robotics-er-1.6-preview",
        contents=[
            types.Part.from_bytes(data=encode_jpeg(color), mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = response.text.strip().strip("```json").strip("```").strip()
    try:
        result = json.loads(raw)
        print(f"Q{i+1}: {query}")
        print(f"  → {result.get('answer', '?')}")
        print(f"  → Object: {result.get('object', '?')}\n")
        if "point" in result:
            ny, nx = result["point"]
            px, py = norm_to_pixel(ny, nx, h, w)
            colors = [(0,255,0),(0,128,255),(255,0,128),(255,200,0)]
            cv2.circle(annotated, (px, py), 10, colors[i], -1)
            cv2.putText(annotated, f"Q{i+1}", (px+10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
    except json.JSONDecodeError:
        print(f"Q{i+1}: {query}")
        print(f"  → {raw}\n")

cv2.imwrite("result_06_spatial.jpg", annotated)
print("Saved result_06_spatial.jpg")
