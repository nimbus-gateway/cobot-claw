"""CLI wrapper — delegates to cobot.agents.vision_agent.generate_scenario().

Usage:
    python generate_scenario.py --pick "water bottle" --place "right side of table"
    python generate_scenario.py --pick "red block" --place "blue tray" --name my_task
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cobot.agents.vision_agent import generate_scenario

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate a vision-guided pick-and-place scenario")
    ap.add_argument("--pick",  required=True, help="Object to pick e.g. 'water bottle'")
    ap.add_argument("--place", required=True, help="Place target e.g. 'right side of table'")
    ap.add_argument("--name",  default="vision_pick", help="Scenario filename (no .yaml)")
    ap.add_argument("--speed", type=float, default=0.3, help="Joint speed rad/s")
    ap.add_argument("--accel", type=float, default=0.5, help="Joint accel rad/s²")
    args = ap.parse_args()

    result = generate_scenario(
        pick=args.pick,
        place=args.place,
        name=args.name,
        speed=args.speed,
        accel=args.accel,
    )

    if not result["success"]:
        print(f"\nERROR: {result['message']}")
        sys.exit(1)
