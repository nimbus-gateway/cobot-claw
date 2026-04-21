"""CLI wrapper — delegates to cobot.agents.vision_agent.calibrate_handeye().

Usage:
    python calibrate_handeye.py              # run full calibration
    python calibrate_handeye.py --check-only # just report current error
    python calibrate_handeye.py --dry-run    # skip robot movement
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cobot.agents.vision_agent import calibrate_handeye

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="Report errors but do not overwrite calibration.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip robot movement (just test Gemini detection)")
    args = ap.parse_args()

    result = asyncio.run(calibrate_handeye(
        check_only=args.check_only,
        dry_run=args.dry_run,
    ))

    if not result["success"]:
        print(f"\nERROR: {result['message']}")
        sys.exit(1)
