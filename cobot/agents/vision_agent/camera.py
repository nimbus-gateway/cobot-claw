"""RealSense D455 capture helpers and coordinate transforms."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

_PKG_DIR   = Path(__file__).parent
CALIB_FILE = os.getenv("CALIB_FILE", str(_PKG_DIR / "calibration.json"))


def load_calibration(path: str = CALIB_FILE) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    T = np.array(data["T_base_cam"], dtype=float)
    print(f"[calib] Loaded from {path}")
    return T


def capture_realsense(retries: int = 3, warmup_frames: int = 30):
    """Capture one aligned RGB-D frame from RealSense D455.

    Returns (color_bgr, depth_uint16, intrinsics, depth_scale).
    Retries on pipeline startup failure (common when USB device hasn't fully
    released from a previous stop).
    """
    import time as _time

    last_exc: Exception | None = None
    for attempt in range(retries):
        pipe = rs.pipeline()
        cfg  = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
        try:
            profile = pipe.start(cfg)
            align   = rs.align(rs.stream.color)
            for _ in range(warmup_frames):
                pipe.wait_for_frames()
            frames = align.process(pipe.wait_for_frames())
            color  = np.asanyarray(frames.get_color_frame().get_data()).copy()
            depth  = np.asanyarray(frames.get_depth_frame().get_data()).copy()
            ci     = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            ds     = profile.get_device().first_depth_sensor().get_depth_scale()
            pipe.stop()
            return color, depth, ci, ds
        except Exception as e:
            last_exc = e
            try:
                pipe.stop()
            except Exception:
                pass
            if attempt < retries - 1:
                print(f"  [camera] capture failed (attempt {attempt + 1}/{retries}): {e} — retrying in 1s")
                _time.sleep(1.0)

    raise RuntimeError(f"RealSense capture failed after {retries} attempts: {last_exc}")


def encode_jpeg(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return bytes(buf)


def depth_at_pixel(depth: np.ndarray, ci, px: int, py: int,
                   scale: float, radius: int = 3) -> float:
    h, w = depth.shape
    vals = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = px + dx, py + dy
            if 0 <= nx < w and 0 <= ny < h:
                v = depth[ny, nx] * scale
                if v > 0.1:
                    vals.append(v)
    return float(np.median(vals)) if vals else 0.0


def depth_at_box(depth: np.ndarray, scale: float,
                 x1: int, y1: int, x2: int, y2: int,
                 shrink: float = 0.5) -> float:
    """Median depth over the central region of a bounding box.

    shrink=0.5 samples only the inner 50% of the box, avoiding
    edge pixels that may land on the table or background.
    """
    h, w = depth.shape
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw = (x2 - x1) / 2 * shrink
    hh = (y2 - y1) / 2 * shrink
    ix1 = max(0, int(cx - hw))
    ix2 = min(w - 1, int(cx + hw))
    iy1 = max(0, int(cy - hh))
    iy2 = min(h - 1, int(cy + hh))
    region = depth[iy1:iy2 + 1, ix1:ix2 + 1].astype(float) * scale
    vals = region[region > 0.1].flatten()
    return float(np.median(vals)) if len(vals) else 0.0


def depth_at_box_top(depth: np.ndarray, scale: float,
                     x1: int, y1: int, x2: int, y2: int,
                     top_fraction: float = 0.3) -> float:
    """Median depth over the top portion of a bounding box.

    Samples the uppermost `top_fraction` of the box height (full width).
    For objects like boxes viewed at an angle, the upper pixels correspond
    to the top surface rather than the front face, giving the correct
    drop-height depth.  Falls back to depth_at_box if the top strip has
    no valid readings.
    """
    h, w = depth.shape
    strip_h = max(1, int((y2 - y1) * top_fraction))
    ix1 = max(0, x1)
    ix2 = min(w - 1, x2)
    iy1 = max(0, y1)
    iy2 = min(h - 1, y1 + strip_h)
    region = depth[iy1:iy2 + 1, ix1:ix2 + 1].astype(float) * scale
    vals = region[region > 0.1].flatten()
    if len(vals):
        return float(np.median(vals))
    # fallback to full-box median
    return depth_at_box(depth, scale, x1, y1, x2, y2)


def pixel_to_camera_xyz(ci, px: int, py: int, d_m: float) -> np.ndarray:
    """Back-project a pixel + depth to a homogeneous camera-frame point."""
    x = (px - ci.ppx) / ci.fx * d_m
    y = (py - ci.ppy) / ci.fy * d_m
    return np.array([x, y, d_m, 1.0])


def camera_to_base(xyz_cam: np.ndarray, T: np.ndarray) -> dict:
    """Transform a homogeneous camera-frame point to robot base frame."""
    pt = T @ xyz_cam
    return {"x": round(float(pt[0]), 4),
            "y": round(float(pt[1]), 4),
            "z": round(float(pt[2]), 4)}
