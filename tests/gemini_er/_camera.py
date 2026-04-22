"""Shared helper — capture one frame from the RealSense D455."""
import cv2
import numpy as np
import pyrealsense2 as rs


def capture_frame(width=640, height=480, warmup=30):
    """Return (color_bgr, depth_uint16, intrinsics, depth_scale)."""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    profile = pipe.start(cfg)

    align = rs.align(rs.stream.color)
    for _ in range(warmup):
        pipe.wait_for_frames()

    frames = align.process(pipe.wait_for_frames())
    color = np.asanyarray(frames.get_color_frame().get_data()).copy()
    depth = np.asanyarray(frames.get_depth_frame().get_data()).copy()

    ci = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    pipe.stop()
    return color, depth, ci, depth_scale


def encode_jpeg(image_bgr, quality=90):
    """Encode BGR image to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def deproject_pixel(ci, depth_uint16, px, py, depth_scale):
    """Convert pixel (px, py) + depth to camera-frame XYZ in metres."""
    d = float(depth_uint16[int(py), int(px)]) * depth_scale
    if d == 0:
        return None
    x = (px - ci.ppx) * d / ci.fx
    y = (py - ci.ppy) * d / ci.fy
    return {"x": round(x, 4), "y": round(y, 4), "z": round(d, 4)}


def norm_to_pixel(norm_y, norm_x, height, width):
    """Convert Gemini normalised [0-1000] coords to pixel coords."""
    px = int(norm_x / 1000 * width)
    py = int(norm_y / 1000 * height)
    return px, py
