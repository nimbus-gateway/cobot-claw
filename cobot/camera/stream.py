"""
CameraStream — singleton that owns the RealSense and fans frames out.

Responsibilities:
  - Single owner of the RealSense device (prevents multi-process conflicts).
  - Captures frames at a configurable FPS in a background thread.
  - Provides get_latest_frame() for the vision agent (replaces capture_realsense()).
  - Provides subscribe() for future agents (monitor, safety) to receive every frame.
  - Optional live OpenCV debug window with detection overlays.

Designed so the monitor/safety agent can subscribe without any camera ownership changes.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import cv2
import numpy as np


class CameraStream:
    """
    Background camera capture loop with subscriber fan-out and optional debug window.

    Usage:
        stream = CameraStream(fps=10)
        stream.start()
        color, depth, ci, ds = stream.get_latest_frame()
        stream.enable_debug_window()   # opens cv2 window
        stream.stop()
    """

    def __init__(self, fps: int = 10) -> None:
        self._fps = fps
        self._interval = 1.0 / fps

        self._latest: tuple | None = None          # (color, depth, ci, ds)
        self._annotated: np.ndarray | None = None  # overlay frame shown in debug window
        self._frame_lock = threading.Lock()
        self._overlay_lock = threading.Lock()

        self._subscribers: list[Callable] = []
        self._sub_lock = threading.Lock()

        self._running = False
        self._capture_thread: threading.Thread | None = None

        # Debug window state
        self._debug_enabled = False
        self._debug_thread: threading.Thread | None = None
        self._debug_stop = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="camera-stream")
        self._capture_thread.start()

    def stop(self) -> None:
        self._running = False
        self.disable_debug_window()
        if self._capture_thread:
            self._capture_thread.join(timeout=3.0)
            self._capture_thread = None

    # ------------------------------------------------------------------
    # Frame access — for vision agent
    # ------------------------------------------------------------------

    def get_latest_frame(self) -> tuple:
        """
        Return the most recent (color, depth, ci, ds) tuple.
        Waits up to 3s for the first frame; falls back to a direct capture if none arrives.
        """
        deadline = time.time() + 3.0
        while self._latest is None and time.time() < deadline:
            time.sleep(0.05)
        with self._frame_lock:
            if self._latest is not None:
                return self._latest

        # Stream hasn't produced a frame yet — capture directly as fallback
        print("  [camera] stream has no frame, capturing directly…", flush=True)
        from cobot.agents.vision_agent.camera import capture_realsense
        color, depth, ci, ds = capture_realsense()
        with self._frame_lock:
            self._latest = (color, depth, ci, ds)
        with self._overlay_lock:
            self._annotated = color.copy()
        return self._latest

    # ------------------------------------------------------------------
    # Subscriber fan-out — for monitor/safety agent
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable) -> None:
        """
        Register a callback(color, depth, ci, ds) called on every new frame.
        Callbacks run in the capture thread — keep them fast (set an event, append to queue).
        """
        with self._sub_lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        with self._sub_lock:
            self._subscribers = [s for s in self._subscribers if s is not callback]

    # ------------------------------------------------------------------
    # Detection overlay — vision agent calls this after each detection
    # ------------------------------------------------------------------

    def overlay_detection(
        self,
        label: str,
        px: int,
        py: int,
        d_m: float,
        box: tuple[int, int, int, int] | None = None,
    ) -> None:
        """
        Draw a detection result onto the debug window frame.
        Safe to call from any thread.
        box: (x1, y1, x2, y2) in pixels, optional.
        """
        with self._overlay_lock:
            if self._annotated is None:
                return
            frame = self._annotated.copy()

        color_bgr = (0, 255, 0)
        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.circle(frame, (px, py), 10, color_bgr, -1)
        cv2.putText(
            frame,
            f"{label}  {d_m:.2f}m",
            (px + 14, py - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color_bgr,
            2,
        )

        with self._overlay_lock:
            self._annotated = frame

    def clear_overlays(self) -> None:
        """Reset annotation layer to the raw camera frame."""
        with self._frame_lock:
            if self._latest is not None:
                color = self._latest[0]
        with self._overlay_lock:
            self._annotated = color.copy()

    # ------------------------------------------------------------------
    # Debug window
    # ------------------------------------------------------------------

    def enable_debug_window(self) -> None:
        if self._debug_enabled:
            return
        self._debug_enabled = True
        self._debug_stop.clear()
        self._debug_thread = threading.Thread(
            target=self._debug_window_loop,
            daemon=True,
            name="camera-debug-window",
        )
        self._debug_thread.start()

    def disable_debug_window(self) -> None:
        if not self._debug_enabled:
            return
        self._debug_enabled = False
        self._debug_stop.set()
        if self._debug_thread:
            self._debug_thread.join(timeout=2.0)
            self._debug_thread = None

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        from cobot.agents.vision_agent.camera import capture_realsense

        while self._running:
            t0 = time.perf_counter()
            try:
                color, depth, ci, ds = capture_realsense()
                with self._frame_lock:
                    self._latest = (color, depth, ci, ds)
                with self._overlay_lock:
                    self._annotated = color.copy()

                # Fan out to subscribers (monitor/safety agent hooks in here)
                with self._sub_lock:
                    subs = list(self._subscribers)
                for cb in subs:
                    try:
                        cb(color, depth, ci, ds)
                    except Exception:
                        pass

            except Exception as e:
                print(f"  [camera] capture error: {e}", flush=True)
                time.sleep(0.5)  # back off before retrying

            elapsed = time.perf_counter() - t0
            sleep_for = self._interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _debug_window_loop(self) -> None:
        window_name = "Cobot Vision — Debug"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)

        while not self._debug_stop.is_set():
            with self._overlay_lock:
                frame = self._annotated

            if frame is not None:
                # Overlay stream info
                display = frame.copy()
                ts = time.strftime("%H:%M:%S")
                cv2.putText(display, f"Cobot Vision  {ts}  {self._fps}fps",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                cv2.imshow(window_name, display)

            key = cv2.waitKey(max(1, int(self._interval * 1000)))
            if key == ord("q"):
                break

        cv2.destroyWindow(window_name)
        self._debug_enabled = False
