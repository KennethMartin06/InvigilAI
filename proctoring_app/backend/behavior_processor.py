"""
behavior_processor.py — Per-session keystroke and mouse event processing.

Receives raw event lists from the frontend and computes the 8 behavioral
features expected by the ML model.

Behavioral features (in model order, indices 8–15):
  [8]  keystroke_rate         keys per second
  [9]  mean_dwell_time        avg key hold duration (ms)
  [10] mean_flight_time       avg inter-key interval (ms)
  [11] burst_coefficient      variance / mean of inter-key intervals
  [12] cursor_velocity        mean cursor speed (px/s)
  [13] click_frequency        mouse clicks per second
  [14] idle_ratio             fraction of window with no input
  [15] trajectory_linearity   straight-line / total path ratio

Event format from frontend:
  Keystroke: {"key": "a", "type": "down"|"up", "timestamp": <ms>}
  Mouse:     {"x": int, "y": int, "type": "move"|"click", "timestamp": <ms>}
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 2.0
_IDLE_THRESHOLD_MS = 500.0  # gap > 500ms counts as idle


class BehaviorProcessor:
    """
    Stateful processor that ingests raw UI events and returns feature dicts.

    One instance per active WebSocket session.
    """

    def __init__(self) -> None:
        self._keystroke_buffer: list[dict] = []
        self._mouse_buffer: list[dict] = []

    # ── Public interface ──────────────────────────────────────────────────────

    def add_keystroke_events(self, events: list[dict]) -> None:
        """
        Append incoming keystroke events to the internal buffer.

        Parameters
        ----------
        events : list[dict] — each with keys: key, type, timestamp (ms).
        """
        self._keystroke_buffer.extend(events)

    def add_mouse_events(self, events: list[dict]) -> None:
        """
        Append incoming mouse events to the internal buffer.

        Parameters
        ----------
        events : list[dict] — each with keys: x, y, type, timestamp (ms).
        """
        self._mouse_buffer.extend(events)

    def compute_and_clear(self) -> dict:
        """
        Compute all 8 behavioral features from the current buffers, then
        clear both buffers for the next window.

        Returns
        -------
        dict — all 8 feature values (safe defaults if buffers are empty).
        """
        ks_features = self._process_keystrokes(self._keystroke_buffer)
        ms_features = self._process_mouse(self._mouse_buffer)

        self._keystroke_buffer = []
        self._mouse_buffer = []

        return {**ks_features, **ms_features}

    # ── Keystroke processing ──────────────────────────────────────────────────

    def _process_keystrokes(self, events: list[dict]) -> dict:
        """
        Compute keystroke_rate, mean_dwell_time, mean_flight_time, burst_coefficient.

        Parameters
        ----------
        events : list[dict]

        Returns
        -------
        dict with the 4 keystroke feature values.
        """
        if not events:
            return self._default_keystroke_features()

        downs = sorted(
            [e for e in events if e.get("type") == "down"],
            key=lambda e: e["timestamp"],
        )
        ups = {
            e["key"]: e["timestamp"]
            for e in events
            if e.get("type") == "up"
        }

        if not downs:
            return self._default_keystroke_features()

        # Keystroke rate — events per second
        n_keys = len(downs)
        keystroke_rate = n_keys / _WINDOW_SECONDS

        # Dwell times (hold duration)
        dwell_times = []
        for d in downs:
            up_ts = ups.get(d["key"])
            if up_ts is not None:
                dwell = up_ts - d["timestamp"]
                if 0 < dwell < 2000:  # sanity check
                    dwell_times.append(dwell)

        mean_dwell_time = float(np.mean(dwell_times)) if dwell_times else 100.0

        # Flight times (inter-key interval)
        ts_list = [d["timestamp"] for d in downs]
        if len(ts_list) >= 2:
            intervals = np.diff(sorted(ts_list))
            intervals = intervals[(intervals > 0) & (intervals < 5000)]
            mean_flight_time = float(np.mean(intervals)) if len(intervals) > 0 else 150.0
            burst_coefficient = (
                float(np.var(intervals) / (np.mean(intervals) + 1e-6))
                if len(intervals) > 1
                else 1.0
            )
        else:
            mean_flight_time = 150.0
            burst_coefficient = 1.0

        return {
            "keystroke_rate": round(keystroke_rate, 4),
            "mean_dwell_time": round(mean_dwell_time, 4),
            "mean_flight_time": round(mean_flight_time, 4),
            "burst_coefficient": round(min(burst_coefficient, 10.0), 4),
        }

    # ── Mouse processing ──────────────────────────────────────────────────────

    def _process_mouse(self, events: list[dict]) -> dict:
        """
        Compute cursor_velocity, click_frequency, idle_ratio, trajectory_linearity.

        Parameters
        ----------
        events : list[dict]

        Returns
        -------
        dict with the 4 mouse feature values.
        """
        if not events:
            return self._default_mouse_features()

        moves = sorted(
            [e for e in events if e.get("type") == "move"],
            key=lambda e: e["timestamp"],
        )
        clicks = [e for e in events if e.get("type") == "click"]

        click_frequency = len(clicks) / _WINDOW_SECONDS

        if len(moves) < 2:
            return {
                "cursor_velocity": 0.0,
                "click_frequency": round(click_frequency, 4),
                "idle_ratio": 0.8,
                "trajectory_linearity": 1.0,
            }

        xs = np.array([e["x"] for e in moves], dtype=np.float64)
        ys = np.array([e["y"] for e in moves], dtype=np.float64)
        ts = np.array([e["timestamp"] for e in moves], dtype=np.float64)  # ms

        # Cursor velocity (px/s)
        dists = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
        dt_s = np.diff(ts) / 1000.0
        dt_s = np.where(dt_s < 1e-4, 1e-4, dt_s)
        speeds = dists / dt_s
        cursor_velocity = float(np.mean(speeds))

        # Idle ratio — fraction of window with no events
        all_ts = sorted([e["timestamp"] for e in events])
        idle_total_ms = sum(
            g for g in np.diff(all_ts) if g > _IDLE_THRESHOLD_MS
        )
        idle_ratio = float(np.clip(idle_total_ms / (_WINDOW_SECONDS * 1000.0), 0, 1))

        # Trajectory linearity
        total_path = float(np.sum(dists))
        straight_line = float(
            np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2)
        )
        if total_path < 1e-4:
            linearity = 1.0
        else:
            linearity = float(np.clip(straight_line / total_path, 0.0, 1.0))

        return {
            "cursor_velocity": round(cursor_velocity, 4),
            "click_frequency": round(click_frequency, 4),
            "idle_ratio": round(idle_ratio, 4),
            "trajectory_linearity": round(linearity, 4),
        }

    # ── Safe defaults ─────────────────────────────────────────────────────────

    @staticmethod
    def _default_keystroke_features() -> dict:
        """Return neutral keystroke features when no data is available."""
        return {
            "keystroke_rate": 0.0,
            "mean_dwell_time": 100.0,
            "mean_flight_time": 150.0,
            "burst_coefficient": 1.0,
        }

    @staticmethod
    def _default_mouse_features() -> dict:
        """Return neutral mouse features when no data is available."""
        return {
            "cursor_velocity": 0.0,
            "click_frequency": 0.0,
            "idle_ratio": 0.5,
            "trajectory_linearity": 0.5,
        }

    def reset(self) -> None:
        """Clear all buffers (call when session ends)."""
        self._keystroke_buffer = []
        self._mouse_buffer = []
