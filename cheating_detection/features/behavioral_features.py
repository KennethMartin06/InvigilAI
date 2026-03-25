"""
behavioral_features.py — Keystroke-dynamics and mouse-trajectory feature extraction.

In a live system the raw event stream is captured by a JavaScript logger
(keydown/keyup timestamps, mouse-move coordinates, click events) and sent
to the server where this module processes it into per-window feature vectors.

Feature definitions follow the KU-RD/Keystroke-Authentication literature and
are consistent with the specification in the project requirements.
"""

import numpy as np


# ── Individual feature computations ──────────────────────────────────────────

def compute_keystroke_rate(timestamps_down: np.ndarray, window_sec: float = 2.0) -> float:
    """
    Keystrokes per second within a time window.

    Parameters
    ----------
    timestamps_down : np.ndarray
        Sorted key-down timestamps in seconds (epoch or relative).
    window_sec : float
        Duration of the window in seconds.

    Returns
    -------
    float — keystrokes / second.
    """
    if window_sec <= 0:
        return 0.0
    return len(timestamps_down) / window_sec


def compute_mean_dwell_time(timestamps_down: np.ndarray, timestamps_up: np.ndarray) -> float:
    """
    Mean key-hold duration (dwell time) in milliseconds.

    Dwell time = keyup_time − keydown_time for each keystroke.

    Parameters
    ----------
    timestamps_down : np.ndarray
        Key-down timestamps (seconds).
    timestamps_up : np.ndarray
        Corresponding key-up timestamps (seconds).

    Returns
    -------
    float — mean dwell time in ms, or 0.0 if no keystrokes.
    """
    if len(timestamps_down) == 0 or len(timestamps_up) == 0:
        return 0.0
    dwell = (timestamps_up - timestamps_down) * 1000  # ms
    dwell = dwell[dwell > 0]  # remove malformed events
    return float(np.mean(dwell)) if len(dwell) > 0 else 0.0


def compute_mean_flight_time(timestamps_down: np.ndarray) -> float:
    """
    Mean inter-keystroke interval (flight time) in milliseconds.

    Flight time (also called UD latency) is the gap between consecutive
    key-down events: t_down[i+1] − t_down[i].

    Parameters
    ----------
    timestamps_down : np.ndarray
        Sorted key-down timestamps in seconds.

    Returns
    -------
    float — mean flight time in ms, or 0.0 if fewer than 2 keystrokes.
    """
    if len(timestamps_down) < 2:
        return 0.0
    intervals = np.diff(timestamps_down) * 1000  # ms
    intervals = intervals[intervals > 0]
    return float(np.mean(intervals)) if len(intervals) > 0 else 0.0


def compute_burst_coefficient(timestamps_down: np.ndarray) -> float:
    """
    Variance-to-mean ratio of inter-keystroke intervals (index of dispersion).

    BC = Var(IKI) / Mean(IKI)

    A BC near 1 indicates Poisson-like regular typing (normal).
    BC >> 1 indicates bursty/irregular typing (e.g. copy-paste spikes).

    Parameters
    ----------
    timestamps_down : np.ndarray
        Sorted key-down timestamps in seconds.

    Returns
    -------
    float — burst coefficient, or 1.0 if insufficient data.
    """
    if len(timestamps_down) < 3:
        return 1.0
    intervals = np.diff(timestamps_down) * 1000
    intervals = intervals[intervals > 0]
    if len(intervals) < 2:
        return 1.0
    mean = np.mean(intervals)
    if mean < 1e-6:
        return 1.0
    return float(np.var(intervals) / mean)


def compute_cursor_velocity(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    timestamps: np.ndarray,
) -> float:
    """
    Mean cursor speed in pixels per second.

    Speed at each step = Euclidean distance / time delta.

    Parameters
    ----------
    x_coords, y_coords : np.ndarray
        Mouse X and Y pixel coordinates sampled at uniform or variable rate.
    timestamps : np.ndarray
        Corresponding timestamps in seconds.

    Returns
    -------
    float — mean cursor velocity (px/s), or 0.0 if fewer than 2 samples.
    """
    if len(x_coords) < 2:
        return 0.0
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    dt = np.diff(timestamps)
    dt = np.where(dt < 1e-6, 1e-6, dt)  # avoid division by zero
    speed = np.sqrt(dx**2 + dy**2) / dt
    return float(np.mean(speed))


def compute_click_frequency(click_times: np.ndarray, window_sec: float = 2.0) -> float:
    """
    Mouse clicks per second within a window.

    Parameters
    ----------
    click_times : np.ndarray
        Timestamps (seconds) of mouse-click events.
    window_sec : float
        Duration of the window.

    Returns
    -------
    float — clicks / second.
    """
    if window_sec <= 0:
        return 0.0
    return len(click_times) / window_sec


def compute_idle_ratio(
    event_times: np.ndarray,
    window_sec: float = 2.0,
    idle_threshold_sec: float = 0.5,
) -> float:
    """
    Fraction of the window where no input event occurred for longer than
    ``idle_threshold_sec``.

    Idle gaps longer than the threshold are summed and divided by window_sec.

    Parameters
    ----------
    event_times : np.ndarray
        Sorted timestamps (seconds) of all input events (keys + clicks + moves).
    window_sec : float
        Duration of the window.
    idle_threshold_sec : float
        Minimum gap to count as "idle".

    Returns
    -------
    float in [0, 1] — idle fraction of the window.
    """
    if window_sec <= 0:
        return 1.0
    if len(event_times) < 2:
        return 1.0
    gaps = np.diff(np.sort(event_times))
    idle_total = np.sum(gaps[gaps > idle_threshold_sec])
    return float(np.clip(idle_total / window_sec, 0.0, 1.0))


def compute_trajectory_linearity(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> float:
    """
    Straight-line distance / total path length ratio.

    A value close to 1 means the cursor moved in a nearly straight line
    (characteristic of automated or copy-paste paste operations).
    Natural mouse movement produces values in [0.4, 0.7].

    Parameters
    ----------
    x_coords, y_coords : np.ndarray
        Mouse X/Y coordinates.

    Returns
    -------
    float in [0, 1].
    """
    if len(x_coords) < 2:
        return 1.0
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    total_path = float(np.sum(np.sqrt(dx**2 + dy**2)))
    if total_path < 1e-6:
        return 1.0
    straight_line = float(
        np.sqrt((x_coords[-1] - x_coords[0]) ** 2 + (y_coords[-1] - y_coords[0]) ** 2)
    )
    return float(np.clip(straight_line / total_path, 0.0, 1.0))


# ── High-level extraction function ───────────────────────────────────────────

def extract_behavioral_features(session_data: dict) -> np.ndarray:
    """
    Extract the 8-dimensional behavioral feature vector for each time window.

    Expected keys in ``session_data`` (all are lists with one element per window):
        "keystroke_rate"      : pre-computed KR (or use "keys_down" + "keys_up")
        "mean_dwell_time"     : pre-computed MDT (ms)
        "mean_flight_time"    : pre-computed MFT (ms)
        "burst_coefficient"   : pre-computed BC
        "cursor_velocity"     : pre-computed CV (px/s)
        "click_frequency"     : pre-computed CF (clicks/s)
        "idle_ratio"          : pre-computed IR
        "trajectory_linearity": pre-computed TL

    When raw event streams are available, pass them under:
        "keys_down"   : list of per-window key-down timestamp arrays
        "keys_up"     : list of per-window key-up timestamp arrays
        "mouse_x/y"   : list of per-window coordinate arrays
        "mouse_times" : list of per-window mouse-event timestamp arrays
        "click_times" : list of per-window click-event timestamp arrays
        "all_event_times" : list of per-window all-event timestamp arrays

    Parameters
    ----------
    session_data : dict
        Dictionary with the keys described above.

    Returns
    -------
    np.ndarray, shape (n_windows, 8)
        Columns (in order):
          0  keystroke_rate
          1  mean_dwell_time
          2  mean_flight_time
          3  burst_coefficient
          4  cursor_velocity
          5  click_frequency
          6  idle_ratio
          7  trajectory_linearity
    """
    n_windows = len(session_data.get("keystroke_rate",
                    session_data.get("keys_down", [[]])))

    features = np.zeros((n_windows, 8), dtype=np.float64)

    # If pre-computed scalars are available use them directly
    if "keystroke_rate" in session_data:
        features[:, 0] = np.asarray(session_data["keystroke_rate"])
        features[:, 1] = np.asarray(session_data["mean_dwell_time"])
        features[:, 2] = np.asarray(session_data["mean_flight_time"])
        features[:, 3] = np.asarray(session_data["burst_coefficient"])
        features[:, 4] = np.asarray(session_data["cursor_velocity"])
        features[:, 5] = np.asarray(session_data["click_frequency"])
        features[:, 6] = np.asarray(session_data["idle_ratio"])
        features[:, 7] = np.asarray(session_data["trajectory_linearity"])
    else:
        # Compute from raw event streams (real-data path)
        keys_down = session_data["keys_down"]
        keys_up = session_data["keys_up"]
        mouse_x = session_data["mouse_x"]
        mouse_y = session_data["mouse_y"]
        mouse_times = session_data["mouse_times"]
        click_times = session_data["click_times"]
        all_event_times = session_data["all_event_times"]

        for i in range(n_windows):
            kd = np.asarray(keys_down[i])
            ku = np.asarray(keys_up[i])
            mx = np.asarray(mouse_x[i])
            my = np.asarray(mouse_y[i])
            mt = np.asarray(mouse_times[i])
            ct = np.asarray(click_times[i])
            et = np.asarray(all_event_times[i])

            features[i, 0] = compute_keystroke_rate(kd)
            features[i, 1] = compute_mean_dwell_time(kd, ku)
            features[i, 2] = compute_mean_flight_time(kd)
            features[i, 3] = compute_burst_coefficient(kd)
            features[i, 4] = compute_cursor_velocity(mx, my, mt)
            features[i, 5] = compute_click_frequency(ct)
            features[i, 6] = compute_idle_ratio(et)
            features[i, 7] = compute_trajectory_linearity(mx, my)

    return features
