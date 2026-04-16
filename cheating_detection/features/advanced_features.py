"""
advanced_features.py -- New visual, behavioral, and audio features.

Adds:
  - Blink rate & eye closure duration
  - Gaze fixation duration & saccade speed
  - Keystroke bigram timings
  - Audio energy variance & spectral features
  - Behavioral entropy
  - Gaze texture (GLCM-style from gaze heatmaps)
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# -- Eye / Gaze Features ----------------------------------------------------

def blink_rate(eye_open_signal, fps=30):
    """Compute blinks per minute from a binary eye-open signal."""
    closed = (eye_open_signal < 0.5).astype(int)
    edges = np.diff(closed)
    blinks = (edges == 1).sum()
    duration_s = len(eye_open_signal) / fps
    return 60.0 * blinks / max(duration_s, 1e-3)


def eye_closure_duration(eye_open_signal, fps=30):
    """Mean eye-closure duration (s)."""
    closed = eye_open_signal < 0.5
    durations = []
    count = 0
    for c in closed:
        if c:
            count += 1
        elif count > 0:
            durations.append(count / fps)
            count = 0
    if count > 0:
        durations.append(count / fps)
    return float(np.mean(durations)) if durations else 0.0


def gaze_fixation_duration(gaze_xy, velocity_threshold=0.02, fps=30):
    """Mean duration of fixations (low gaze velocity segments)."""
    v = np.sqrt(np.sum(np.diff(gaze_xy, axis=0) ** 2, axis=1))
    fixating = v < velocity_threshold
    durations = []
    count = 0
    for f in fixating:
        if f:
            count += 1
        elif count > 0:
            durations.append(count / fps)
            count = 0
    if count > 0:
        durations.append(count / fps)
    return float(np.mean(durations)) if durations else 0.0


def saccade_speed(gaze_xy, fps=30):
    """Mean saccade velocity (degrees/sec approx)."""
    v = np.sqrt(np.sum(np.diff(gaze_xy, axis=0) ** 2, axis=1))
    # peaks above 80th percentile treated as saccades
    thresh = np.percentile(v, 80)
    sacc = v[v > thresh]
    return float(sacc.mean() * fps) if len(sacc) else 0.0


def gaze_heatmap_texture(gaze_xy, grid=8):
    """GLCM-like texture features from gaze heatmap."""
    if len(gaze_xy) < 2:
        return {"texture_contrast": 0.0, "texture_homogeneity": 0.0, "texture_energy": 0.0}
    x = np.clip(gaze_xy[:, 0], 0, 1)
    y = np.clip(gaze_xy[:, 1], 0, 1)
    H, _, _ = np.histogram2d(x, y, bins=grid, range=[[0, 1], [0, 1]])
    H = H / (H.sum() + 1e-6)
    # contrast: sum (i-j)^2 * P(i,j)
    i, j = np.indices(H.shape)
    contrast = float(((i - j) ** 2 * H).sum())
    homogeneity = float((H / (1.0 + np.abs(i - j))).sum())
    energy = float((H ** 2).sum())
    return {"texture_contrast": contrast, "texture_homogeneity": homogeneity, "texture_energy": energy}


# -- Keystroke Bigram Features ---------------------------------------------

def keystroke_bigram_features(key_events):
    """Extract bigram timing stats from keystroke events.

    key_events: list of dicts with 'key', 'press', 'release' timestamps
    """
    if len(key_events) < 2:
        return {"bigram_mean": 0.0, "bigram_std": 0.0, "bigram_skew": 0.0}
    intervals = []
    for a, b in zip(key_events[:-1], key_events[1:]):
        intervals.append(b["press"] - a["press"])
    intervals = np.array(intervals)
    mean = float(intervals.mean())
    std = float(intervals.std())
    # Fisher-Pearson skewness
    if std < 1e-6:
        skew = 0.0
    else:
        skew = float(((intervals - mean) ** 3).mean() / (std ** 3))
    return {"bigram_mean": mean, "bigram_std": std, "bigram_skew": skew}


# -- Audio Features --------------------------------------------------------

def audio_energy_variance(audio, frame_size=1024, hop_size=512):
    """Variance of frame-wise audio energy."""
    if len(audio) < frame_size:
        return 0.0
    frames = []
    for i in range(0, len(audio) - frame_size, hop_size):
        frames.append(np.sum(audio[i:i + frame_size] ** 2))
    return float(np.var(frames))


def audio_spectral_features(audio, sr=16000, n_fft=1024):
    """Spectral centroid, bandwidth, rolloff, zero-crossing rate."""
    if len(audio) < n_fft:
        return {"spectral_centroid": 0.0, "spectral_bandwidth": 0.0,
                "spectral_rolloff": 0.0, "zero_crossing_rate": 0.0}

    # FFT magnitude
    win = np.hanning(n_fft)
    seg = audio[:n_fft] * win
    spectrum = np.abs(np.fft.rfft(seg)) + 1e-8
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / np.sum(spectrum)))
    cumsum = np.cumsum(spectrum)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)

    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "zero_crossing_rate": zcr,
    }


# -- Behavioral Entropy ----------------------------------------------------

def behavioral_entropy(feature_values, n_bins=10):
    """Shannon entropy over a histogram of behavioral feature values."""
    if len(feature_values) == 0:
        return 0.0
    hist, _ = np.histogram(feature_values, bins=n_bins, density=False)
    p = hist / (hist.sum() + 1e-6)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def multivariate_entropy(X, n_bins=8):
    """Sum of per-column Shannon entropies (upper bound on joint entropy)."""
    total = 0.0
    for c in range(X.shape[1]):
        total += behavioral_entropy(X[:, c], n_bins=n_bins)
    return float(total)


# -- Combined extractor ----------------------------------------------------

def compute_all_advanced_features(session_data):
    """Compute all advanced features from a session dict.

    session_data: dict with optional keys:
      eye_open, gaze_xy, key_events, audio, behavioral_matrix
    """
    out = {}
    if "eye_open" in session_data:
        out["blink_rate"] = blink_rate(session_data["eye_open"])
        out["eye_closure_duration"] = eye_closure_duration(session_data["eye_open"])
    if "gaze_xy" in session_data:
        g = np.asarray(session_data["gaze_xy"])
        out["gaze_fixation_duration"] = gaze_fixation_duration(g)
        out["saccade_speed"] = saccade_speed(g)
        out.update(gaze_heatmap_texture(g))
    if "key_events" in session_data:
        out.update(keystroke_bigram_features(session_data["key_events"]))
    if "audio" in session_data:
        audio = np.asarray(session_data["audio"])
        out["audio_energy_variance"] = audio_energy_variance(audio)
        out.update(audio_spectral_features(audio))
    if "behavioral_matrix" in session_data:
        out["behavioral_entropy"] = multivariate_entropy(np.asarray(session_data["behavioral_matrix"]))
    return out
