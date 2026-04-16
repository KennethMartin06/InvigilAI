"""
keystroke_loader.py -- Loaders for CMU and GREYC keystroke dynamics datasets.

CMU:  Killourhy & Maxion 2009 "Comparing Anomaly-Detection Algorithms for
      Keystroke Dynamics" -- 51 subjects, password typing.
GREYC: University of Caen -- ~130 subjects, free-text keystrokes.
"""

import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    CMU_KEYSTROKE_PATH,
    GREYC_KEYSTROKE_PATH,
)


def load_cmu_keystroke(path=CMU_KEYSTROKE_PATH, n_synth=500, verbose=True):
    """Load CMU keystroke CSV (subject, sessionIndex, rep, H/DD/UD times)."""
    if not os.path.isdir(path):
        if verbose:
            print(f"[CMU] Path missing: {path}, synthesizing")
        return _synth_cmu(n_synth)

    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not files:
        return _synth_cmu(n_synth)

    dfs = [pd.read_csv(os.path.join(path, f)) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    if verbose:
        print(f"[CMU] Loaded {len(df)} trials from {len(files)} files")
    return df


def _synth_cmu(n):
    rng = np.random.default_rng(RANDOM_SEED + 2)
    n_subj = 20
    data = {
        "subject": rng.integers(0, n_subj, n),
        "sessionIndex": rng.integers(1, 9, n),
        "rep": rng.integers(1, 51, n),
    }
    # Typical hold/DD/UD keys for password ".tie5Roanl"
    for k in [".", "t", "i", "e", "5", "R", "o", "a", "n", "l"]:
        data[f"H.{k}"] = rng.normal(0.09, 0.02, n).clip(0.02, 0.3)
        data[f"DD.{k}"] = rng.normal(0.18, 0.05, n).clip(0.05, 0.5)
        data[f"UD.{k}"] = rng.normal(0.09, 0.04, n).clip(0.01, 0.35)
    return pd.DataFrame(data)


def load_greyc_keystroke(path=GREYC_KEYSTROKE_PATH, n_synth=500, verbose=True):
    """Load GREYC keystroke dataset (tab-separated press/release times)."""
    if not os.path.isdir(path):
        if verbose:
            print(f"[GREYC] Path missing: {path}, synthesizing")
        return _synth_greyc(n_synth)

    files = [f for f in os.listdir(path) if f.endswith((".csv", ".tsv", ".txt"))]
    if not files:
        return _synth_greyc(n_synth)

    dfs = []
    for f in files:
        sep = "\t" if f.endswith((".tsv", ".txt")) else ","
        dfs.append(pd.read_csv(os.path.join(path, f), sep=sep))
    df = pd.concat(dfs, ignore_index=True)
    if verbose:
        print(f"[GREYC] Loaded {len(df)} events from {len(files)} files")
    return df


def _synth_greyc(n):
    rng = np.random.default_rng(RANDOM_SEED + 3)
    base_time = np.cumsum(rng.exponential(0.15, n))
    key_codes = rng.integers(65, 91, n)  # A-Z
    return pd.DataFrame({
        "subject": rng.integers(0, 50, n),
        "keycode": key_codes,
        "press_time": base_time,
        "release_time": base_time + rng.normal(0.09, 0.02, n).clip(0.02, 0.25),
    })


def extract_keystroke_features(df, source="cmu"):
    """Extract invigilation-compatible keystroke features."""
    if source == "cmu":
        hold_cols = [c for c in df.columns if c.startswith("H.")]
        dd_cols = [c for c in df.columns if c.startswith("DD.")]
        ud_cols = [c for c in df.columns if c.startswith("UD.")]

        dwell = df[hold_cols].values.flatten() if hold_cols else np.array([0.1])
        flight_dd = df[dd_cols].values.flatten() if dd_cols else np.array([0.18])
        flight_ud = df[ud_cols].values.flatten() if ud_cols else np.array([0.09])

        return {
            "keystroke_rate": float(60.0 / (flight_dd.mean() + 1e-6)),
            "mean_dwell_time": float(dwell.mean()),
            "mean_flight_time": float(flight_ud.mean()),
            "dwell_std": float(dwell.std()),
            "flight_std": float(flight_ud.std()),
            "burst_coefficient": float(flight_dd.std() / (flight_dd.mean() + 1e-6)),
        }

    # GREYC: event-based
    press = df["press_time"].values
    release = df["release_time"].values
    dwell = release - press
    flight = np.diff(press)

    return {
        "keystroke_rate": float(60.0 / (np.mean(flight) + 1e-6)),
        "mean_dwell_time": float(np.mean(dwell)),
        "mean_flight_time": float(np.mean(flight)) if len(flight) else 0.1,
        "dwell_std": float(np.std(dwell)),
        "flight_std": float(np.std(flight)) if len(flight) else 0.05,
        "burst_coefficient": float(np.std(flight) / (np.mean(flight) + 1e-6))
            if len(flight) else 0.5,
    }


def compute_bigram_timings(df, source="greyc", top_k=20):
    """Compute bigram (key-pair) transition times."""
    if source != "greyc":
        return {}
    keys = df["keycode"].values
    times = df["press_time"].values
    bigrams = {}
    for i in range(len(keys) - 1):
        b = (int(keys[i]), int(keys[i + 1]))
        bigrams.setdefault(b, []).append(times[i + 1] - times[i])
    # keep most frequent
    sorted_b = sorted(bigrams.items(), key=lambda kv: -len(kv[1]))[:top_k]
    return {f"bigram_{a}_{b}": float(np.mean(ts)) for (a, b), ts in sorted_b}
