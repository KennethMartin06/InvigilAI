"""
temporal_augment.py -- Temporal augmentation for sequential/time-series features.

Techniques:
  - Time warping: non-linear time-axis distortion via cubic splines
  - Magnitude warping: amplitude scaling along the time axis
  - Window slicing: random cropping of time windows
  - Permutation: shuffle sub-segments within a sequence
"""

import numpy as np
from scipy.interpolate import CubicSpline

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    TIME_WARP_SIGMA,
    TIME_WARP_KNOT,
    MAGNITUDE_WARP_SIGMA,
)


def _generate_random_curves(n_samples, n_features, sigma, knot, rng):
    xx = np.arange(0, n_samples, (n_samples - 1) / (knot + 1))[:knot + 2]
    yy = rng.normal(loc=1.0, scale=sigma, size=(knot + 2, n_features))
    x_range = np.arange(n_samples)
    curves = np.zeros((n_samples, n_features))
    for i in range(n_features):
        cs = CubicSpline(xx, yy[:, i])
        curves[:, i] = cs(x_range)
    return curves


def time_warp(X_seq, sigma=TIME_WARP_SIGMA, knot=TIME_WARP_KNOT, rng=None):
    """Apply non-linear time warping via cubic-spline distortion.

    X_seq shape: (n_samples, seq_len, n_features) or (seq_len, n_features)
    """
    rng = rng or np.random.default_rng(RANDOM_SEED)
    single = (X_seq.ndim == 2)
    if single:
        X_seq = X_seq[None, ...]

    n_samples, seq_len, n_features = X_seq.shape
    out = np.zeros_like(X_seq)

    for s in range(n_samples):
        curves = _generate_random_curves(seq_len, n_features, sigma, knot, rng)
        t_cum = np.cumsum(curves, axis=0)
        t_scale = (seq_len - 1) / t_cum[-1]
        t_warped = t_cum * t_scale
        for f in range(n_features):
            out[s, :, f] = np.interp(np.arange(seq_len), t_warped[:, f], X_seq[s, :, f])

    return out[0] if single else out


def magnitude_warp(X_seq, sigma=MAGNITUDE_WARP_SIGMA, knot=TIME_WARP_KNOT, rng=None):
    """Apply per-feature magnitude scaling curves."""
    rng = rng or np.random.default_rng(RANDOM_SEED + 10)
    single = (X_seq.ndim == 2)
    if single:
        X_seq = X_seq[None, ...]

    n_samples, seq_len, n_features = X_seq.shape
    out = np.zeros_like(X_seq)
    for s in range(n_samples):
        curves = _generate_random_curves(seq_len, n_features, sigma, knot, rng)
        out[s] = X_seq[s] * curves

    return out[0] if single else out


def window_slice(X_seq, slice_ratio=0.9, rng=None):
    """Randomly crop sub-window and resize back to original length."""
    rng = rng or np.random.default_rng(RANDOM_SEED + 20)
    single = (X_seq.ndim == 2)
    if single:
        X_seq = X_seq[None, ...]

    n_samples, seq_len, n_features = X_seq.shape
    target_len = int(seq_len * slice_ratio)
    out = np.zeros_like(X_seq)

    for s in range(n_samples):
        start = rng.integers(0, seq_len - target_len + 1)
        sliced = X_seq[s, start:start + target_len]
        for f in range(n_features):
            out[s, :, f] = np.interp(
                np.linspace(0, target_len - 1, seq_len),
                np.arange(target_len),
                sliced[:, f],
            )
    return out[0] if single else out


def permutation(X_seq, n_segments=4, rng=None):
    """Split sequence into segments and permute their order."""
    rng = rng or np.random.default_rng(RANDOM_SEED + 30)
    single = (X_seq.ndim == 2)
    if single:
        X_seq = X_seq[None, ...]

    n_samples, seq_len, n_features = X_seq.shape
    out = np.zeros_like(X_seq)
    seg_len = seq_len // n_segments

    for s in range(n_samples):
        idx = rng.permutation(n_segments)
        for new_i, old_i in enumerate(idx):
            out[s, new_i * seg_len:(new_i + 1) * seg_len] = X_seq[s, old_i * seg_len:(old_i + 1) * seg_len]
        # Copy leftover tail
        out[s, n_segments * seg_len:] = X_seq[s, n_segments * seg_len:]
    return out[0] if single else out


def apply_temporal_augmentation(X_seq, y, techniques=("time_warp", "magnitude_warp"),
                                 verbose=True):
    """Apply a set of temporal augmentation techniques."""
    rng = np.random.default_rng(RANDOM_SEED)
    all_X = [X_seq]
    all_y = [y]

    funcs = {
        "time_warp": time_warp,
        "magnitude_warp": magnitude_warp,
        "window_slice": window_slice,
        "permutation": permutation,
    }

    for tech in techniques:
        if tech not in funcs:
            continue
        X_aug = funcs[tech](X_seq, rng=rng)
        all_X.append(X_aug)
        all_y.append(y.copy())

    X_out = np.concatenate(all_X, axis=0)
    y_out = np.concatenate(all_y, axis=0)

    if verbose:
        print(f"[TemporalAug] Applied {techniques} -> {len(y_out)} samples")

    return X_out, y_out
