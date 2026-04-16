"""
augment.py -- Data augmentation for tabular cheating detection features.

Techniques:
  - Gaussian noise injection
  - Feature dropout (random zeroing)
  - Temporal jittering (small random shifts)
  - Minority-class targeted augmentation
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    AUGMENTATION_NOISE_STD,
    AUGMENTATION_FEATURE_DROPOUT_RATE,
    AUGMENTATION_MULTIPLIER,
)


def augment_with_noise(X, y, noise_std=AUGMENTATION_NOISE_STD, rng=None):
    """Add Gaussian noise to all features."""
    rng = rng or np.random.default_rng(RANDOM_SEED)
    noise = rng.normal(0, noise_std, X.shape)
    X_aug = X + noise
    return X_aug, y.copy()


def augment_with_feature_dropout(X, y, dropout_rate=AUGMENTATION_FEATURE_DROPOUT_RATE, rng=None):
    """Randomly zero out features to simulate missing data."""
    rng = rng or np.random.default_rng(RANDOM_SEED + 1)
    mask = rng.random(X.shape) > dropout_rate
    X_aug = X * mask
    return X_aug, y.copy()


def augment_with_jitter(X, y, scale=0.02, rng=None):
    """Small random multiplicative jittering."""
    rng = rng or np.random.default_rng(RANDOM_SEED + 2)
    jitter = 1.0 + rng.normal(0, scale, X.shape)
    X_aug = X * jitter
    return X_aug, y.copy()


def augment_minority_classes(X_train, y_train, multiplier=AUGMENTATION_MULTIPLIER,
                              verbose=True):
    """Augment minority classes with noise, dropout, and jitter.

    Only augments classes with fewer samples than the median class size.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    unique, counts = np.unique(y_train, return_counts=True)
    median_count = np.median(counts)

    all_X = [X_train]
    all_y = [y_train]
    total_added = 0

    for cls, cnt in zip(unique, counts):
        if cnt >= median_count:
            continue

        mask = y_train == cls
        X_cls = X_train[mask]

        for _ in range(multiplier):
            # Noise augmentation
            X_n, y_n = augment_with_noise(X_cls, np.full(len(X_cls), cls), rng=rng)
            all_X.append(X_n)
            all_y.append(y_n)

            # Feature dropout
            X_d, y_d = augment_with_feature_dropout(X_cls, np.full(len(X_cls), cls), rng=rng)
            all_X.append(X_d)
            all_y.append(y_d)

            # Jitter
            X_j, y_j = augment_with_jitter(X_cls, np.full(len(X_cls), cls), rng=rng)
            all_X.append(X_j)
            all_y.append(y_j)

            total_added += len(X_cls) * 3

    X_aug = np.vstack(all_X)
    y_aug = np.concatenate(all_y)

    # Shuffle
    perm = rng.permutation(len(y_aug))
    X_aug = X_aug[perm]
    y_aug = y_aug[perm]

    if verbose:
        print(f"[Augment] Added {total_added} augmented samples ({multiplier}x for minority)")
        unique2, counts2 = np.unique(y_aug, return_counts=True)
        for cls, cnt in zip(unique2, counts2):
            print(f"  Class {cls}: {cnt}")

    return X_aug, y_aug
