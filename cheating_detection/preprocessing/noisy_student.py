"""
noisy_student.py -- Iterative self-training via NoisyStudent.

Trains a teacher on labeled data, uses it to pseudo-label unlabeled data,
then trains a student on the combined set with stronger noise. Repeat.
"""

import numpy as np
from sklearn.base import clone

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    NOISY_STUDENT_ITERATIONS,
    NOISY_STUDENT_UNLABELED_RATIO,
    NOISY_STUDENT_CONFIDENCE_THRESHOLD,
    AUGMENTATION_NOISE_STD,
)


def _add_noise(X, std, rng):
    return X + rng.normal(0, std, X.shape)


def _dropout(X, rate, rng):
    mask = rng.random(X.shape) > rate
    return X * mask


class NoisyStudentTrainer:
    """Iterative NoisyStudent self-training."""

    def __init__(self, base_model_factory,
                 iterations=NOISY_STUDENT_ITERATIONS,
                 confidence_threshold=NOISY_STUDENT_CONFIDENCE_THRESHOLD,
                 noise_std=AUGMENTATION_NOISE_STD):
        self.factory = base_model_factory
        self.iterations = iterations
        self.confidence_threshold = confidence_threshold
        self.noise_std = noise_std
        self.final_model = None
        self.history = []

    def fit(self, X_labeled, y_labeled, X_unlabeled, verbose=True):
        rng = np.random.default_rng(RANDOM_SEED)

        teacher = self.factory()
        teacher.fit(X_labeled, y_labeled)

        X_combined = X_labeled.copy()
        y_combined = y_labeled.copy()

        for it in range(self.iterations):
            if hasattr(teacher, "predict_proba"):
                probs = teacher.predict_proba(X_unlabeled)
            else:
                # Fallback: pseudo-probabilities from hard predictions
                preds = teacher.predict(X_unlabeled)
                probs = np.eye(int(preds.max() + 1))[preds.astype(int)]

            pseudo_conf = probs.max(axis=1)
            pseudo_labels = probs.argmax(axis=1)
            keep = pseudo_conf >= self.confidence_threshold

            n_kept = int(keep.sum())
            if verbose:
                print(f"[NoisyStudent iter {it+1}] pseudo-kept={n_kept}/{len(X_unlabeled)} "
                      f"(mean conf={pseudo_conf.mean():.3f})")

            if n_kept == 0:
                break

            X_pseudo = X_unlabeled[keep]
            y_pseudo = pseudo_labels[keep]

            # Inject noise into student training data
            X_noisy_lab = _add_noise(X_labeled, self.noise_std * (1 + it * 0.3), rng)
            X_noisy_pse = _dropout(_add_noise(X_pseudo, self.noise_std * (1 + it * 0.3), rng),
                                   rate=0.1, rng=rng)

            X_student = np.vstack([X_noisy_lab, X_noisy_pse])
            y_student = np.concatenate([y_labeled, y_pseudo])

            student = self.factory()
            student.fit(X_student, y_student)

            self.history.append({
                "iteration": it + 1,
                "n_pseudo_kept": n_kept,
                "mean_confidence": float(pseudo_conf.mean()),
                "pseudo_label_dist": dict(zip(
                    *[arr.tolist() for arr in np.unique(y_pseudo, return_counts=True)]
                )),
            })

            teacher = student
            X_combined = X_student
            y_combined = y_student

        self.final_model = teacher
        self.X_combined = X_combined
        self.y_combined = y_combined
        if verbose:
            print(f"[NoisyStudent] Completed {len(self.history)} iterations")
        return X_combined, y_combined

    def predict(self, X):
        return self.final_model.predict(X)

    def predict_proba(self, X):
        return self.final_model.predict_proba(X)


def generate_unlabeled_pool(X_labeled, ratio=NOISY_STUDENT_UNLABELED_RATIO, rng=None):
    """Bootstrap an unlabeled pool by adding noise to labeled samples.

    Intended for scenarios where true unlabeled data isn't available: treat
    noisy copies of labeled data as the unlabeled pool.
    """
    rng = rng or np.random.default_rng(RANDOM_SEED)
    n_unlabeled = int(len(X_labeled) * ratio)
    idx = rng.integers(0, len(X_labeled), n_unlabeled)
    X_u = X_labeled[idx]
    noise = rng.normal(0, 0.1, X_u.shape)
    return X_u + noise
