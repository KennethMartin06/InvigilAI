---
title: Tabular Preprocessing Template — InvigilAI
project: invigilai
tags: [ml, preprocessing, data, learnings]
source: cheating_detection/preprocessing/preprocess.py
---

# Tabular Preprocessing Template

Full pipeline: raw → clean → features → augment → synthesize → balance → scale.

## Order of operations matters

```
1. replace_invalid_values      (KNN impute, k=5)
2. add_derived_features         (16 base → 32 total)
3. split_dataset                (70/15/15 stratified)
4. apply_temporal_augmentation  (time warp, magnitude warp, window slice) ← train only
5. apply_gan_synthesis          (WGAN-GP, 200 samples/class)              ← train only
6. apply_noisy_student          (3 iterations, 0.80 confidence)           ← train only
7. augment_minority_classes     (noise + feature dropout)                 ← train only
8. apply_smote                  (Borderline-SMOTE + ENN cleaning)         ← train only
9. fit_scaler                   (RobustScaler)
10. apply_feature_selection     (top-K mutual information, optional)
```

**Rule:** anything that creates new samples only runs on training data. Scaling is fit on train, applied to val/test.

## Derived features — the 16 that matter

Grouped by what they encode:

**Coupling / divergence (cross-modal)**
- `gaze_head_coupling` = gaze_speed × head_mag
- `head_gaze_divergence` = |head_mag − gaze_speed|

**Rhythm / regularity (keystroke)**
- `typing_rhythm` = dwell / flight
- `keystroke_irregularity` = burst × rate
- `keystroke_variability` = |dwell − flight| / (dwell + flight)

**Attention / focus**
- `gaze_fixation_score` = 1 / (gaze_speed + 1)
- `focus_score` = (1 − gaze_dev_ratio)(1 − idle_ratio)
- `behavioral_entropy` = entropy over 8 behavioral features

**Complexity / deviation**
- `trajectory_deviation` = (1 − linearity) × cursor_velocity
- `movement_complexity` = trajectory_dev × head_mag

**Activity / engagement**
- `interaction_intensity` = rate × click_freq
- `engagement_score` = (1 − idle) × rate
- `activity_imbalance` = cursor_velocity / rate
- `suspicious_idle_pattern` = idle × burst

## GAN synthesis — conditional WGAN-GP

Generates per-class synthetic tabular samples. Use when:
- Minority class < 100 samples
- SMOTE is producing implausible interpolations

Config: latent_dim=32, hidden=128, epochs=200, batch=64, lr=0.0002.

## NoisyStudent — semi-supervised boost

1. Teacher predicts on unlabeled data
2. Keep samples with `max_prob ≥ 0.80`
3. Add Gaussian noise to features (σ=0.05)
4. Student trains on labeled + pseudo-labeled
5. Repeat 3×

Works best when unlabeled:labeled ratio ≈ 2:1.

## Reuse checklist

- [ ] Map your base features → edit `add_derived_features`
- [ ] Set `USE_*` flags in config.py to toggle stages on/off
- [ ] Start with just SMOTE; add GAN only if min-class < 50 samples
- [ ] Run preprocessing once with `verbose=True` and save `splits` — reload for every experiment
