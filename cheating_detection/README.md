# Multi-Modal AI Cheating Detection System

A complete Python project for detecting cheating in online examinations by fusing **visual features** (eye gaze, head pose, face count) with **behavioral features** (keystroke dynamics, mouse trajectories) into a single classification pipeline.

## Features

- Synthetic dataset generation for 240 exam sessions across 5 cheating categories
- 16-dimensional fused feature vector (8 visual + 8 behavioral)
- Three trained classifiers: SVM, Random Forest, and MLP (PyTorch)
- 5-fold stratified cross-validation
- Confusion matrix, ROC curve, Precision-Recall curve
- Ablation study comparing visual-only, behavioral-only, and multi-modal fusion
- Threshold sweep analysis for binary cheating detection

## Cheating Categories

| Label | Category                    | Sessions |
|-------|-----------------------------|----------|
| 0     | Normal Behavior             | 102      |
| 1     | Gaze / Distraction          | 58       |
| 2     | External Device Use         | 42       |
| 3     | Multi-Person / Impersonation| 24       |
| 4     | Abnormal Keystroke Behavior | 14       |

## Project Structure

```
cheating_detection/
├── data/
│   └── generate_dataset.py       # Synthetic data generator
├── preprocessing/
│   └── preprocess.py             # Cleaning, normalization, windowing
├── features/
│   ├── visual_features.py        # Visual feature extraction
│   └── behavioral_features.py    # Keystroke + mouse feature extraction
├── models/
│   ├── train.py                  # SVM, RF, MLP training pipeline
│   ├── evaluate.py               # Evaluation, ablation, curves
│   └── model_utils.py            # Saving, loading, metrics
├── config.py                     # All hyperparameters and constants
├── main.py                       # End-to-end entry point
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python -m cheating_detection.main
```

## Output Artifacts

After running `main.py` the following files are generated:

| File | Description |
|------|-------------|
| `data/synthetic_dataset.npz` | Compressed numpy dataset |
| `data/synthetic_dataset.csv` | Human-readable dataset |
| `models/scaler.joblib` | Fitted StandardScaler |
| `models/svm_model.joblib` | Trained SVM |
| `models/rf_model.joblib` | Trained Random Forest |
| `models/mlp_model.pth` | Trained MLP weights |
| `outputs/confusion_matrix.png` | Confusion matrix heatmap |
| `outputs/training_curves.png` | MLP loss/accuracy curves |
| `outputs/ablation_results.png` | Modality comparison bar chart |
| `outputs/threshold_analysis.png` | Precision/Recall vs threshold |
| `outputs/roc_curve.png` | ROC curve with AUC |
| `outputs/precision_recall_curve.png` | PR curve with average precision |
| `outputs/results.json` | All numeric results |

## Feature Descriptions

### Visual Features (8 dimensions)
| Feature | Description |
|---------|-------------|
| `gaze_yaw` | Horizontal gaze angle (degrees) |
| `gaze_pitch` | Vertical gaze angle (degrees) |
| `head_yaw` | Head horizontal rotation (degrees) |
| `head_pitch` | Head vertical tilt (degrees) |
| `head_roll` | Head roll angle (degrees) |
| `face_count` | Number of detected faces in frame |
| `gaze_deviation_ratio` | Fraction of frames with gaze > ±25° |
| `face_embedding_norm` | L2 norm of ResNet-50 facial embedding |

### Behavioral Features (8 dimensions)
| Feature | Description |
|---------|-------------|
| `keystroke_rate` | Keys per second |
| `mean_dwell_time` | Average key-hold duration (ms) |
| `mean_flight_time` | Average inter-key interval (ms) |
| `burst_coefficient` | Variance/mean of inter-key intervals |
| `cursor_velocity` | Mean cursor speed (px/s) |
| `click_frequency` | Mouse clicks per second |
| `idle_ratio` | Fraction of window with zero input |
| `trajectory_linearity` | Straight-line / total path ratio |

## Configuration

All hyperparameters are centralised in `config.py`:

```python
RANDOM_SEED = 42
N_SESSIONS = 240
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MLP_HIDDEN_LAYERS = [128, 64]
MLP_DROPOUT = 0.3
MLP_LR = 0.001
MLP_MAX_EPOCHS = 80
MLP_PATIENCE = 10
RF_N_ESTIMATORS = 100
CHEATING_THRESHOLD = 0.70
```

## Extending to Real Data

To use real proctoring data instead of synthetic data:

1. Replace `data/generate_dataset.py` with your data loader that produces the same `(X, y)` arrays.
2. Update `features/visual_features.py` to accept raw frame sequences and call the provided extraction functions.
3. Update `features/behavioral_features.py` to accept raw event streams (keydown/keyup timestamps, mouse coordinates).
4. All downstream preprocessing, training, and evaluation code remains unchanged.
