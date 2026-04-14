"""
config.py -- Central configuration for all hyperparameters and constants.
"""

import os

# -- Reproducibility --------------------------------------------------------
RANDOM_SEED = 42

# -- Dataset ----------------------------------------------------------------
N_SESSIONS = 240

# Class distribution
CLASS_COUNTS = {
    0: 102,   # Normal Behavior
    1: 58,    # Gaze / Distraction
    2: 42,    # External Device Use
    3: 24,    # Multi-Person / Impersonation
    4: 14,    # Abnormal Keystroke Behavior
}
CLASS_NAMES = [
    "Normal",
    "Gaze/Distraction",
    "External Device",
    "Multi-Person",
    "Abnormal Keystroke",
]

# Windows per session (~2-second windows)
WINDOWS_PER_SESSION_MIN = 15
WINDOWS_PER_SESSION_MAX = 30

# -- Features ---------------------------------------------------------------
N_VISUAL_FEATURES = 8
N_BEHAVIORAL_FEATURES = 8
N_TOTAL_FEATURES = N_VISUAL_FEATURES + N_BEHAVIORAL_FEATURES

VISUAL_FEATURE_NAMES = [
    "gaze_yaw",
    "gaze_pitch",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "face_count",
    "gaze_deviation_ratio",
    "face_embedding_norm",
]

BEHAVIORAL_FEATURE_NAMES = [
    "keystroke_rate",
    "mean_dwell_time",
    "mean_flight_time",
    "burst_coefficient",
    "cursor_velocity",
    "click_frequency",
    "idle_ratio",
    "trajectory_linearity",
]

ALL_FEATURE_NAMES = VISUAL_FEATURE_NAMES + BEHAVIORAL_FEATURE_NAMES

# -- Train / Val / Test Split -----------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# -- SVM --------------------------------------------------------------------
SVM_PARAM_GRID = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", "auto"],
}

# -- Random Forest -----------------------------------------------------------
RF_N_ESTIMATORS = 300

# -- MLP (PyTorch) -----------------------------------------------------------
MLP_HIDDEN_LAYERS = [256, 128, 64]
MLP_DROPOUT = 0.3
MLP_LR = 0.001
MLP_BATCH_SIZE = 64
MLP_MAX_EPOCHS = 150
MLP_PATIENCE = 20
MLP_USE_BATCH_NORM = True
MLP_USE_CLASS_WEIGHTS = True
MLP_LR_SCHEDULER = True

# -- Cross-Validation -------------------------------------------------------
CV_FOLDS = 5

# -- Thresholding ------------------------------------------------------------
CHEATING_THRESHOLD = 0.45
THRESHOLD_RANGE_START = 0.30
THRESHOLD_RANGE_END = 0.95
THRESHOLD_STEP = 0.05

# -- SMOTE -------------------------------------------------------------------
USE_SMOTE = True

# -- Paths -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

DATASET_NPZ = os.path.join(DATA_DIR, "synthetic_dataset.npz")
DATASET_CSV = os.path.join(DATA_DIR, "synthetic_dataset.csv")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.joblib")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.joblib")
MLP_MODEL_PATH = os.path.join(MODELS_DIR, "mlp_model.pth")
ENSEMBLE_MODEL_PATH = os.path.join(MODELS_DIR, "ensemble_model.joblib")
RESULTS_JSON = os.path.join(OUTPUTS_DIR, "results.json")

CONFUSION_MATRIX_PNG = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
TRAINING_CURVES_PNG = os.path.join(OUTPUTS_DIR, "training_curves.png")
ABLATION_PNG = os.path.join(OUTPUTS_DIR, "ablation_results.png")
THRESHOLD_PNG = os.path.join(OUTPUTS_DIR, "threshold_analysis.png")
ROC_PNG = os.path.join(OUTPUTS_DIR, "roc_curve.png")
PR_CURVE_PNG = os.path.join(OUTPUTS_DIR, "precision_recall_curve.png")
