"""
config.py -- Central configuration for all hyperparameters and constants.

v5: 16 derived features (32 total), SMOTE-ENN, data augmentation, temperature
    scaling, conformal prediction, OOD detection, concept drift, cost-sensitive
    evaluation, adversarial robustness, cascade classifier.
"""

import os

# -- Reproducibility --------------------------------------------------------
RANDOM_SEED = 42

# -- Dataset ----------------------------------------------------------------
N_SESSIONS = 240

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

WINDOWS_PER_SESSION_MIN = 15
WINDOWS_PER_SESSION_MAX = 30

# -- Features ---------------------------------------------------------------
N_VISUAL_FEATURES = 8
N_BEHAVIORAL_FEATURES = 8
N_DERIVED_FEATURES = 16   # 10 original + 6 new
N_BASE_FEATURES = N_VISUAL_FEATURES + N_BEHAVIORAL_FEATURES      # 16
N_TOTAL_FEATURES = N_BASE_FEATURES + N_DERIVED_FEATURES            # 32

VISUAL_FEATURE_NAMES = [
    "gaze_yaw", "gaze_pitch", "head_yaw", "head_pitch",
    "head_roll", "face_count", "gaze_deviation_ratio", "face_embedding_norm",
]

BEHAVIORAL_FEATURE_NAMES = [
    "keystroke_rate", "mean_dwell_time", "mean_flight_time", "burst_coefficient",
    "cursor_velocity", "click_frequency", "idle_ratio", "trajectory_linearity",
]

DERIVED_FEATURE_NAMES = [
    # Original 4
    "gaze_speed",                # sqrt(yaw^2 + pitch^2)
    "head_movement_magnitude",   # sqrt(head_yaw^2 + head_pitch^2 + head_roll^2)
    "keystroke_irregularity",    # burst_coefficient * keystroke_rate
    "activity_imbalance",        # cursor_velocity / (keystroke_rate + eps)
    # v4 batch (6)
    "typing_rhythm",             # mean_dwell_time / (mean_flight_time + eps)
    "interaction_intensity",     # keystroke_rate * click_frequency
    "gaze_head_coupling",        # gaze_speed * head_movement_magnitude
    "suspicious_idle_pattern",   # idle_ratio * burst_coefficient
    "trajectory_deviation",      # (1 - trajectory_linearity) * cursor_velocity
    "engagement_score",          # (1 - idle_ratio) * keystroke_rate
    # v5 batch (6)
    "gaze_fixation_score",       # 1 / (gaze_speed + 1)
    "head_gaze_divergence",      # |head_mag - gaze_speed|
    "keystroke_variability",     # |dwell - flight| / (dwell + flight + eps)
    "movement_complexity",       # trajectory_deviation * head_mag
    "focus_score",               # (1 - gaze_dev_ratio) * (1 - idle_ratio)
    "behavioral_entropy",        # entropy of normalized behavioral features
]

ALL_FEATURE_NAMES = VISUAL_FEATURE_NAMES + BEHAVIORAL_FEATURE_NAMES + DERIVED_FEATURE_NAMES

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

# -- LightGBM ---------------------------------------------------------------
LGB_N_ESTIMATORS = 300
LGB_LEARNING_RATE = 0.05
LGB_MAX_DEPTH = 7
LGB_NUM_LEAVES = 63

# -- XGBoost -----------------------------------------------------------------
XGB_N_ESTIMATORS = 300
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH = 6

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
MLP_LABEL_SMOOTHING = 0.1

# -- Focal Loss --------------------------------------------------------------
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = None   # None = use class weights, or list of per-class alphas

# -- MixUp -------------------------------------------------------------------
USE_MIXUP = True
MIXUP_ALPHA = 0.4

# -- Cosine Annealing --------------------------------------------------------
USE_COSINE_ANNEALING = True
COSINE_T_0 = 20       # restart every 20 epochs
COSINE_T_MULT = 2     # double the period after each restart

# -- Gradient Clipping -------------------------------------------------------
USE_GRADIENT_CLIPPING = True
GRADIENT_CLIP_NORM = 1.0

# -- Stochastic Weight Averaging (SWA) --------------------------------------
USE_SWA = True
SWA_START_EPOCH = 100   # start averaging after this epoch
SWA_LR = 0.0005         # fixed LR for SWA phase

# -- MC Dropout (uncertainty estimation) -------------------------------------
USE_MC_DROPOUT = True
MC_DROPOUT_SAMPLES = 30

# -- Cross-Validation -------------------------------------------------------
CV_FOLDS = 5

# -- Thresholding ------------------------------------------------------------
CHEATING_THRESHOLD = 0.45
THRESHOLD_RANGE_START = 0.30
THRESHOLD_RANGE_END = 0.95
THRESHOLD_STEP = 0.05

# -- SMOTE -------------------------------------------------------------------
USE_SMOTE = True
USE_BORDERLINE_SMOTE = True   # use Borderline-SMOTE instead of basic SMOTE
USE_SMOTE_ENN = True          # apply ENN cleaning after SMOTE

# -- Data Augmentation -------------------------------------------------------
USE_DATA_AUGMENTATION = True
AUGMENTATION_NOISE_STD = 0.05
AUGMENTATION_FEATURE_DROPOUT_RATE = 0.1
AUGMENTATION_MULTIPLIER = 2   # how many augmented copies per minority sample

# -- Preprocessing -----------------------------------------------------------
USE_KNN_IMPUTATION = True
KNN_IMPUTE_NEIGHBORS = 5
USE_ROBUST_SCALER = True

# -- Feature Selection -------------------------------------------------------
USE_FEATURE_SELECTION = False   # mutual-information based feature selection
FEATURE_SELECTION_K = 28        # top K features to keep

# -- SHAP --------------------------------------------------------------------
USE_SHAP = True

# -- Optuna ------------------------------------------------------------------
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = 600   # seconds

# -- Temperature Scaling (post-hoc calibration) ------------------------------
USE_TEMPERATURE_SCALING = True

# -- Conformal Prediction ---------------------------------------------------
USE_CONFORMAL_PREDICTION = True
CONFORMAL_ALPHA = 0.10   # target error rate (90% coverage)

# -- OOD Detection -----------------------------------------------------------
USE_OOD_DETECTION = True
OOD_PERCENTILE = 95   # samples above this Mahalanobis percentile are OOD

# -- Concept Drift -----------------------------------------------------------
USE_DRIFT_DETECTION = True
DRIFT_PSI_THRESHOLD = 0.2   # Population Stability Index threshold

# -- Cost-Sensitive Evaluation -----------------------------------------------
# Cost of false negative (missed cheater) vs false positive (wrongly accused)
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0

# -- Adversarial Robustness -------------------------------------------------
USE_ADVERSARIAL_TEST = True
ADVERSARIAL_EPSILON = 0.1   # FGSM perturbation magnitude

# -- Cascade Classifier ------------------------------------------------------
USE_CASCADE_CLASSIFIER = True
CASCADE_CONFIDENCE_THRESHOLD = 0.85   # below this -> second-stage review

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
LGB_MODEL_PATH = os.path.join(MODELS_DIR, "lgb_model.joblib")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.joblib")
MLP_MODEL_PATH = os.path.join(MODELS_DIR, "mlp_model.pth")
ENSEMBLE_MODEL_PATH = os.path.join(MODELS_DIR, "ensemble_model.joblib")
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "mlp_model.onnx")
RESULTS_JSON = os.path.join(OUTPUTS_DIR, "results.json")

CONFUSION_MATRIX_PNG = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
TRAINING_CURVES_PNG = os.path.join(OUTPUTS_DIR, "training_curves.png")
ABLATION_PNG = os.path.join(OUTPUTS_DIR, "ablation_results.png")
THRESHOLD_PNG = os.path.join(OUTPUTS_DIR, "threshold_analysis.png")
ROC_PNG = os.path.join(OUTPUTS_DIR, "roc_curve.png")
PR_CURVE_PNG = os.path.join(OUTPUTS_DIR, "precision_recall_curve.png")
SHAP_PNG = os.path.join(OUTPUTS_DIR, "shap_feature_importance.png")
CALIBRATION_PNG = os.path.join(OUTPUTS_DIR, "calibration_curve.png")
UNCERTAINTY_PNG = os.path.join(OUTPUTS_DIR, "uncertainty_analysis.png")
FEATURE_IMPORTANCE_PNG = os.path.join(OUTPUTS_DIR, "feature_importance_comparison.png")
OOD_PNG = os.path.join(OUTPUTS_DIR, "ood_detection.png")
DRIFT_PNG = os.path.join(OUTPUTS_DIR, "drift_analysis.png")
ADVERSARIAL_PNG = os.path.join(OUTPUTS_DIR, "adversarial_robustness.png")
CASCADE_PNG = os.path.join(OUTPUTS_DIR, "cascade_analysis.png")
COST_ANALYSIS_PNG = os.path.join(OUTPUTS_DIR, "cost_analysis.png")
