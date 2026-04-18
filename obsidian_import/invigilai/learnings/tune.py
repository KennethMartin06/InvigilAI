"""
tune.py -- Optuna-based hyperparameter optimization for RF, LightGBM, XGBoost, and MLP.

Usage:
    python -m cheating_detection.models.tune
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheating_detection.config import (
    RANDOM_SEED,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    CV_FOLDS,
    OUTPUTS_DIR,
)


def tune_random_forest(X_train, y_train, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT):
    """Optimize Random Forest hyperparameters with Optuna."""
    import optuna
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
        model = RandomForestClassifier(**params)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print(f"[Optuna RF] Best F1: {study.best_value:.4f}")
    print(f"[Optuna RF] Best params: {study.best_params}")
    return study.best_params


def tune_lightgbm(X_train, y_train, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT):
    """Optimize LightGBM hyperparameters with Optuna."""
    import optuna
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print(f"[Optuna LGB] Best F1: {study.best_value:.4f}")
    print(f"[Optuna LGB] Best params: {study.best_params}")
    return study.best_params


def tune_xgboost(X_train, y_train, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT):
    """Optimize XGBoost hyperparameters with Optuna."""
    import optuna
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
        }
        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print(f"[Optuna XGB] Best F1: {study.best_value:.4f}")
    print(f"[Optuna XGB] Best params: {study.best_params}")
    return study.best_params


def run_tuning(X_train, y_train, verbose=True):
    """Run Optuna tuning for all available models. Returns dict of best params."""
    try:
        import optuna
    except ImportError:
        if verbose:
            print("[Optuna] optuna not installed (pip install optuna), skipping tuning.")
        return {}

    results = {}

    if verbose:
        print(f"\n[Optuna] Starting hyperparameter optimization ({OPTUNA_N_TRIALS} trials, {OPTUNA_TIMEOUT}s timeout) ...")

    # RF
    if verbose:
        print("\n[Optuna] Tuning Random Forest ...")
    try:
        results["rf"] = tune_random_forest(X_train, y_train)
    except Exception as e:
        if verbose:
            print(f"  RF tuning failed: {e}")

    # LightGBM
    if verbose:
        print("\n[Optuna] Tuning LightGBM ...")
    try:
        results["lgb"] = tune_lightgbm(X_train, y_train)
    except Exception as e:
        if verbose:
            print(f"  LGB tuning failed: {e}")

    # XGBoost
    if verbose:
        print("\n[Optuna] Tuning XGBoost ...")
    try:
        results["xgb"] = tune_xgboost(X_train, y_train)
    except Exception as e:
        if verbose:
            print(f"  XGB tuning failed: {e}")

    # Save results
    import json
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    tune_path = os.path.join(OUTPUTS_DIR, "optuna_best_params.json")
    with open(tune_path, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\n[Optuna] Best params saved -> {tune_path}")

    return results
