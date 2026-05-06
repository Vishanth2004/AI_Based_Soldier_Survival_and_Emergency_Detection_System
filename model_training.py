"""
AI-Based Soldier Survival & Emergency Detection System
MODEL TRAINING PIPELINE

Trains 4 ML models, evaluates them thoroughly, and saves the best one.

Models:
  1. Random Forest
  2. XGBoost (Gradient Boosting)
  3. Support Vector Machine (SVM)
  4. Multi-Layer Perceptron (Neural Network)

Usage:
  1. Collect data using data_collection.py → soldier_data.csv
  2. Run: python model_training.py
  3. Best model saved to: models/best_model.joblib
  4. Use with realtime_dashboard.py for live inference

Output:
  - models/best_model.joblib          (trained model)
  - models/scaler.joblib              (feature scaler)
  - models/label_encoder.joblib       (label encoder)
  - models/feature_names.joblib       (feature list)
  - models/training_report.txt        (full comparison report)
  - models/confusion_matrices.png     (visual comparison)
  - models/feature_importance.png     (what the model learned)
  - models/roc_curves.png             (ROC curves per class)
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import time
import warnings
from datetime import datetime

# ML
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_auc_score, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight

# Plotting
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Joblib for saving
import joblib

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: XGBoost not installed. Install with: pip install xgboost")
    print("         Will train 3 models instead of 4.\n")

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DATA_FILE = "soldier_data.csv"
MODEL_DIR = "models"

# Features to use for ML (from DATA_COLLECTION_PROTOCOL.md)
ML_FEATURES = [
    "bpm",
    "hrv_sdnn",
    "hrv_rmssd",
    "dynamic_accel",
    "impact",
    "pitch",
    "roll",
    "gx",
    "gy",
    "gz",
    "movement_var",
    "bpm_mean_10s",
    "bpm_std_10s",
    "dynamic_accel_mean_5s",
    "dynamic_accel_max_5s",
    "impact_max_5s",
    "pitch_mean_5s",
    "movement_var_mean_5s",
    "gyro_magnitude_mean_5s",
]

LABEL_COL = "label"
VALID_LABELS = ["normal", "high_exertion", "man_down"]

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cross-validation folds
CV_FOLDS = 5

# ==================== HELPER FUNCTIONS ====================

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title):
    print(f"\n--- {title} ---")


# ==================== STEP 1: LOAD & CLEAN DATA ====================

def load_and_clean_data(filepath):
    """Load CSV, clean it, and return features + labels."""
    print_header("STEP 1: LOADING & CLEANING DATA")

    if not os.path.exists(filepath):
        print(f"ERROR: Data file '{filepath}' not found!")
        print("Run data_collection.py first to collect sensor data.")
        sys.exit(1)

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    print(f"Columns: {list(df.columns)}")

    # Check for required columns
    missing_features = [f for f in ML_FEATURES if f not in df.columns]
    if missing_features:
        print(f"ERROR: Missing features in data: {missing_features}")
        sys.exit(1)

    if LABEL_COL not in df.columns:
        print(f"ERROR: Label column '{LABEL_COL}' not found!")
        sys.exit(1)

    # Show class distribution before cleaning
    print_section("Class Distribution (Raw)")
    print(df[LABEL_COL].value_counts().to_string())

    # Filter to valid labels only
    df = df[df[LABEL_COL].isin(VALID_LABELS)].copy()
    print(f"\nAfter filtering to valid labels: {len(df)} rows")

    # Remove rows where ECG lead was off (bad ECG data)
    if "ecg_lead_off" in df.columns:
        lead_off_count = df["ecg_lead_off"].sum()
        df = df[df["ecg_lead_off"] == 0].copy()
        print(f"Removed {lead_off_count} rows with ECG lead-off")

    # Remove first 10 seconds of each session (startup artifacts)
    if "session_id" in df.columns and "timestamp" in df.columns:
        rows_before = len(df)
        cleaned_dfs = []
        for session in df["session_id"].unique():
            session_df = df[df["session_id"] == session].copy()
            session_start = session_df["timestamp"].min()
            session_df = session_df[session_df["timestamp"] >= session_start + 10]
            cleaned_dfs.append(session_df)
        if cleaned_dfs:
            df = pd.concat(cleaned_dfs, ignore_index=True)
        print(f"Removed {rows_before - len(df)} startup rows (first 10s per session)")

    # Drop rows with NaN in ML features
    nan_before = len(df)
    df = df.dropna(subset=ML_FEATURES)
    if nan_before - len(df) > 0:
        print(f"Removed {nan_before - len(df)} rows with NaN values")

    # Remove rows with all-zero windowed features (buffer not yet filled)
    windowed_cols = [
        "bpm_mean_10s", "bpm_std_10s", "dynamic_accel_mean_5s",
        "dynamic_accel_max_5s", "impact_max_5s", "pitch_mean_5s",
        "movement_var_mean_5s", "gyro_magnitude_mean_5s"
    ]
    existing_windowed = [c for c in windowed_cols if c in df.columns]
    if existing_windowed:
        all_zero_mask = (df[existing_windowed] == 0).all(axis=1)
        all_zero_count = all_zero_mask.sum()
        df = df[~all_zero_mask].copy()
        print(f"Removed {all_zero_count} rows with all-zero windowed features")

    # Remove stuck sensors (constant readings for >50 consecutive rows)
    # Simple check: if movement_var is exactly 0 for a long stretch in non-man_down data
    # (In man_down, zero movement is expected after a fall)

    # Final class distribution
    print_section("Class Distribution (Cleaned)")
    class_dist = df[LABEL_COL].value_counts()
    print(class_dist.to_string())
    print(f"\nTotal samples: {len(df)}")

    if len(df) < 100:
        print("\nWARNING: Very few samples! Model may not train well.")
        print("Collect more data following DATA_COLLECTION_PROTOCOL.md")

    # Check minimum per class
    min_class = class_dist.min()
    if min_class < 50:
        print(f"\nWARNING: Smallest class has only {min_class} samples.")
        print("Recommend at least 500+ per class for good results.")

    # Reset index so df.index == 0..N-1 (critical for numpy/pandas alignment)
    df = df.reset_index(drop=True)

    return df


# ==================== STEP 2: FEATURE ENGINEERING ====================

def prepare_features(df):
    """Extract features and labels. Scaling is applied later (train-only) to avoid leakage."""
    print_header("STEP 2: PREPARING FEATURES")

    X = df[ML_FEATURES].copy()
    y = df[LABEL_COL].copy()

    # Feature statistics
    print_section("Feature Statistics")
    print(X.describe().round(3).to_string())

    # Encode labels
    le = LabelEncoder()
    le.fit(VALID_LABELS)  # Consistent encoding order
    y_encoded = le.transform(y)

    print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    print(f"Feature matrix shape: {X.shape}")
    return X, y_encoded, le


def split_by_session_stratified(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split into train/test *by session_id* while keeping label balance.

    Why this matters:
      - Your data is time-series like; consecutive rows in a session are highly correlated.
      - Random row-wise split can inflate accuracy.
      - Splitting by session gives a more honest evaluation.
    """

    if "session_id" not in df.columns:
        return None

    # Each session in your collector is single-label; take the first label per session.
    sessions = (
        df[["session_id", LABEL_COL]]
        .drop_duplicates("session_id")
        .reset_index(drop=True)
    )

    # Build session lists per label
    rng = np.random.default_rng(random_state)
    test_sessions = []
    train_sessions = []

    for label in VALID_LABELS:
        label_sessions = sessions[sessions[LABEL_COL] == label]["session_id"].tolist()
        if not label_sessions:
            continue
        rng.shuffle(label_sessions)

        n_total = len(label_sessions)
        n_test = max(1, int(round(n_total * test_size))) if n_total >= 2 else 0

        if n_test == 0:
            # Only 1 session for this class — can't do honest session-wise split.
            # Fall back to row-wise split.
            print(f"  [WARN] Label '{label}' has only 1 session — session-wise split cannot evaluate it.")
            print(f"         Falling back to row-wise split for a complete evaluation.")
            return None

        test_sessions.extend(label_sessions[:n_test])
        train_sessions.extend(label_sessions[n_test:])

    # If we couldn't form a meaningful split (e.g. only 1 session per label), fallback.
    if not test_sessions or not train_sessions:
        return None

    train_mask = df["session_id"].isin(train_sessions)
    test_mask = df["session_id"].isin(test_sessions)

    # Ensure disjoint
    if (train_mask & test_mask).any():
        return None

    X_train_idx = df[train_mask].index
    X_test_idx = df[test_mask].index
    return X_train_idx, X_test_idx


def stratified_session_folds(df, n_splits=CV_FOLDS, random_state=RANDOM_STATE):
    """Yield CV folds by splitting on session_id (stratified by label)."""
    if "session_id" not in df.columns:
        return None

    sessions = (
        df[["session_id", LABEL_COL]]
        .drop_duplicates("session_id")
        .reset_index(drop=True)
    )

    # Need at least n_splits sessions in total AND per class for stratified split.
    if len(sessions) < n_splits:
        return None
    min_class_sessions = sessions[LABEL_COL].value_counts().min()
    if min_class_sessions < n_splits:
        print(f"  [INFO] Only {min_class_sessions} session(s) in smallest class "
              f"— cannot do {n_splits}-fold session-wise CV. Falling back to row-wise CV.")
        return None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    sess_ids = sessions["session_id"].to_numpy()
    sess_labels = sessions[LABEL_COL].to_numpy()

    for train_sess_idx, val_sess_idx in skf.split(sess_ids, sess_labels):
        train_sessions = set(sess_ids[train_sess_idx])
        val_sessions = set(sess_ids[val_sess_idx])
        train_idx = df[df["session_id"].isin(train_sessions)].index
        val_idx = df[df["session_id"].isin(val_sessions)].index
        yield train_idx, val_idx


# ==================== STEP 3: TRAIN MODELS ====================

def get_models(n_classes, class_weights_dict):
    """Return dict of model name → model instance."""
    models = {}

    # 1. Random Forest
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # 2. XGBoost
    if HAS_XGBOOST:
        # Compute sample weights for XGBoost
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1
        )

    # 3. SVM (with probability for ROC curves)
    models["SVM"] = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_STATE
    )

    # 4. Neural Network (MLP)
    models["Neural Network"] = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RANDOM_STATE
    )

    return models


def train_and_evaluate(models, X_train, X_test, y_train, y_test, le):
    """Train all models and return results."""
    print_header("STEP 3: TRAINING & EVALUATING MODELS")

    # Fit scaler ONLY on the training split (prevents leakage into test set)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Row-wise CV is acceptable when sessions are many and randomized,
    # but session-wise CV is more realistic if session_id exists.
    # We implement CV outside this function (see main) to avoid leakage.

    for name, model in models.items():
        print_section(f"Training: {name}")
        start = time.time()

        # Handle XGBoost sample weights for class imbalance
        if name == "XGBoost" and HAS_XGBOOST:
            classes = np.unique(y_train)
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            weight_dict = dict(zip(classes, weights))
            sample_weights = np.array([weight_dict[y] for y in y_train])
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train_scaled, y_train)

        train_time = time.time() - start

        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        f1_macro = f1_score(y_test, y_pred, average="macro")

        # CV scores are filled by caller (main) to avoid leakage in scaling.
        cv_scores = np.array([np.nan])

        # Per-class report
        report = classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            output_dict=True
        )
        report_str = classification_report(
            y_test, y_pred,
            target_names=le.classes_
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "scaler": scaler,
            "accuracy": acc,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "cv_mean": float("nan"),
            "cv_std": float("nan"),
            "y_pred": y_pred,
            "y_proba": y_proba,
            "confusion_matrix": cm,
            "report": report,
            "report_str": report_str,
            "train_time": train_time,
        }

        print(f"  Accuracy:       {acc:.4f}")
        print(f"  F1 (weighted):  {f1_weighted:.4f}")
        print(f"  F1 (macro):     {f1_macro:.4f}")
        print(f"  CV F1 (mean):   (computed after training)")
        print(f"  Training time:  {train_time:.2f}s")
        print(f"\n{report_str}")

    return results, scaler


def cross_validate_models(models, X_train, y_train, df_train=None):
    """Compute leakage-safe CV scores for each model.

    - Fits scaler inside each fold
    - Supports session-wise folds if df_train has session_id
    """
    print_header("CROSS-VALIDATION (LEAKAGE-SAFE)")

    if df_train is not None:
        folds = stratified_session_folds(df_train, n_splits=CV_FOLDS, random_state=RANDOM_STATE)
    else:
        folds = None

    if folds is None:
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_indices = list(skf.split(X_train, y_train))
        fold_iter = range(len(fold_indices))
        row_wise = True
    else:
        fold_indices = list(folds)
        fold_iter = range(len(fold_indices))
        row_wise = False


    # Pre-compute index-to-position map for session-wise folds
    if not row_wise:
        df_index = df_train.index
        index_to_pos = {idx: pos for pos, idx in enumerate(df_index)}

    cv_results = {}

    for name, model in models.items():
        print_section(f"CV: {name}")
        fold_scores = []

        for k in fold_iter:
            if row_wise:
                train_idx, val_idx = fold_indices[k]
                X_tr = X_train.iloc[train_idx]
                y_tr = y_train[train_idx]
                X_va = X_train.iloc[val_idx]
                y_va = y_train[val_idx]
            else:
                tr_idx, va_idx = fold_indices[k]
                tr_pos = [index_to_pos[i] for i in tr_idx]
                va_pos = [index_to_pos[i] for i in va_idx]
                X_tr = X_train.iloc[tr_pos]
                y_tr = y_train[tr_pos]
                X_va = X_train.iloc[va_pos]
                y_va = y_train[va_pos]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)

            m = clone(model)
            if name == "XGBoost" and HAS_XGBOOST:
                classes = np.unique(y_tr)
                weights = compute_class_weight("balanced", classes=classes, y=y_tr)
                wdict = dict(zip(classes, weights))
                sw = np.array([wdict[y] for y in y_tr])
                m.fit(X_tr_s, y_tr, sample_weight=sw)
            else:
                m.fit(X_tr_s, y_tr)

            y_hat = m.predict(X_va_s)
            fold_scores.append(f1_score(y_va, y_hat, average="weighted"))

        fold_scores = np.array(fold_scores, dtype=float)
        cv_results[name] = {
            "scores": fold_scores,
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
        }
        print(f"  CV F1 (weighted): {cv_results[name]['mean']:.4f} ± {cv_results[name]['std']:.4f}")

    return cv_results


# ==================== STEP 4: COMPARE & SELECT BEST ====================

def compare_and_select(results):
    """Compare models and return the best one."""
    print_header("STEP 4: MODEL COMPARISON")

    # Create comparison table
    comparison = []
    for name, res in results.items():
        comparison.append({
            "Model": name,
            "Accuracy": f"{res['accuracy']:.4f}",
            "F1 (Weighted)": f"{res['f1_weighted']:.4f}",
            "F1 (Macro)": f"{res['f1_macro']:.4f}",
            "CV F1 Mean": f"{res['cv_mean']:.4f}",
            "CV F1 Std": f"±{res['cv_std']:.4f}",
            "Train Time": f"{res['train_time']:.2f}s",
        })

    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))

    # Select best model based on CV F1 (most robust metric)
    # (cv_mean is populated in main after leakage-safe CV)
    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    best_result = results[best_name]

    print(f"\n{'*' * 50}")
    print(f"  BEST MODEL: {best_name}")
    print(f"  CV F1 Score: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
    print(f"  Test Accuracy: {best_result['accuracy']:.4f}")
    print(f"{'*' * 50}")

    return best_name, best_result


# ==================== STEP 5: VISUALIZATIONS ====================

def plot_confusion_matrices(results, le, save_path):
    """Plot confusion matrix for each model."""
    print_section("Generating confusion matrix plots")

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for idx, (name, res) in enumerate(results.items()):
        cm = res["confusion_matrix"]
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[idx], vmin=0, vmax=1
        )
        axes[idx].set_title(f"{name}\nAcc: {res['accuracy']:.3f} | F1: {res['f1_weighted']:.3f}")
        axes[idx].set_ylabel("True Label")
        axes[idx].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_importance(best_model, best_name, feature_names, save_path):
    """Plot feature importance for the best model."""
    print_section("Generating feature importance plot")

    importances = None

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_).mean(axis=0)
    else:
        print("  Model doesn't support feature importance (skipping)")
        return

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    bars = ax.barh(
        range(len(feature_names)),
        importances[indices[::-1]],
        color=colors
    )
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {best_name}")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curves(results, le, y_test, save_path):
    """Plot ROC curves for each model and each class."""
    print_section("Generating ROC curves")

    n_classes = len(le.classes_)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']

    for class_idx in range(n_classes):
        ax = axes[class_idx]
        class_name = le.classes_[class_idx]

        for model_idx, (name, res) in enumerate(results.items()):
            if res["y_proba"] is not None:
                y_binary = (y_test == class_idx).astype(int)
                proba = res["y_proba"][:, class_idx]
                fpr, tpr, _ = roc_curve(y_binary, proba)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[model_idx % len(colors)],
                        label=f"{name} (AUC={roc_auc:.3f})", linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_title(f"ROC — {class_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(results, save_path):
    """Bar chart comparing models across metrics."""
    print_section("Generating model comparison chart")

    names = list(results.keys())
    metrics = {
        "Accuracy": [results[n]["accuracy"] for n in names],
        "F1 (Weighted)": [results[n]["f1_weighted"] for n in names],
        "F1 (Macro)": [results[n]["f1_macro"] for n in names],
        "CV F1 Mean": [results[n]["cv_mean"] for n in names],
    }

    x = np.arange(len(names))
    width = 0.2
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric_name, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, values, width, label=metric_name, color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ==================== STEP 6: SAVE EVERYTHING ====================

def save_model(best_name, best_result, scaler, le, results):
    """Save the best model and all artifacts."""
    print_header("STEP 5: SAVING MODEL & ARTIFACTS")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(best_result["model"], model_path)
    print(f"  Model saved: {model_path}")

    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved: {scaler_path}")

    # Save label encoder
    le_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
    joblib.dump(le, le_path)
    print(f"  Label encoder saved: {le_path}")

    # Save feature names
    features_path = os.path.join(MODEL_DIR, "feature_names.joblib")
    joblib.dump(ML_FEATURES, features_path)
    print(f"  Feature names saved: {features_path}")

    # Save model metadata
    metadata = {
        "best_model": best_name,
        "accuracy": best_result["accuracy"],
        "f1_weighted": best_result["f1_weighted"],
        "f1_macro": best_result["f1_macro"],
        "cv_f1_mean": best_result["cv_mean"],
        "cv_f1_std": best_result["cv_std"],
        "features": ML_FEATURES,
        "labels": list(le.classes_),
        "trained_at": datetime.now().isoformat(),
        "n_features": len(ML_FEATURES),
        "n_classes": len(le.classes_),
    }
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {meta_path}")

    # Save full training report
    report_path = os.path.join(MODEL_DIR, "training_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  AI-BASED SOLDIER SURVIVAL DETECTION — TRAINING REPORT\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Best Model: {best_name}\n")
        f.write(f"Test Accuracy: {best_result['accuracy']:.4f}\n")
        f.write(f"F1 Weighted: {best_result['f1_weighted']:.4f}\n")
        f.write(f"F1 Macro: {best_result['f1_macro']:.4f}\n")
        f.write(f"CV F1: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}\n\n")

        for name, res in results.items():
            f.write(f"\n{'=' * 50}\n")
            f.write(f"MODEL: {name}\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Accuracy: {res['accuracy']:.4f}\n")
            f.write(f"F1 Weighted: {res['f1_weighted']:.4f}\n")
            f.write(f"F1 Macro: {res['f1_macro']:.4f}\n")
            f.write(f"CV F1: {res['cv_mean']:.4f} ± {res['cv_std']:.4f}\n")
            f.write(f"Train Time: {res['train_time']:.2f}s\n\n")
            f.write("Classification Report:\n")
            f.write(res["report_str"])
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(res["confusion_matrix"]))
            f.write("\n")

    print(f"  Report saved: {report_path}")


# ==================== MAIN ====================

def main():
    print("\n" + "█" * 70)
    print("█  AI-BASED SOLDIER SURVIVAL & EMERGENCY DETECTION SYSTEM")
    print("█  MODEL TRAINING PIPELINE")
    print("█" * 70)

    # Step 1: Load and clean data
    df = load_and_clean_data(DATA_FILE)

    if len(df) < 20:
        print("\nERROR: Not enough data to train a model!")
        print("You need at least a few hundred samples per class.")
        print("Run data_collection.py and follow DATA_COLLECTION_PROTOCOL.md")
        sys.exit(1)

    # Step 2: Prepare features (no scaling yet)
    X, y, le = prepare_features(df)

    # Step 3: Train/test split — prefer session-wise split when possible
    split_idx = split_by_session_stratified(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    if split_idx is not None:
        train_idx, test_idx = split_idx
        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        df_train = df.loc[train_idx]
        print_section("Split strategy")
        print("Using SESSION-WISE split (no session overlap).")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        df_train = None
        print_section("Split strategy")
        print("Using ROW-WISE stratified split (fallback).")
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")

    # Step 4: Get models
    n_classes = len(le.classes_)
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, weights))
    models = get_models(n_classes, class_weights_dict)

    # Step 5a: Leakage-safe cross-validation on TRAIN split only
    cv_results = cross_validate_models(models, X_train, y_train, df_train=df_train)

    # Step 5b: Train on train split, evaluate on held-out test split
    results, scaler = train_and_evaluate(models, X_train, X_test, y_train, y_test, le)

    # Attach CV stats
    for name in results.keys():
        results[name]["cv_mean"] = cv_results[name]["mean"]
        results[name]["cv_std"] = cv_results[name]["std"]

    # Step 6: Compare and select best
    best_name, best_result = compare_and_select(results)

    # Step 7: Generate visualizations
    print_header("STEP 5: GENERATING VISUALIZATIONS")
    os.makedirs(MODEL_DIR, exist_ok=True)

    plot_confusion_matrices(results, le, os.path.join(MODEL_DIR, "confusion_matrices.png"))
    plot_feature_importance(
        best_result["model"], best_name, ML_FEATURES,
        os.path.join(MODEL_DIR, "feature_importance.png")
    )
    plot_roc_curves(results, le, y_test, os.path.join(MODEL_DIR, "roc_curves.png"))
    plot_model_comparison(results, os.path.join(MODEL_DIR, "model_comparison.png"))

    # Step 8: Refit BEST model on ALL DATA for deployment artifacts
    print_header("STEP 6: REFIT BEST MODEL ON FULL DATA")
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X)

    best_model_final = clone(best_result["model"])
    if best_name == "XGBoost" and HAS_XGBOOST:
        classes_all = np.unique(y)
        weights_all = compute_class_weight("balanced", classes=classes_all, y=y)
        wdict_all = dict(zip(classes_all, weights_all))
        sample_weights_all = np.array([wdict_all[yy] for yy in y])
        best_model_final.fit(X_all_scaled, y, sample_weight=sample_weights_all)
    else:
        best_model_final.fit(X_all_scaled, y)

    # Replace best_result model with the fully-refit one for saving
    best_result_to_save = dict(best_result)
    best_result_to_save["model"] = best_model_final

    # Save everything (deployment artifacts use full-data refit)
    save_model(best_name, best_result_to_save, final_scaler, le, results)

    # Final summary
    print_header("TRAINING COMPLETE!")
    print(f"""
  Best Model:     {best_name}
  Test Accuracy:  {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.1f}%)
  F1 Score:       {best_result['f1_weighted']:.4f}
  CV Score:       {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}

  Files saved in '{MODEL_DIR}/' directory:
    - best_model.joblib          → Trained model
    - scaler.joblib              → Feature scaler
    - label_encoder.joblib       → Label encoder
    - feature_names.joblib       → Feature list
    - model_metadata.json        → Model info
    - training_report.txt        → Full report
    - confusion_matrices.png     → Confusion matrices
    - feature_importance.png     → Feature importance
    - roc_curves.png             → ROC curves
    - model_comparison.png       → Model comparison chart

  NEXT STEP:
    Run the real-time dashboard:
      python realtime_dashboard.py

    Then power on your ESP32 device and point it to this laptop's IP.
""")


if __name__ == "__main__":
    main()
