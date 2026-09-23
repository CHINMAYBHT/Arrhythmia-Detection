"""
svm_baseline.py

Classical ML baseline (SVM / Random Forest) for ECG arrhythmia classification.
Loads the preprocessed beat windows from data/processed/, extracts hand-crafted
features (morphology stats + wavelet coefficients + RR-interval features),
trains SVM and Random Forest classifiers (with light hyperparameter tuning and
SMOTE oversampling of the training set to address severe class imbalance), and
evaluates both on the held-out patient-independent test set.

Usage:
    python src/svm_baseline.py

Requires `imbalanced-learn` (see requirements.txt) for the SMOTE step.
"""

import os
from collections import Counter
import numpy as np
import pywt
from scipy.stats import skew, kurtosis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

PROCESSED_DATA_DIR = "data/processed"
RESULTS_DIR = "results"
CLASS_NAMES = ["N", "S", "V", "F", "Q"]  # must match CLASS_TO_IDX order in preprocessing.py


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_morphology_features(window):
    """Basic shape statistics of one beat window.

    Args:
        window: 1D numpy array, one beat segment.

    Returns:
        1D numpy array of scalar features.
    """
    return np.array([
        np.max(window),
        np.min(window),
        np.mean(window),
        np.std(window),
        skew(window),
        kurtosis(window),
        np.max(window) - np.min(window),   # peak-to-peak amplitude
        np.sum(np.abs(np.diff(window))),   # rough signal energy/roughness
    ])


def extract_wavelet_features(window, wavelet="db4", level=3):
    """Wavelet decomposition coefficients, summarized to a fixed-length vector.

    Args:
        window: 1D numpy array, one beat segment.
        wavelet: wavelet family to use.
        level: decomposition level.

    Returns:
        1D numpy array of wavelet-derived scalar features (mean + std of each
        coefficient sub-band).
    """
    coeffs = pywt.wavedec(window, wavelet, level=level)
    features = []
    for c in coeffs:
        features.append(np.mean(c))
        features.append(np.std(c))
    return np.array(features)


def extract_features(X, rr_features=None):
    """Build a hand-crafted feature matrix from raw beat windows.

    Combines morphology stats + wavelet coefficients with RR-interval
    features (previous RR, next RR, local average RR, RR ratio) computed
    per-record in preprocessing.py (see `compute_rr_features()` there) and
    saved as rr_features_train.npy / rr_features_test.npy. RR intervals
    capture rhythm-level information (e.g. how early/late a beat arrives
    relative to its neighbours) that pure beat-shape features cannot, and
    are one of the standard feature groups used for AAMI-class ECG beat
    classification.

    Args:
        X: numpy array of shape (n_beats, window_length).
        rr_features: optional numpy array of shape (n_beats, 4) with
            [prev_rr, next_rr, local_avg_rr, rr_ratio] per beat, in the same
            order as X. If None, only morphology + wavelet features are used
            (kept for backward compatibility / quick experiments).

    Returns:
        numpy array of shape (n_beats, n_features).
    """
    feature_rows = []
    for window in X:
        morph = extract_morphology_features(window)
        wave = extract_wavelet_features(window)
        feature_rows.append(np.concatenate([morph, wave]))
    features = np.array(feature_rows)

    if rr_features is not None:
        features = np.concatenate([features, rr_features], axis=1)

    return features


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------

def evaluate_model(name, model, X_test, y_test, results_dir=RESULTS_DIR):
    """Run predictions, print + save a classification report and confusion
    matrix to results_dir, matching cnn_model.py / cnn_lstm_model.py's
    evaluate_model() so all models' results land in the same results/
    directory for a direct cross-model comparison.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(report)
    print("Confusion matrix:")
    print(cm)

    os.makedirs(results_dir, exist_ok=True)
    safe_name = name.lower().replace(' ', '_').replace('-', '_')
    log_path = os.path.join(results_dir, f"{safe_name}_results.txt")
    with open(log_path, "w") as f:
        f.write(f"{name} results\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(str(cm))

    print(f"Saved results to {log_path}")
    return acc


def load_rr_features(split, n_beats):
    """Load RR-interval features for a split ('train'/'test') if present.

    Falls back to None (and prints a warning) if the file doesn't exist yet,
    e.g. because data/processed/ was produced by an older version of
    preprocessing.py -- in that case extract_features() just skips RR
    features so the pipeline still runs end-to-end.
    """
    path = os.path.join(PROCESSED_DATA_DIR, f"rr_features_{split}.npy")
    if not os.path.exists(path):
        print(
            f"  [warn] {path} not found -- re-run src/preprocessing.py to "
            "generate RR-interval features. Continuing without them."
        )
        return None
    rr = np.load(path)
    assert rr.shape[0] == n_beats, (
        f"rr_features_{split}.npy has {rr.shape[0]} rows but X_{split} has {n_beats} beats "
        "-- data/processed/ is likely stale, re-run src/preprocessing.py."
    )
    return rr


def main():
    print("Loading processed data...")
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"))

    rr_train = load_rr_features("train", len(y_train))
    rr_test = load_rr_features("test", len(y_test))

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("\nExtracting hand-crafted features (morphology + wavelet + RR-interval)...")
    X_train_feat = extract_features(X_train, rr_train)
    X_test_feat = extract_features(X_test, rr_test)
    print(f"Feature matrix shape: {X_train_feat.shape}")

    # Scale features (important for SVM, harmless for RF)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    # --- Address severe class imbalance ---
    # class_weight="balanced" alone left S/F/Q with near-zero recall (see
    # results/random_forest_results.txt, results/svm_results.txt from the
    # interim submission): reweighting the loss doesn't give the model more
    # *examples* of the minority classes, which matters a lot for the RBF-SVM
    # decision boundary and for RF splits that rarely see rare classes at all.
    # We apply SMOTE (synthetic minority oversampling) to the TRAINING set
    # only -- never to X_test_scaled/y_test, which must stay a true reflection
    # of the real-world class distribution for evaluation. We still keep
    # class_weight="balanced" on both models as a cheap complementary measure.
    print("\nApplying SMOTE oversampling to the training set only...")
    print("Class counts before SMOTE:", {c: int(np.sum(y_train == i)) for c, i in zip(CLASS_NAMES, range(5))})

    # SMOTE's default sampling_strategy="auto" oversamples every minority
    # class up to the MAJORITY class's count. Here the majority (N) class has
    # tens of thousands of beats, so fully balancing would make the resampled
    # training set roughly 5x the majority count -- impractical to train an
    # RBF-SVM on. Instead we cap how many synthetic samples each minority
    # class is boosted to (still a large increase over the original few
    # hundred/thousand samples for S/F/Q), and leave the majority class alone.
    class_counts = Counter(y_train)
    majority_count = max(class_counts.values())
    target_per_class = min(majority_count, 20000)
    sampling_strategy = {
        cls: max(count, target_per_class)
        for cls, count in class_counts.items()
        if count < majority_count
    }
    smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=sampling_strategy)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print("Class counts after SMOTE: ", {c: int(np.sum(y_train_res == i)) for c, i in zip(CLASS_NAMES, range(5))})

    # --- Random Forest (small grid search) ---
    print("\nTuning Random Forest hyperparameters (GridSearchCV)...")
    rf_param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2],
    }
    # n_jobs=-1 on the *outer* GridSearchCV parallelizes across param/fold
    # combinations; the inner RandomForestClassifier is kept single-threaded
    # (n_jobs=1) to avoid nested parallelism oversubscribing all CPU cores at
    # once, which can badly slow down (or, on some platforms, deadlock) a
    # grid search.
    rf_search = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1),
        rf_param_grid,
        scoring="f1_macro",  # macro-F1 rewards doing well on minority classes, not just accuracy
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1,
    )
    rf_search.fit(X_train_res, y_train_res)
    print(f"Best RF params: {rf_search.best_params_} (cv macro-F1={rf_search.best_score_:.4f})")

    # Refit the final model with the tuned params and full parallelism
    # (no nested GridSearchCV wrapping it anymore, so n_jobs=-1 is safe here).
    rf = RandomForestClassifier(
        class_weight="balanced", random_state=42, n_jobs=-1, **rf_search.best_params_
    )
    rf.fit(X_train_res, y_train_res)
    evaluate_model("Random Forest", rf, X_test_scaled, y_test)

    # --- SVM (small grid search) ---
    # SVC training scales poorly with n_samples, so tune on a stratified
    # subsample of the SMOTE-resampled training set to keep grid search
    # tractable, then fit the chosen params on the full resampled set.
    print("\nTuning SVM hyperparameters (GridSearchCV on a subsample)...")
    rng = np.random.RandomState(42)
    max_tune_samples = 20000
    if len(y_train_res) > max_tune_samples:
        idx = rng.choice(len(y_train_res), size=max_tune_samples, replace=False)
        X_tune, y_tune = X_train_res[idx], y_train_res[idx]
    else:
        X_tune, y_tune = X_train_res, y_train_res

    svm_param_grid = {
        "C": [1, 10],
        "gamma": ["scale", 0.01],
    }
    svm_search = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", random_state=42),
        svm_param_grid,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1,
    )
    svm_search.fit(X_tune, y_tune)
    print(f"Best SVM params: {svm_search.best_params_} (cv macro-F1={svm_search.best_score_:.4f})")

    # RBF-SVM training cost grows super-linearly with n_samples, so even the
    # capped SMOTE output (majority class alone can be tens of thousands of
    # beats) is impractical to fit directly on a laptop. Fit the final SVM on
    # a larger stratified subsample instead of the full resampled set --
    # Random Forest above doesn't need this since tree ensembles scale much
    # better with n_samples.
    max_final_svm_samples = 60000
    if len(y_train_res) > max_final_svm_samples:
        idx = rng.choice(len(y_train_res), size=max_final_svm_samples, replace=False)
        X_svm_fit, y_svm_fit = X_train_res[idx], y_train_res[idx]
    else:
        X_svm_fit, y_svm_fit = X_train_res, y_train_res

    print(f"\nTraining final SVM on {len(y_svm_fit)} (SMOTE-resampled, subsampled) beats...")
    svm = SVC(kernel="rbf", class_weight="balanced", random_state=42, **svm_search.best_params_)
    svm.fit(X_svm_fit, y_svm_fit)
    evaluate_model("SVM", svm, X_test_scaled, y_test)

    print("\nDone. Results saved to results/.")


if __name__ == "__main__":
    main()