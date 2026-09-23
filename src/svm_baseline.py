"""
svm_baseline.py

Classical ML baseline (SVM / Random Forest) for ECG arrhythmia classification.
Loads the preprocessed beat windows from data/processed/, extracts hand-crafted
features (morphology stats + wavelet coefficients), trains SVM and Random Forest
classifiers, and evaluates both on the held-out patient-independent test set.

Preliminary/unoptimized results for the interim submission — no hyperparameter
tuning yet, just a working end-to-end pipeline.

Usage:
    python src/svm_baseline.py
"""

import os
import numpy as np
import pywt
from scipy.stats import skew, kurtosis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

PROCESSED_DATA_DIR = "data/processed"
LOGS_DIR = "logs"
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


def extract_features(X):
    """Build a hand-crafted feature matrix from raw beat windows.

    Note: RR-interval features are not included here since the current
    preprocessing pipeline stores only the windowed signal, not each beat's
    absolute sample position. This is a reasonable follow-up enhancement
    before the final submission (would need beat sample indices to be saved
    alongside the windows). Morphology + wavelet features are enough for a
    solid preliminary baseline.

    Args:
        X: numpy array of shape (n_beats, window_length).

    Returns:
        numpy array of shape (n_beats, n_features).
    """
    feature_rows = []
    for window in X:
        morph = extract_morphology_features(window)
        wave = extract_wavelet_features(window)
        feature_rows.append(np.concatenate([morph, wave]))
    return np.array(feature_rows)


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------

def evaluate_model(name, model, X_test, y_test, logs_dir=LOGS_DIR):
    """Run predictions, print + save a classification report and confusion matrix."""
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

    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{name.lower().replace(' ', '_')}_results.txt")
    with open(log_path, "w") as f:
        f.write(f"{name} results\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(str(cm))

    print(f"Saved results to {log_path}")
    return acc


def main():
    print("Loading processed data...")
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"))

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("\nExtracting hand-crafted features (morphology + wavelet)...")
    X_train_feat = extract_features(X_train)
    X_test_feat = extract_features(X_test)
    print(f"Feature matrix shape: {X_train_feat.shape}")

    # Scale features (important for SVM, harmless for RF)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    # --- Random Forest ---
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    evaluate_model("Random Forest", rf, X_test_scaled, y_test)

    # --- SVM ---
    print("\nTraining SVM...")
    svm = SVC(kernel="rbf", class_weight="balanced", random_state=42)
    svm.fit(X_train_scaled, y_train)
    evaluate_model("SVM", svm, X_test_scaled, y_test)

    print("\nDone. Preliminary baseline results saved to logs/.")


if __name__ == "__main__":
    main()