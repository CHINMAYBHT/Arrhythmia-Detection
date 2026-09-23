"""
cnn_model.py

Model 2: 1D-CNN trained directly on raw ECG beat windows (no hand-crafted
features), for AAMI 5-class arrhythmia classification (N/S/V/F/Q).

Loads the preprocessed beat windows from data/processed/ (see
src/preprocessing.py for the exact format: X_train/X_test shape
(n_beats, window_length), y as integer class indices 0-4 matching
CLASS_TO_IDX = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}), builds a 1D-CNN
with `build_cnn_model()`, trains it with class weighting to counter the
severe class imbalance (S/F/Q are rare - see results/random_forest_results.txt
and results/svm_results.txt), evaluates on the held-out patient-independent
test set, and saves the trained model + a results log.

--------------------------------------------------------------------------
RUN THIS ON KAGGLE (or any machine with a real TensorFlow+GPU install) -
NOT in this local dev environment.
--------------------------------------------------------------------------
This repo's local environment is Python 3.14 with no stable TensorFlow
wheel available (only a 2.22.0rc0 release candidate) and no GPU, so this
script cannot be executed here. It is written to be uploaded as a Kaggle
notebook/script, where TensorFlow/Keras + GPU come preinstalled.

How to point it at Kaggle's data when you run it there:
    1. Upload data/processed/*.npy (X_train, y_train, X_test, y_test) as a
       Kaggle Dataset, then in the notebook: "Add Data" -> attach it. Kaggle
       mounts it read-only under /kaggle/input/<dataset-name>/.
    2. Either set the environment variable PROCESSED_DATA_DIR before running,
       e.g. (in a notebook cell):
           import os
           os.environ["PROCESSED_DATA_DIR"] = "/kaggle/input/<dataset-name>"
       or just edit the PROCESSED_DATA_DIR constant below directly.
    3. Trained model + logs are written under /kaggle/working/ by default
       when run on Kaggle (MODELS_DIR / LOGS_DIR / RESULTS_DIR also respect
       the PROJECT_ROOT override - see Config section below).

Usage (on Kaggle or a local machine with TensorFlow installed):
    python src/cnn_model.py
"""

import os
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# PROJECT_ROOT-relative paths, following the same pattern as preprocessing.py.
# Override via environment variables to point at Kaggle's mounted dataset
# path (e.g. /kaggle/input/<dataset-name>) without touching any logic below.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = Path(os.environ.get(
    "PROCESSED_DATA_DIR", PROJECT_ROOT / "data" / "processed"
))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", PROJECT_ROOT / "models"))
LOGS_DIR = Path(os.environ.get("LOGS_DIR", PROJECT_ROOT / "logs"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", PROJECT_ROOT / "results"))

CLASS_NAMES = ["N", "S", "V", "F", "Q"]  # must match CLASS_TO_IDX order in preprocessing.py
N_CLASSES = len(CLASS_NAMES)

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_cnn_model(
    input_shape,
    n_classes=N_CLASSES,
    num_filters=(32, 64, 128),
    kernel_size=7,
    pool_size=2,
    dropout_rate=0.3,
    dense_units=64,
    learning_rate=1e-3,
):
    """Build and compile a 1D-CNN for raw ECG beat-window classification.

    This is the shared entry point for both the plain Adam-trained baseline
    run in this file and any later metaheuristic (GWO/WOA) hyperparameter
    search - call it directly with candidate hyperparameter values, it
    returns a freshly built and compiled `keras.Model` each time.

    Architecture: a stack of (Conv1D -> BatchNorm -> ReLU -> MaxPooling1D)
    blocks (one block per entry in `num_filters`), followed by Dropout,
    GlobalAveragePooling1D, a Dense head, and a softmax output layer.

    Args:
        input_shape: tuple, shape of one input sample as (window_length, 1).
            (window_length is WINDOW_BEFORE + WINDOW_AFTER from
            preprocessing.py, e.g. 200; the trailing 1 is the channel dim
            Conv1D expects.)
        n_classes: int, number of output classes (5 for N/S/V/F/Q).
        num_filters: tuple/list of ints, number of filters for each
            successive Conv1D block, e.g. (32, 64, 128) for a 3-block model.
        kernel_size: int, Conv1D kernel size (same for every block).
        pool_size: int, MaxPooling1D pool size (same for every block).
        dropout_rate: float in [0, 1), dropout applied after the conv stack
            and before the dense head.
        dense_units: int, number of units in the dense head layer before
            the final softmax output.
        learning_rate: float, Adam optimizer learning rate.

    Returns:
        A compiled `tensorflow.keras.Model` using
        `sparse_categorical_crossentropy` loss (expects integer-encoded
        labels 0-4 directly, no one-hot conversion needed) and an accuracy
        metric, ready for `model.fit(...)`.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for filters in num_filters:
        x = layers.Conv1D(filters, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling1D(pool_size=pool_size)(x)

    x = layers.Dropout(dropout_rate)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Data loading / prep
# ---------------------------------------------------------------------------

def load_data(processed_data_dir=PROCESSED_DATA_DIR):
    """Load the preprocessed train/test arrays produced by preprocessing.py.

    Args:
        processed_data_dir: directory containing X_train.npy, y_train.npy,
            X_test.npy, y_test.npy.

    Returns:
        X_train: (n_train, window_length) float array
        y_train: (n_train,) int array, class indices 0-4
        X_test: (n_test, window_length) float array
        y_test: (n_test,) int array, class indices 0-4
    """
    X_train = np.load(os.path.join(processed_data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(processed_data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_data_dir, "y_test.npy"))
    return X_train, y_train, X_test, y_test


def prepare_inputs(X):
    """Reshape and z-score normalize raw windows for the Conv1D input layer.

    Args:
        X: numpy array of shape (n_beats, window_length).

    Returns:
        numpy array of shape (n_beats, window_length, 1), per-beat
        z-score normalized (each beat window scaled to zero mean / unit
        std) since Conv1D expects a trailing channel dimension.
    """
    X = X.astype("float32")
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    X = (X - mean) / std
    return X[..., np.newaxis]


def compute_class_weights(y_train):
    """Compute balanced class weights from the training label distribution.

    Args:
        y_train: (n_train,) int array of class indices.

    Returns:
        dict mapping class index -> weight, suitable for
        `model.fit(..., class_weight=...)`.
    """
    classes = np.arange(N_CLASSES)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------

def evaluate_model(name, model, X_test, y_test, results_dir=RESULTS_DIR):
    """Run predictions, print + save a classification report and confusion
    matrix, mirroring the SVM/RF baseline's evaluate_model() pattern in
    src/svm_baseline.py for a consistent cross-model comparison.
    """
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

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


def train_and_evaluate(
    X_train, y_train, X_test, y_test,
    hyperparams=None,
    epochs=50,
    batch_size=128,
    logs_dir=LOGS_DIR,
    models_dir=MODELS_DIR,
    model_name="cnn_baseline",
):
    """Full train + evaluate pipeline for the 1D-CNN baseline.

    Builds the model via `build_cnn_model()`, trains with class weighting
    and EarlyStopping/ReduceLROnPlateau callbacks, logs per-epoch metrics
    to logs_dir, evaluates on the test set, and saves the trained model.

    Args:
        X_train, y_train, X_test, y_test: preprocessed arrays as returned by
            `load_data()` (raw, not yet reshaped/normalized - this function
            calls `prepare_inputs()` internally).
        hyperparams: dict of keyword args forwarded to `build_cnn_model()`
            (num_filters, kernel_size, dropout_rate, learning_rate, etc.).
            Defaults to `build_cnn_model()`'s own defaults if None - this is
            the "before metaheuristic tuning" baseline configuration.
        epochs: max training epochs (EarlyStopping will typically stop sooner).
        batch_size: training batch size.
        logs_dir: directory for the per-epoch CSV training log.
        models_dir: directory to save the trained model file.
        model_name: base filename (no extension) for the saved model/log.

    Returns:
        (model, history): the trained keras.Model and its History object.
    """
    hyperparams = hyperparams or {}

    X_train_prep = prepare_inputs(X_train)
    X_test_prep = prepare_inputs(X_test)
    input_shape = X_train_prep.shape[1:]  # (window_length, 1)

    print(f"Input shape per sample: {input_shape}")
    model = build_cnn_model(input_shape, n_classes=N_CLASSES, **hyperparams)
    model.summary()

    class_weights = compute_class_weights(y_train)
    print(f"Class weights (balanced): {class_weights}")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    csv_log_path = os.path.join(logs_dir, f"{model_name}_training_log.csv")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        keras.callbacks.CSVLogger(csv_log_path),
    ]

    print(f"\nTraining CNN (per-epoch metrics logged to {csv_log_path})...")
    history = model.fit(
        X_train_prep, y_train,
        validation_data=(X_test_prep, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    model_path = os.path.join(models_dir, f"{model_name}.keras")
    model.save(model_path)
    print(f"Saved trained model to {model_path}")

    # Use model_name (not a fixed "CNN") so the baseline run and each
    # metaheuristic-tuned run (see metaheuristic_tuning.py, which calls this
    # with model_name="gwo_tuned_cnn" / "woa_tuned_cnn") write to distinct
    # results files instead of all overwriting results/cnn_results.txt.
    evaluate_model(model_name, model, X_test_prep, y_test)

    return model, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    keras.utils.set_random_seed(RANDOM_SEED)

    print(f"Loading processed data from {PROCESSED_DATA_DIR} ...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Baseline hyperparameters (before metaheuristic/GWO-WOA tuning). A
    # teammate's optimizer script should call build_cnn_model() directly
    # with its own candidate hyperparameter dicts instead of editing this.
    baseline_hyperparams = dict(
        num_filters=(32, 64, 128),
        kernel_size=7,
        pool_size=2,
        dropout_rate=0.3,
        dense_units=64,
        learning_rate=1e-3,
    )

    train_and_evaluate(
        X_train, y_train, X_test, y_test,
        hyperparams=baseline_hyperparams,
        epochs=50,
        batch_size=128,
        model_name="cnn_baseline",
    )

    print("\nDone. Preliminary CNN baseline results saved to results/ and logs/.")


if __name__ == "__main__":
    main()
