"""
cnn_lstm_model.py

Model 3: CNN-LSTM hybrid for ECG arrhythmia classification.

Unlike the plain 1D-CNN (Model 2), which classifies each beat window in
isolation, this model captures BEAT-TO-BEAT TEMPORAL dependencies: consecutive
beats from the same patient record are grouped into short sequences, a
Conv1D+pooling block extracts per-beat morphology features (applied to every
beat in the sequence via TimeDistributed), and an LSTM then models how those
per-beat feature vectors evolve across the sequence. A dense head classifies
the beat at the END of each sequence (the "current" beat, given its recent
history) into one of the 5 AAMI classes.

Why classify the last beat rather than the whole sequence at once:
preprocessing.py's labels are per-beat, and the classical approach in
arrhythmia literature (RR-interval-aware classifiers) is to predict the
current beat conditioned on a short window of preceding beats. Predicting
every beat in the sequence in parallel would work too, but would require a
many-to-many output head and repeats every beat's label ~seq_len times in the
loss, which is more complex for a mini-project with modest benefit. Using the
last beat also mirrors how this model would be used online: at each new beat,
look back at the last few beats and classify the newest one.

Sequence construction (see build_beat_sequences() / build_sequence_dataset()):
preprocessing.build_dataset() concatenates beats from all records into one
flat (n_beats, window_length) array, which loses each record's beat order and
would let a sliding window span two different patients. This file re-derives
per-record beat order by calling preprocessing.load_and_segment_record() once
per record (same function the shared pipeline already uses), and slides a
window of `seq_len` consecutive beats (default 5) over each record
independently, stepping by 1 beat, using the LAST beat's label as the
sequence's target. Sequences are only ever built from beats within a single
record, so no sequence mixes two patients.

Resulting shapes:
    X_seq: (n_sequences, seq_len, window_length)   float32
    y_seq: (n_sequences,)                          int labels 0-4 (AAMI order N,S,V,F,Q)

Architecture (build_cnn_lstm_model):
    Input (seq_len, window_length, 1)
      -> TimeDistributed(Conv1D -> BatchNorm -> ReLU -> MaxPool1D)  [x n_conv_blocks]
      -> TimeDistributed(GlobalAveragePooling1D)     # per-beat feature vector
      -> LSTM(lstm_units)                             # models across the beat sequence
      -> Dropout
      -> Dense(dense_units, relu)
      -> Dense(n_classes, softmax)                    # classifies the LAST beat

IMPORTANT — WHERE TO RUN THIS:
This script requires `tensorflow.keras` with a working TensorFlow install and
is intended to run on Kaggle (or any machine with GPU + a stable TensorFlow),
NOT in this project's local dev environment. The local machine only has
Python 3.14, for which no stable TensorFlow release exists yet (only a
2.22.0rc0 release candidate), and there is no local GPU. Do not try to `pip
install tensorflow` and run this locally — write/edit the code here, then
upload/run it as a Kaggle notebook or script.

Adapting the data path for Kaggle:
- Locally, data is read from PROJECT_ROOT/data/raw (raw MIT-BIH .dat/.hea/.atr
  files) per preprocessing.py's layout.
- On Kaggle, add the MIT-BIH dataset via "Add Input" in the notebook's data
  panel; it will be mounted read-only under /kaggle/input/<dataset-slug>/...
- Point this script at the right location by setting the RAW_DATA_DIR
  environment variable before running, e.g. in a Kaggle notebook cell:
      import os
      os.environ["RAW_DATA_DIR"] = "/kaggle/input/mitbih-arrhythmia/raw"
  or just edit the RAW_DATA_DIR constant below directly. Trained models and
  logs are written under MODELS_DIR / LOGS_DIR, which default to
  /kaggle/working/{models,logs} style relative paths when run from a Kaggle
  working directory, or PROJECT_ROOT-relative paths locally -- override with
  the MODELS_DIR / LOGS_DIR env vars if needed.

Usage (on Kaggle / a GPU+TensorFlow machine):
    python src/cnn_lstm_model.py
"""

import os
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# preprocessing.py holds the shared record lists, AAMI label mapping, and the
# per-record loading function we reuse to rebuild per-record beat order.
from preprocessing import (
    TRAIN_RECORDS,
    TEST_RECORDS,
    CLASS_TO_IDX,
    load_and_segment_record,
)

# ---------------------------------------------------------------------------
# Config -- override any of these via environment variables so the same
# script works unchanged locally (paths relative to the project root) and on
# Kaggle (paths under /kaggle/input or /kaggle/working).
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw MIT-BIH files. On Kaggle, set RAW_DATA_DIR to wherever the dataset was
# added under /kaggle/input/, e.g. "/kaggle/input/mitbih-arrhythmia/raw".
RAW_DATA_DIR = Path(os.environ.get("RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw"))

MODELS_DIR = Path(os.environ.get("MODELS_DIR", PROJECT_ROOT / "models"))
LOGS_DIR = Path(os.environ.get("LOGS_DIR", PROJECT_ROOT / "logs"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", PROJECT_ROOT / "results"))

CLASS_NAMES = ["N", "S", "V", "F", "Q"]  # must match CLASS_TO_IDX order in preprocessing.py

# Beats per sequence and the sliding-window step (in beats) used to build
# sequences. seq_len=5 with step=1 means every beat (once enough history has
# accumulated) becomes the target of one sequence, looking back 4 beats.
DEFAULT_SEQ_LEN = 5
DEFAULT_STEP = 1


# ---------------------------------------------------------------------------
# Sequence construction (respects per-record boundaries)
# ---------------------------------------------------------------------------

def build_beat_sequences(windows, labels, seq_len=DEFAULT_SEQ_LEN, step=DEFAULT_STEP):
    """Slide a window of `seq_len` consecutive beats over ONE record's beats.

    Args:
        windows: numpy array of shape (n_beats, window_length), beats from a
            single record, in the original chronological order they were
            annotated in.
        labels: numpy array of shape (n_beats,), integer class indices
            aligned with `windows`.
        seq_len: number of consecutive beats per sequence.
        step: how many beats to advance the window start by each step.

    Returns:
        X_seq: numpy array of shape (n_sequences, seq_len, window_length)
        y_seq: numpy array of shape (n_sequences,) -- label of the LAST beat
            in each sequence (the beat being classified).
    """
    n_beats = len(windows)
    X_seq, y_seq = [], []
    for start in range(0, n_beats - seq_len + 1, step):
        end = start + seq_len
        X_seq.append(windows[start:end])
        y_seq.append(labels[end - 1])  # label of the last (most recent) beat
    if not X_seq:
        window_length = windows.shape[1] if windows.ndim == 2 else 0
        return (
            np.empty((0, seq_len, window_length), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)


def build_sequence_dataset(record_ids, data_dir=RAW_DATA_DIR, seq_len=DEFAULT_SEQ_LEN,
                            step=DEFAULT_STEP):
    """Build a full beat-sequence dataset (X, y) from a list of record IDs.

    Mirrors preprocessing.build_dataset(), but instead of concatenating all
    beats into one flat array (which would lose each record's beat order and
    let a sequence span two different patients), this loads each record
    separately, builds sliding-window beat sequences WITHIN that record only,
    and then concatenates the resulting sequences across records.

    Args:
        record_ids: list of MIT-BIH record ID strings (e.g. TRAIN_RECORDS).
        data_dir: directory containing the raw .dat/.hea/.atr files.
        seq_len: beats per sequence (see build_beat_sequences).
        step: sliding-window step in beats (see build_beat_sequences).

    Returns:
        X: numpy array of shape (n_sequences, seq_len, window_length)
        y: numpy array of shape (n_sequences,) with integer class indices (0-4)
    """
    all_X, all_y = [], []

    for record_id in record_ids:
        try:
            windows, label_strs, _sample_indices = load_and_segment_record(record_id, data_dir)
        except FileNotFoundError:
            print(f"  [skip] record {record_id} not found in {data_dir}")
            continue

        if len(label_strs) < seq_len:
            print(f"  [skip] record {record_id}: only {len(label_strs)} beats (< seq_len={seq_len})")
            continue

        label_idx = np.array([CLASS_TO_IDX[lbl] for lbl in label_strs])
        X_seq, y_seq = build_beat_sequences(windows, label_idx, seq_len=seq_len, step=step)

        all_X.append(X_seq)
        all_y.append(y_seq)
        print(f"  record {record_id}: {len(label_strs)} beats -> {len(y_seq)} sequences")

    if not all_X:
        raise FileNotFoundError(
            f"No records were found in {Path(data_dir).resolve()}. "
            "Check that the MIT-BIH files are downloaded there, or set the "
            "RAW_DATA_DIR environment variable (see this file's module docstring)."
        )

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_cnn_lstm_model(
    input_shape,
    n_classes=5,
    conv_filters=(32, 64),
    conv_kernel_size=5,
    pool_size=2,
    lstm_units=64,
    dense_units=64,
    dropout_rate=0.3,
    learning_rate=1e-3,
):
    """Build the CNN-LSTM hybrid model.

    Args:
        input_shape: tuple (seq_len, window_length, n_channels), the shape of
            one input sequence (n_channels is 1 for the raw single-lead ECG
            windows produced by build_sequence_dataset -- add a trailing
            channel axis before feeding data in, see main()).
        n_classes: number of output classes (5 for N/S/V/F/Q).
        conv_filters: tuple of filter counts, one Conv1D+pool block per entry.
        conv_kernel_size: kernel size for every Conv1D block.
        pool_size: pool size for every MaxPool1D block.
        lstm_units: number of units in the LSTM layer that runs across beats.
        dense_units: number of units in the dense layer before the output.
        dropout_rate: dropout applied after the LSTM and after the dense layer.
        learning_rate: Adam learning rate.

    Returns:
        A compiled tensorflow.keras.Model. Input shape (batch, seq_len,
        window_length, n_channels); output shape (batch, n_classes) softmax
        probabilities for the LAST beat in each input sequence.
    """
    from tensorflow.keras import layers, models, optimizers

    inputs = layers.Input(shape=input_shape, name="beat_sequence")
    x = inputs

    # Per-beat convolutional feature extraction, applied identically to every
    # beat in the sequence (shared weights across the time axis).
    for n_filters in conv_filters:
        x = layers.TimeDistributed(
            layers.Conv1D(n_filters, conv_kernel_size, padding="same")
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.Activation("relu"))(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(pool_size=pool_size))(x)

    # Collapse each beat's conv feature map to one feature vector per beat.
    x = layers.TimeDistributed(layers.GlobalAveragePooling1D())(x)

    # Model temporal dependencies ACROSS the beat sequence.
    x = layers.LSTM(lstm_units)(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="beat_class")(x)

    model = models.Model(inputs, outputs, name="cnn_lstm_hybrid")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------

def evaluate_model(name, model, X_test, y_test, results_dir=RESULTS_DIR):
    """Run predictions, print + save a classification report and confusion
    matrix to results_dir (RESULTS_DIR), matching cnn_model.py's
    evaluate_model() so all models' final results land in the same results/
    directory for a direct cross-model comparison. Per-epoch training
    metrics (loss/accuracy per epoch) are logged separately to LOGS_DIR via
    CSVLogger in main() -- this function only writes the final test-set
    evaluation.
    """
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)

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


def main():
    # Imported here (not at module level) so this file can still be read /
    # imported for its data-prep helpers (build_sequence_dataset etc.) in an
    # environment without TensorFlow installed -- only main() actually needs
    # tensorflow.keras.
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger

    seq_len = int(os.environ.get("SEQ_LEN", DEFAULT_SEQ_LEN))
    step = int(os.environ.get("SEQ_STEP", DEFAULT_STEP))

    print(f"Building CNN-LSTM sequence dataset (seq_len={seq_len}, step={step})...")
    print("Training set (patient-independent):")
    X_train, y_train = build_sequence_dataset(TRAIN_RECORDS, RAW_DATA_DIR, seq_len, step)
    print("\nTest set (patient-independent):")
    X_test, y_test = build_sequence_dataset(TEST_RECORDS, RAW_DATA_DIR, seq_len, step)

    print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape},  y_test shape:  {y_test.shape}")

    # Add a trailing channel axis: Conv1D expects (steps, channels) per beat,
    # so each sequence element goes from (window_length,) to (window_length, 1).
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Class distribution (S/F/Q are rare -- see results/random_forest_results.txt
    # and results/svm_results.txt).
    print("\nTrain class distribution:")
    for cls, idx in CLASS_TO_IDX.items():
        count = np.sum(y_train == idx)
        print(f"  {cls}: {count} ({100 * count / len(y_train):.1f}%)")

    # Handle class imbalance the same way as the baseline models: balanced
    # class weights passed into model.fit() rather than resampling the data.
    class_weight_arr = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(CLASS_TO_IDX)), y=y_train
    )
    class_weight = {i: w for i, w in enumerate(class_weight_arr)}
    print(f"\nClass weights: {class_weight}")

    input_shape = X_train.shape[1:]  # (seq_len, window_length, 1)
    model = build_cnn_lstm_model(input_shape, n_classes=len(CLASS_TO_IDX))
    model.summary()

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        # Per-epoch metrics logged to logs/, per the project convention of
        # logging metrics per epoch for the interim/final report.
        CSVLogger(os.path.join(LOGS_DIR, "cnn_lstm_training_log.csv")),
    ]

    print("\nTraining CNN-LSTM hybrid...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=int(os.environ.get("EPOCHS", 50)),
        batch_size=int(os.environ.get("BATCH_SIZE", 64)),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    evaluate_model("CNN-LSTM", model, X_test, y_test)

    model_path = os.path.join(MODELS_DIR, "cnn_lstm_model.keras")
    model.save(model_path)
    print(f"\nSaved trained model to {model_path}")

    print("\nDone. CNN-LSTM hybrid results saved to results/.")


if __name__ == "__main__":
    main()
