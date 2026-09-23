"""
preprocessing.py

Shared preprocessing pipeline for the ECG arrhythmia classification project.
Loads MIT-BIH records, filters the signal, segments fixed-length windows around
each annotated beat, maps labels to AAMI EC57 classes, and produces a
patient-independent train/test split.

All three models (SVM/RF baseline, 1D-CNN, CNN-LSTM) should load their data
via load_dataset() so every model is trained/evaluated on identical data.

Usage:
    python src/preprocessing.py
"""

import os
from pathlib import Path
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

FS = 360  # MIT-BIH sampling frequency (Hz)
WINDOW_BEFORE = 90  # samples before R-peak (~0.25s at 360Hz)
WINDOW_AFTER = 110   # samples after R-peak (~0.30s at 360Hz)
# Total window length = WINDOW_BEFORE + WINDOW_AFTER

# Patient-independent split (DS1 / DS2 style split commonly used in MIT-BIH literature)
TRAIN_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
]
TEST_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
]

# AAMI EC57 beat label mapping (MIT-BIH annotation symbol -> AAMI class)
AAMI_MAP = {
    # Normal
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # Supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # Ventricular ectopic
    "V": "V", "E": "V",
    # Fusion
    "F": "F",
    # Unknown / paced
    "/": "Q", "f": "Q", "Q": "Q",
}
CLASS_TO_IDX = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def bandpass_filter(signal, fs=FS, lowcut=0.5, highcut=40.0, order=3):
    """Apply a Butterworth bandpass filter to remove baseline wander and
    high-frequency noise.

    Args:
        signal: 1D numpy array, raw ECG signal.
        fs: sampling frequency in Hz.
        lowcut, highcut: filter cutoff frequencies in Hz.
        order: filter order.

    Returns:
        1D numpy array, filtered signal (same length as input).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def load_and_segment_record(record_id, data_dir=RAW_DATA_DIR):
    """Load one MIT-BIH record, filter it, and extract fixed-length windows
    around every annotated beat.

    Args:
        record_id: string record name, e.g. '100'.
        data_dir: directory containing the raw .dat/.hea/.atr files.

    Returns:
        windows: numpy array of shape (n_beats, WINDOW_BEFORE + WINDOW_AFTER)
        labels: numpy array of shape (n_beats,) with AAMI class strings
    """
    record_path = os.path.join(data_dir, record_id)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    # Use channel 0 (MLII lead in most MIT-BIH records)
    signal = record.p_signal[:, 0]
    filtered = bandpass_filter(signal)

    windows = []
    labels = []

    for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in AAMI_MAP:
            continue  # skip non-beat annotations (rhythm labels, etc.)

        start = sample_idx - WINDOW_BEFORE
        end = sample_idx + WINDOW_AFTER

        # Skip beats too close to the start/end of the recording
        if start < 0 or end > len(filtered):
            continue

        windows.append(filtered[start:end])
        labels.append(AAMI_MAP[symbol])

    return np.array(windows), np.array(labels)


def build_dataset(record_ids, data_dir=RAW_DATA_DIR):
    """Build a full dataset (X, y) from a list of record IDs.

    Returns:
        X: numpy array of shape (n_total_beats, window_length)
        y: numpy array of shape (n_total_beats,) with integer class indices
    """
    all_windows = []
    all_labels = []

    for record_id in record_ids:
        try:
            windows, labels = load_and_segment_record(record_id, data_dir)
        except FileNotFoundError:
            print(f"  [skip] record {record_id} not found in {data_dir}")
            continue

        all_windows.append(windows)
        all_labels.append(labels)
        print(f"  record {record_id}: {len(labels)} beats")

    if not all_windows:
        raise FileNotFoundError(
            f"No records were found in {Path(data_dir).resolve()}. "
            "Check that the MIT-BIH files are downloaded there."
        )

    X = np.concatenate(all_windows, axis=0)
    y_str = np.concatenate(all_labels, axis=0)
    y = np.array([CLASS_TO_IDX[label] for label in y_str])

    return X, y


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print("Building training set (patient-independent)...")
    X_train, y_train = build_dataset(TRAIN_RECORDS)

    print("\nBuilding test set (patient-independent)...")
    X_test, y_test = build_dataset(TEST_RECORDS)

    print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape},  y_test shape:  {y_test.shape}")

    # Class distribution
    print("\nTrain class distribution:")
    for cls, idx in CLASS_TO_IDX.items():
        count = np.sum(y_train == idx)
        print(f"  {cls}: {count} ({100 * count / len(y_train):.1f}%)")

    print("\nTest class distribution:")
    for cls, idx in CLASS_TO_IDX.items():
        count = np.sum(y_test == idx)
        print(f"  {cls}: {count} ({100 * count / len(y_test):.1f}%)")

    # Save
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"), y_test)

    print(f"\nSaved processed arrays to {PROCESSED_DATA_DIR}/")


if __name__ == "__main__":
    main()