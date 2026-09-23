"""
preprocessing.py

Shared preprocessing pipeline for the ECG arrhythmia classification project.
Loads MIT-BIH records, filters the signal, segments fixed-length windows around
each annotated beat, maps labels to AAMI EC57 classes, and produces a
patient-independent train/test split.

All three models (SVM/RF baseline, 1D-CNN, CNN-LSTM) should load their data
via build_dataset() (or the saved data/processed/*.npy arrays it produces)
so every model is trained/evaluated on identical data.

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

    Also returns each kept beat's absolute R-peak sample index (position in
    the original, unfiltered recording). These indices are needed downstream
    to compute RR-interval features correctly -- RR intervals must only be
    computed between consecutive beats *within the same record*, never across
    record boundaries (different records are different half-hour recordings
    from different/unrelated time spans, so a "gap" between the last beat of
    one record and the first beat of the next is meaningless).

    Args:
        record_id: string record name, e.g. '100'.
        data_dir: directory containing the raw .dat/.hea/.atr files.

    Returns:
        windows: numpy array of shape (n_beats, WINDOW_BEFORE + WINDOW_AFTER)
        labels: numpy array of shape (n_beats,) with AAMI class strings
        sample_indices: numpy array of shape (n_beats,) with each beat's
            absolute R-peak sample index within this record (int), in the
            same order as windows/labels.
    """
    record_path = os.path.join(data_dir, record_id)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    # Use channel 0 (MLII lead in most MIT-BIH records)
    signal = record.p_signal[:, 0]
    filtered = bandpass_filter(signal)

    windows = []
    labels = []
    sample_indices = []

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
        sample_indices.append(sample_idx)

    return np.array(windows), np.array(labels), np.array(sample_indices, dtype=np.int64)


def compute_rr_features(sample_indices, fs=FS, local_window=10):
    """Compute RR-interval-derived features for the beats of a single record.

    Must be called separately per record (never on sample indices concatenated
    across multiple records), since RR intervals are only meaningful between
    consecutive beats of the same continuous recording.

    Features per beat (4 columns, all in seconds except the unitless ratio):
        prev_rr:      interval (s) from the previous beat to this beat.
        next_rr:      interval (s) from this beat to the next beat.
        local_avg_rr: mean of the previous `local_window` prev_rr values
                       (early beats use whatever history is available).
        rr_ratio:     prev_rr / next_rr, a standard measure of local rhythm
                       irregularity (close to 1.0 for a regular rhythm).

    Boundary handling: for the first beat in a record there is no previous
    beat, so prev_rr falls back to the first available interval (next_rr);
    symmetrically the last beat's next_rr falls back to the last available
    interval. This avoids NaNs while keeping values within a physiologically
    plausible range, instead of arbitrarily using 0.

    Args:
        sample_indices: 1D numpy array of shape (n_beats,), absolute R-peak
            sample indices for ONE record, in temporal order.
        fs: sampling frequency in Hz.
        local_window: number of preceding beats to average for local_avg_rr.

    Returns:
        numpy array of shape (n_beats, 4): [prev_rr, next_rr, local_avg_rr, rr_ratio]
    """
    sample_indices = np.asarray(sample_indices, dtype=np.float64)
    n = len(sample_indices)

    if n == 0:
        return np.zeros((0, 4))

    if n == 1:
        # Single beat in the record: no neighbouring interval exists at all.
        return np.array([[0.0, 0.0, 0.0, 1.0]])

    rr_intervals = np.diff(sample_indices) / fs  # shape (n-1,)

    prev_rr = np.empty(n)
    next_rr = np.empty(n)
    prev_rr[0] = rr_intervals[0]     # first beat: fall back to first interval
    prev_rr[1:] = rr_intervals
    next_rr[-1] = rr_intervals[-1]   # last beat: fall back to last interval
    next_rr[:-1] = rr_intervals

    local_avg_rr = np.empty(n)
    for i in range(n):
        lo = max(0, i - local_window)
        local_avg_rr[i] = np.mean(prev_rr[lo:i + 1])

    rr_ratio = np.where(next_rr != 0, prev_rr / np.where(next_rr != 0, next_rr, 1.0), 1.0)

    return np.stack([prev_rr, next_rr, local_avg_rr, rr_ratio], axis=1)


def build_dataset(record_ids, data_dir=RAW_DATA_DIR):
    """Build a full dataset (X, y, rr_features) from a list of record IDs.

    RR-interval features are computed per-record (using each record's own
    beat sample indices) before concatenating across records, so values never
    leak across record boundaries.

    Returns:
        X: numpy array of shape (n_total_beats, window_length)
        y: numpy array of shape (n_total_beats,) with integer class indices
        rr_features: numpy array of shape (n_total_beats, 4), see
            compute_rr_features() for column definitions.
    """
    all_windows = []
    all_labels = []
    all_rr_features = []

    for record_id in record_ids:
        try:
            windows, labels, sample_indices = load_and_segment_record(record_id, data_dir)
        except FileNotFoundError:
            print(f"  [skip] record {record_id} not found in {data_dir}")
            continue

        rr_features = compute_rr_features(sample_indices)

        all_windows.append(windows)
        all_labels.append(labels)
        all_rr_features.append(rr_features)
        print(f"  record {record_id}: {len(labels)} beats")

    if not all_windows:
        raise FileNotFoundError(
            f"No records were found in {Path(data_dir).resolve()}. "
            "Check that the MIT-BIH files are downloaded there."
        )

    X = np.concatenate(all_windows, axis=0)
    y_str = np.concatenate(all_labels, axis=0)
    y = np.array([CLASS_TO_IDX[label] for label in y_str])
    rr_features = np.concatenate(all_rr_features, axis=0)

    return X, y, rr_features


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print("Building training set (patient-independent)...")
    X_train, y_train, rr_train = build_dataset(TRAIN_RECORDS)

    print("\nBuilding test set (patient-independent)...")
    X_test, y_test, rr_test = build_dataset(TEST_RECORDS)

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

    # Save (X/y arrays kept exactly as before for backward compatibility;
    # RR-interval features are saved separately alongside them)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"), y_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, "rr_features_train.npy"), rr_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "rr_features_test.npy"), rr_test)

    print(f"\nSaved processed arrays (including RR-interval features) to {PROCESSED_DATA_DIR}/")


if __name__ == "__main__":
    main()