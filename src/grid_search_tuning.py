"""
grid_search_tuning.py

Control experiment for the GWO/WOA side experiment in
src/metaheuristic_tuning.py: does a plain grid search, given the SAME
compute budget, find 1D-CNN hyperparameters as good as the metaheuristics?
If it does, the extra machinery of GWO/WOA isn't buying anything on this
dataset/model size.

This script does NOT redefine the CNN, the data pipeline, or the fitness
function. It reuses, unchanged:
    from cnn_model.py:
        build_cnn_model(), load_data(), prepare_inputs(),
        compute_class_weights(), train_and_evaluate()
    from metaheuristic_tuning.py:
        make_fitness_fn() -- the exact macro-F1-on-validation fitness used
            to score every GWO/WOA candidate (internally calls
            build_cnn_model() and compute_class_weights()),
        decode_position() -- the position-vector -> hyperparams mapping,
        retrain_and_evaluate_best() -- the final full-training-set retrain
            via train_and_evaluate(),
        FITNESS_EPOCHS, FITNESS_PATIENCE (baked into make_fitness_fn()),
        BATCH_SIZE, FINAL_EPOCHS, VAL_FRACTION, RANDOM_STATE.
The train/validation split is the same train_test_split() call, with the
same VAL_FRACTION / RANDOM_STATE / stratification, that
metaheuristic_tuning.main() makes, so every grid candidate is scored on
exactly the same beats as every GWO/WOA candidate.

--------------------------------------------------------------------------
RUN THIS ON KAGGLE (or any machine with a real TensorFlow+GPU install) -
NOT in this local dev environment.
--------------------------------------------------------------------------
Same constraint as cnn_model.py / metaheuristic_tuning.py.

Search space and compute cost:
    Grid over the three hyperparameters most likely to matter for this
    small CNN; everything else stays at cnn_model.py's baseline value
    (dropout_rate=0.3, dense_units=64, pool_size=2):
        learning_rate     in {3e-4, 1e-3, 3e-3}        (3 values)
        num_filters base  in {16, 32, 64}              (3 values; blocks
                                                        use base, 2x, 4x)
        kernel_size       in {3, 5, 7, 9}              (4 values)
    3 * 3 * 4 = 36 candidates, each trained for up to FITNESS_EPOCHS = 8
    epochs with EarlyStopping(patience=FITNESS_PATIENCE = 3) -- exactly the
    same per-candidate budget, and the same total number of short
    trainings (POP_SIZE * MAX_ITER = 36), as ONE of the GWO/WOA runs.
    Plus 1 final retrain of the best config for FINAL_EPOCHS = 50 epochs.
    The grid contains cnn_model.py's baseline config (lr=1e-3, base 32,
    kernel 7) as one of its points.
    Budget roughly 15-30 minutes on a Kaggle T4/P100.

Outputs:
    results/grid_tuned_cnn_results.txt   (same format as every other model,
                                          written by cnn_model.evaluate_model)
    results/grid_search_scores.txt       (validation macro-F1 of every grid
                                          point, sorted best-first)
    models/grid_tuned_cnn.keras, logs/grid_tuned_cnn_training_log.csv

Usage (on Kaggle or a local machine with TensorFlow installed):
    python src/grid_search_tuning.py
"""

import os
import sys
import time
import itertools
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow import keras

# cnn_model.py and metaheuristic_tuning.py live next to this file in src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cnn_model import (  # noqa: E402
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    load_data,
    prepare_inputs,
)
from metaheuristic_tuning import (  # noqa: E402
    FITNESS_EPOCHS,
    FITNESS_PATIENCE,
    VAL_FRACTION,
    RANDOM_STATE,
    decode_position,
    make_fitness_fn,
    retrain_and_evaluate_best,
)

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

LEARNING_RATES = (3e-4, 1e-3, 3e-3)
FILTER_BASES = (16, 32, 64)
KERNEL_SIZES = (3, 5, 7, 9)

# Hyperparameters not searched here are pinned to cnn_model.py's baseline.
FIXED_DROPOUT_RATE = 0.3
FIXED_DENSE_UNITS = 64


def grid_positions():
    """Yield every grid point as a position vector in metaheuristic_tuning's
    encoding ([log10_lr, dropout_rate, num_filters_base, dense_units,
    kernel_size]), so it can be scored by the unchanged make_fitness_fn().
    """
    for lr, base, kernel in itertools.product(LEARNING_RATES, FILTER_BASES, KERNEL_SIZES):
        yield np.array([
            np.log10(lr), FIXED_DROPOUT_RATE, base, FIXED_DENSE_UNITS, kernel,
        ], dtype=float)


def _check_decodes_exactly(position):
    """Guard against decode_position() silently clipping/rounding a grid
    point to something other than what the grid says it is (e.g. if
    metaheuristic_tuning.BOUNDS were narrowed later).
    """
    hp = decode_position(position)
    expected_lr = 10.0 ** position[0]
    assert np.isclose(hp["learning_rate"], expected_lr), (hp, position)
    assert hp["num_filters"][0] == int(position[2]), (hp, position)
    assert hp["kernel_size"] == int(position[4]), (hp, position)
    assert np.isclose(hp["dropout_rate"], position[1]), (hp, position)
    assert hp["dense_units"] == int(position[3]), (hp, position)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    keras.utils.set_random_seed(RANDOM_SEED)

    print(f"Loading processed data from {PROCESSED_DATA_DIR} ...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data()
    print(f"Train: {X_train_raw.shape}, Test: {X_test_raw.shape}")

    # Identical split to metaheuristic_tuning.main(): validation carved out of
    # the TRAINING data only; the test set is untouched until the final retrain.
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_train_raw, y_train_raw,
        test_size=VAL_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y_train_raw,
    )

    X_tr = prepare_inputs(X_tr_raw)
    X_val = prepare_inputs(X_val_raw)
    input_shape = X_tr.shape[1:]
    print(f"Search train: {X_tr.shape}, search val: {X_val.shape}")

    fitness_fn = make_fitness_fn(X_tr, y_tr, X_val, y_val, input_shape)

    positions = list(grid_positions())
    for pos in positions:
        _check_decodes_exactly(pos)
    n = len(positions)

    print("\n" + "=" * 70)
    print(f"Running grid search over {n} candidates "
          f"(up to {FITNESS_EPOCHS} epochs each, patience {FITNESS_PATIENCE})")
    print("=" * 70)

    t0 = time.time()
    scored = []
    for i, pos in enumerate(positions):
        score, hp = fitness_fn(pos)
        scored.append((score, hp))
        print(f"[GRID] {i + 1}/{n} macro_f1={score:.4f} hp={hp}")
    search_time = time.time() - t0

    # Stable sort keeps the earlier grid point on ties.
    scored.sort(key=lambda s: -s[0])
    best_val_f1, best_hp = scored[0]
    print(f"Grid best validation macro-F1: {best_val_f1:.4f}, "
          f"hyperparams: {best_hp} (search took {search_time:.1f}s)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    scores_path = os.path.join(RESULTS_DIR, "grid_search_scores.txt")
    with open(scores_path, "w") as f:
        f.write(f"Grid search validation macro-F1 ({n} candidates, "
                f"FITNESS_EPOCHS={FITNESS_EPOCHS}, patience={FITNESS_PATIENCE}, "
                f"search took {search_time:.1f}s)\n\n")
        for score, hp in scored:
            f.write(f"{score:.4f}  {hp}\n")
    print(f"Saved per-candidate scores to {scores_path}")

    # Retrains for FINAL_EPOCHS (50) on the full training set and writes
    # results/grid_tuned_cnn_results.txt via cnn_model.evaluate_model().
    retrain_and_evaluate_best(
        "grid_tuned_cnn", best_hp, X_train_raw, y_train_raw, X_test_raw, y_test_raw,
    )

    print("\nDone. Compare results/grid_tuned_cnn_results.txt against "
          "results/gwo_tuned_cnn_results.txt and results/woa_tuned_cnn_results.txt.")


if __name__ == "__main__":
    main()
