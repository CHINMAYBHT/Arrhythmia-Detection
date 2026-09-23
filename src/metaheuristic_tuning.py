"""
metaheuristic_tuning.py

Side experiment (see .github/copilot-instructions.md): does metaheuristic
optimization -- Grey Wolf Optimizer (GWO, Mirjalili et al. 2014) and Whale
Optimization Algorithm (WOA, Mirjalili & Lewis 2016) -- find better 1D-CNN
hyperparameters than the standard Adam-trained baseline in src/cnn_model.py?

This script does NOT redefine the CNN. It imports and calls the exact
functions already written in src/cnn_model.py:
    build_cnn_model(input_shape, n_classes=5, num_filters=(32,64,128),
                     kernel_size=7, pool_size=2, dropout_rate=0.3,
                     dense_units=64, learning_rate=1e-3)
    load_data(), prepare_inputs(X), compute_class_weights(y_train),
    evaluate_model(name, model, X_test, y_test),
    train_and_evaluate(X_train, y_train, X_test, y_test, hyperparams=None, ...)
Every candidate solution produced by GWO/WOA is turned into a hyperparams
dict and fed straight into build_cnn_model()/train_and_evaluate() -- nothing
here reimplements the model architecture.

--------------------------------------------------------------------------
RUN THIS ON KAGGLE (or any machine with a real TensorFlow+GPU install) -
NOT in this local dev environment.
--------------------------------------------------------------------------
Same constraint as cnn_model.py: this repo's local environment is Python
3.14 with no working stable TensorFlow wheel and no GPU. This script is
written to be uploaded to Kaggle alongside cnn_model.py (same directory,
same data-loading conventions) and run there.

Expected compute cost (read this before hitting Run on Kaggle):
    Each fitness evaluation trains one full CNN for a few epochs. With the
    settings chosen below:
        POP_SIZE = 6 candidates, MAX_ITER = 6 iterations per optimizer,
        FITNESS_EPOCHS = 8 epochs per candidate (with early stopping, so
        often fewer in practice).
    Total fitness-evaluation trainings per optimizer = POP_SIZE * MAX_ITER
        = 6 * 6 = 36 CNN trainings (up to 8 epochs each).
    Running BOTH GWO and WOA = 72 short CNN trainings.
    Plus 2 final "retrain the best config on the full training set" runs
    (FINAL_EPOCHS = 50 each, same as the baseline in cnn_model.py).
    On a Kaggle T4/P100 GPU, each short fitness-evaluation training of this
    small 1D-CNN on MIT-BIH beat windows typically takes well under a
    minute; the two final full retrains are comparable in cost to running
    cnn_model.py's own baseline twice. Budget roughly 30-60 minutes total
    for the whole script, but this scales directly with POP_SIZE, MAX_ITER,
    FITNESS_EPOCHS and dataset size -- reduce those first if you are close
    to Kaggle's session time limit.

How to point it at Kaggle's data: identical to cnn_model.py -- set
PROCESSED_DATA_DIR (and optionally MODELS_DIR/LOGS_DIR/RESULTS_DIR) env
vars before running, or edit the constants directly.

Usage (on Kaggle or a local machine with TensorFlow installed):
    python src/metaheuristic_tuning.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from tensorflow import keras

# cnn_model.py lives next to this file in src/ -- import its exact public
# interface directly rather than re-implementing anything here. When this
# script is run as `python src/metaheuristic_tuning.py` (or uploaded next
# to cnn_model.py on Kaggle), the script's own directory is already on
# sys.path, but we add it defensively in case this is imported differently.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cnn_model import (  # noqa: E402
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    CLASS_NAMES,
    N_CLASSES,
    RANDOM_SEED,
    build_cnn_model,
    load_data,
    prepare_inputs,
    compute_class_weights,
    evaluate_model,
    train_and_evaluate,
)

# ---------------------------------------------------------------------------
# Config: metaheuristic search settings
# ---------------------------------------------------------------------------
# Deliberately small -- see the module docstring above for the compute-cost
# accounting. This is a mini-project demonstration of GWO/WOA, not a
# research-grade hyperparameter search.
POP_SIZE = 6          # wolves / whales
MAX_ITER = 6          # optimizer iterations
FITNESS_EPOCHS = 8    # epochs per candidate during search (early stopping applies)
FITNESS_PATIENCE = 3  # EarlyStopping patience during the (short) fitness runs
FINAL_EPOCHS = 50     # epochs for the final retrain-on-full-data run (matches baseline)
BATCH_SIZE = 128
VAL_FRACTION = 0.15   # carved out of the TRAINING set only, never the test set

RANDOM_STATE = RANDOM_SEED

# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------
# Each optimizer works with a real-valued position vector of length 5. The
# dimensions and their (lower, upper) bounds are:
#
#   idx  name               bounds           meaning / decoding
#   ---  -----------------  ---------------  --------------------------------
#   0    log10_lr           (-4.0, -2.0)     learning_rate = 10**log10_lr
#                                             -> continuous, searched on a
#                                             log scale, i.e. lr in [1e-4, 1e-2]
#   1    dropout_rate       (0.1, 0.5)       used directly (continuous)
#   2    num_filters_base   (16, 128)        rounded to nearest int; the 3
#                                             conv blocks use
#                                             (base, base*2, base*4) filters,
#                                             mirroring cnn_model.py's default
#                                             (32, 64, 128) shape
#   3    dense_units        (32, 256)        rounded to nearest int
#   4    kernel_size        (3, 11)          rounded to nearest int, then
#                                             nudged to the nearest odd
#                                             number (odd kernels keep
#                                             "same" padding symmetric) and
#                                             re-clipped to [3, 11]
#
# GWO/WOA both operate on these real-valued vectors and clip back into
# BOUNDS after every position update; decode_position() below is the single
# place that maps a raw vector to the keyword args build_cnn_model() expects.

BOUNDS = np.array([
    (-4.0, -2.0),   # log10 learning rate
    (0.1, 0.5),     # dropout_rate
    (16.0, 128.0),  # num_filters_base
    (32.0, 256.0),  # dense_units
    (3.0, 11.0),    # kernel_size
])
DIM = BOUNDS.shape[0]
LOWER = BOUNDS[:, 0]
UPPER = BOUNDS[:, 1]


def decode_position(pos):
    """Map a raw continuous position vector to a build_cnn_model() kwargs dict.

    Args:
        pos: array-like of length DIM (see BOUNDS table above for the
            per-dimension meaning and bounds). Not assumed to already be
            clipped to bounds -- this function clips defensively.

    Returns:
        dict with keys learning_rate, dropout_rate, num_filters, dense_units,
        kernel_size, ready to pass as build_cnn_model(**hyperparams) /
        train_and_evaluate(..., hyperparams=hyperparams).
    """
    pos = np.clip(np.asarray(pos, dtype=float), LOWER, UPPER)

    learning_rate = float(10.0 ** pos[0])
    dropout_rate = float(pos[1])

    filters_base = int(round(pos[2]))
    dense_units = int(round(pos[3]))

    kernel_size = int(round(pos[4]))
    if kernel_size % 2 == 0:
        kernel_size += 1  # keep kernels odd for symmetric "same" padding
    kernel_size = int(np.clip(kernel_size, BOUNDS[4, 0], BOUNDS[4, 1]))
    if kernel_size % 2 == 0:  # clipping could re-introduce an even number at the edge
        kernel_size -= 1

    num_filters = (filters_base, filters_base * 2, filters_base * 4)

    return dict(
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        num_filters=num_filters,
        dense_units=dense_units,
        kernel_size=kernel_size,
    )


def random_population(pop_size, rng):
    """Uniformly sample `pop_size` positions within BOUNDS.

    Returns an (pop_size, DIM) array.
    """
    return rng.uniform(LOWER, UPPER, size=(pop_size, DIM))


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------

def make_fitness_fn(X_tr, y_tr, X_val, y_val, input_shape):
    """Build a fitness(position) -> macro-F1 closure over a fixed train/val split.

    The validation split is carved out of the TRAINING data only (see
    main()); the real held-out test set from cnn_model.load_data() is never
    touched during the search, so hyperparameters are not tuned against it.

    Macro-F1 (rather than accuracy) is used as the fitness because the AAMI
    classes are heavily imbalanced (S/F/Q are rare -- see
    results/svm_results.txt / results/random_forest_results.txt), so a
    trivial "always predict N" model would score high accuracy but ~0
    macro-F1. Maximizing macro-F1 rewards candidates that actually learn the
    rare classes, which is the point of tuning in the first place.
    """
    class_weights = compute_class_weights(y_tr)

    def fitness(position):
        hyperparams = decode_position(position)
        model = build_cnn_model(input_shape, n_classes=N_CLASSES, **hyperparams)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=FITNESS_PATIENCE,
                restore_best_weights=True,
            ),
        ]
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=FITNESS_EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0,
        )

        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

        # Free GPU/CPU memory between candidates -- important since we build
        # a fresh model per fitness evaluation, many times per optimizer run.
        keras.backend.clear_session()

        return float(macro_f1), hyperparams

    return fitness


# ---------------------------------------------------------------------------
# Grey Wolf Optimizer (GWO) -- Mirjalili, Mirjalili & Lewis, 2014
# ---------------------------------------------------------------------------

def grey_wolf_optimizer(fitness_fn, pop_size=POP_SIZE, max_iter=MAX_ITER, seed=RANDOM_STATE):
    """Minimal Grey Wolf Optimizer over a continuous, bounded search space.

    Standard GWO: a pack of `pop_size` wolves searches for prey (the
    optimum). The three best wolves each iteration are labelled alpha, beta,
    delta; every wolf (including alpha/beta/delta themselves) then moves
    towards the average of the positions those three "leaders" imply,
    with the exploration/exploitation balance controlled by the coefficient
    `a`, which decays linearly from 2 to 0 over the iterations.

    Args:
        fitness_fn: callable position -> (score, hyperparams_dict), where
            higher score is better (we MAXIMIZE macro-F1 here, so GWO's
            usual "minimize" convention is flipped by tracking the largest
            score as alpha/beta/delta instead of the smallest).
        pop_size: number of wolves.
        max_iter: number of iterations.
        seed: RNG seed for reproducibility.

    Returns:
        (best_position, best_score, best_hyperparams, history) where
        history is a list of (iteration, best_score_so_far) tuples.
    """
    rng = np.random.default_rng(seed)
    positions = random_population(pop_size, rng)
    scores = np.full(pop_size, -np.inf)
    hp_cache = [None] * pop_size

    history = []
    best_score = -np.inf
    best_position = positions[0].copy()
    best_hp = None

    for it in range(max_iter):
        # Evaluate current pack.
        for i in range(pop_size):
            score, hp = fitness_fn(positions[i])
            scores[i] = score
            hp_cache[i] = hp
            print(f"[GWO] iter {it + 1}/{max_iter} wolf {i + 1}/{pop_size} "
                  f"macro_f1={score:.4f} hp={hp}")

        # Rank wolves: alpha = best, beta = 2nd best, delta = 3rd best.
        order = np.argsort(-scores)  # descending, since we maximize
        alpha_pos = positions[order[0]]
        beta_pos = positions[order[1 % pop_size]]
        delta_pos = positions[order[2 % pop_size]]

        if scores[order[0]] > best_score:
            best_score = scores[order[0]]
            best_position = positions[order[0]].copy()
            best_hp = hp_cache[order[0]]
        history.append((it, best_score))

        a = 2.0 - it * (2.0 / max_iter)  # linearly decreases 2 -> 0

        new_positions = np.empty_like(positions)
        for i in range(pop_size):
            leader_positions = (alpha_pos, beta_pos, delta_pos)
            x_estimates = []
            for leader_pos in leader_positions:
                r1, r2 = rng.random(DIM), rng.random(DIM)
                A = 2 * a * r1 - a          # exploration/exploitation coefficient
                C = 2 * r2                  # random weight on the leader's position
                D = np.abs(C * leader_pos - positions[i])
                x_estimates.append(leader_pos - A * D)
            new_positions[i] = np.mean(x_estimates, axis=0)

        positions = np.clip(new_positions, LOWER, UPPER)

    return best_position, best_score, best_hp, history


# ---------------------------------------------------------------------------
# Whale Optimization Algorithm (WOA) -- Mirjalili & Lewis, 2016
# ---------------------------------------------------------------------------

def whale_optimization_algorithm(fitness_fn, pop_size=POP_SIZE, max_iter=MAX_ITER, seed=RANDOM_STATE):
    """Minimal Whale Optimization Algorithm over a continuous, bounded space.

    Standard WOA: each whale updates its position relative to the current
    best solution ("bubble-net feeding"), choosing between three moves each
    iteration:
      - shrinking encirclement of the best whale (|A| < 1, p < 0.5),
      - a random search using a randomly chosen whale instead of the best
        one (|A| >= 1, p < 0.5), which drives exploration, or
      - a logarithmic spiral update around the best whale (p >= 0.5), which
        mimics the bubble-net attack.
    `a` decays linearly from 2 to 0 over the iterations, shrinking the
    encirclement radius as the search progresses.

    Args:
        fitness_fn: callable position -> (score, hyperparams_dict), higher
            score is better (macro-F1, maximized).
        pop_size: number of whales.
        max_iter: number of iterations.
        seed: RNG seed for reproducibility.

    Returns:
        (best_position, best_score, best_hyperparams, history) where
        history is a list of (iteration, best_score_so_far) tuples.
    """
    rng = np.random.default_rng(seed)
    positions = random_population(pop_size, rng)

    history = []
    best_score = -np.inf
    best_position = positions[0].copy()
    best_hp = None

    b = 1.0  # spiral shape constant

    for it in range(max_iter):
        scores = np.full(pop_size, -np.inf)
        for i in range(pop_size):
            score, hp = fitness_fn(positions[i])
            scores[i] = score
            print(f"[WOA] iter {it + 1}/{max_iter} whale {i + 1}/{pop_size} "
                  f"macro_f1={score:.4f} hp={hp}")
            if score > best_score:
                best_score = score
                best_position = positions[i].copy()
                best_hp = hp
        history.append((it, best_score))

        a = 2.0 - it * (2.0 / max_iter)   # linearly decreases 2 -> 0
        a2 = -1.0 + it * (-1.0 / max_iter)  # linearly decreases -1 -> -2, controls spiral l

        new_positions = np.empty_like(positions)
        for i in range(pop_size):
            r1, r2 = rng.random(DIM), rng.random(DIM)
            A = 2 * a * r1 - a
            C = 2 * r2
            p = rng.random()

            if p < 0.5:
                if np.abs(A).mean() < 1.0:
                    # Encircle the current best whale.
                    D = np.abs(C * best_position - positions[i])
                    new_positions[i] = best_position - A * D
                else:
                    # Explore: move relative to a random whale in the pack.
                    rand_whale = positions[rng.integers(pop_size)]
                    D = np.abs(C * rand_whale - positions[i])
                    new_positions[i] = rand_whale - A * D
            else:
                # Spiral update around the best whale (bubble-net attack).
                l = rng.uniform(-1.0, 1.0)
                D_best = np.abs(best_position - positions[i])
                new_positions[i] = (
                    D_best * np.exp(b * l) * np.cos(2 * np.pi * l) + best_position
                )

        positions = np.clip(new_positions, LOWER, UPPER)

    return best_position, best_score, best_hp, history


# ---------------------------------------------------------------------------
# Final retrain + evaluate on the real test set, for a tuned hyperparameter set
# ---------------------------------------------------------------------------

def retrain_and_evaluate_best(name, hyperparams, X_train, y_train, X_test, y_test):
    """Retrain a model with `hyperparams` on the FULL training set for
    FINAL_EPOCHS, then evaluate it on the held-out test set via
    cnn_model.evaluate_model() (same metric/report/confusion-matrix format
    as every other model in this project).

    This is the only place a GWO/WOA-tuned model ever touches the test set,
    and it only does so once its hyperparameters are already fixed.
    """
    print(f"\n=== Retraining final '{name}' model with tuned hyperparams: {hyperparams} ===")
    model, history = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        hyperparams=hyperparams,
        epochs=FINAL_EPOCHS,
        batch_size=BATCH_SIZE,
        model_name=name,
    )
    return model, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    keras.utils.set_random_seed(RANDOM_SEED)

    print(f"Loading processed data from {PROCESSED_DATA_DIR} ...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_data()
    print(f"Train: {X_train_raw.shape}, Test: {X_test_raw.shape}")

    # Carve a validation split out of the TRAINING data only, for use during
    # the GWO/WOA search. The real test set (X_test_raw/y_test_raw) is never
    # used until retrain_and_evaluate_best(), after hyperparameters are
    # already fixed -- this avoids tuning against the test set.
    #
    # NOTE / caveat: cnn_model.load_data() returns pre-split, pre-processed
    # arrays without patient/subject IDs attached, so this split is a
    # stratified-by-label random split of the training beats rather than a
    # further patient-independent split. That matches what's available from
    # the existing preprocessing.py output; if patient IDs are exposed
    # alongside X_train/y_train in the future, prefer a patient-independent
    # split here too, for consistency with the project's evaluation protocol.
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_train_raw, y_train_raw,
        test_size=VAL_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y_train_raw,
    )

    X_tr = prepare_inputs(X_tr_raw)
    X_val = prepare_inputs(X_val_raw)
    X_test = prepare_inputs(X_test_raw)
    input_shape = X_tr.shape[1:]
    print(f"Search train: {X_tr.shape}, search val: {X_val.shape}, "
          f"held-out test (untouched during search): {X_test.shape}")

    fitness_fn = make_fitness_fn(X_tr, y_tr, X_val, y_val, input_shape)

    # --- Grey Wolf Optimizer ------------------------------------------------
    print("\n" + "=" * 70)
    print("Running Grey Wolf Optimizer (GWO) hyperparameter search")
    print("=" * 70)
    t0 = time.time()
    gwo_best_pos, gwo_best_val_f1, gwo_best_hp, gwo_history = grey_wolf_optimizer(
        fitness_fn, pop_size=POP_SIZE, max_iter=MAX_ITER, seed=RANDOM_STATE,
    )
    gwo_search_time = time.time() - t0
    print(f"GWO best validation macro-F1: {gwo_best_val_f1:.4f}, "
          f"hyperparams: {gwo_best_hp} (search took {gwo_search_time:.1f}s)")

    # --- Whale Optimization Algorithm --------------------------------------
    print("\n" + "=" * 70)
    print("Running Whale Optimization Algorithm (WOA) hyperparameter search")
    print("=" * 70)
    t0 = time.time()
    woa_best_pos, woa_best_val_f1, woa_best_hp, woa_history = whale_optimization_algorithm(
        fitness_fn, pop_size=POP_SIZE, max_iter=MAX_ITER, seed=RANDOM_STATE + 1,
    )
    woa_search_time = time.time() - t0
    print(f"WOA best validation macro-F1: {woa_best_val_f1:.4f}, "
          f"hyperparams: {woa_best_hp} (search took {woa_search_time:.1f}s)")

    # --- Retrain best configs on the full training set, evaluate on test --
    retrain_and_evaluate_best(
        "gwo_tuned_cnn", gwo_best_hp, X_train_raw, y_train_raw, X_test_raw, y_test_raw,
    )
    retrain_and_evaluate_best(
        "woa_tuned_cnn", woa_best_hp, X_train_raw, y_train_raw, X_test_raw, y_test_raw,
    )

    # --- Final comparison summary ------------------------------------------
    # The plain-Adam baseline is NOT re-run here -- src/cnn_model.py already
    # produces it (results/cnn_baseline_results.txt / results/cnn_results.txt
    # depending on the model_name used when it was run). We just try to read
    # its accuracy back out of that file so the summary is self-contained;
    # if it isn't found, we say so explicitly rather than guessing a number.
    baseline_acc = _read_baseline_accuracy(RESULTS_DIR)

    summary_lines = []
    summary_lines.append("Metaheuristic (GWO/WOA) vs standard Adam CNN tuning - comparison")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    summary_lines.append(
        "Baseline (plain Adam, default hyperparams, from src/cnn_model.py):"
    )
    if baseline_acc is not None:
        summary_lines.append(f"  Test accuracy: {baseline_acc:.4f}")
    else:
        summary_lines.append(
            "  Not found in results/ -- run src/cnn_model.py first and re-run "
            "this script's summary step, or compare manually against its "
            "results/cnn_baseline_results.txt."
        )
    summary_lines.append("")
    summary_lines.append("GWO-tuned CNN:")
    summary_lines.append(f"  Search validation macro-F1: {gwo_best_val_f1:.4f}")
    summary_lines.append(f"  Tuned hyperparameters: {gwo_best_hp}")
    summary_lines.append("  Final test-set metrics: see results/gwo_tuned_cnn_results.txt")
    summary_lines.append("")
    summary_lines.append("WOA-tuned CNN:")
    summary_lines.append(f"  Search validation macro-F1: {woa_best_val_f1:.4f}")
    summary_lines.append(f"  Tuned hyperparameters: {woa_best_hp}")
    summary_lines.append("  Final test-set metrics: see results/woa_tuned_cnn_results.txt")
    summary_lines.append("")
    summary_lines.append(
        "Research question: is metaheuristic tuning worth it here?\n"
        "Compare the test accuracy / macro-F1 in results/cnn_baseline_results.txt\n"
        "against results/gwo_tuned_cnn_results.txt and results/woa_tuned_cnn_results.txt.\n"
        "If GWO/WOA test performance is not meaningfully better than the plain-Adam\n"
        "baseline given the extra compute (36 short trainings each on top of the\n"
        "final retrain), the practical answer for this dataset/model size is likely\n"
        "'no, not worth it' -- but let the actual numbers in those three files decide."
    )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "metaheuristic_comparison.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSaved comparison summary to {summary_path}")


def _read_baseline_accuracy(results_dir):
    """Best-effort scrape of the plain-Adam baseline's test accuracy out of
    the results file cnn_model.py's evaluate_model()/train_and_evaluate()
    already wrote (results/cnn_results.txt, since evaluate_model() is called
    there with name="CNN"; older runs may instead have written
    results/cnn_baseline_results.txt if model_name was used as the report
    name). Returns a float, or None if neither file is found/parseable.
    """
    for fname in ("cnn_results.txt", "cnn_baseline_results.txt"):
        path = Path(results_dir) / fname
        if not path.exists():
            continue
        try:
            for line in path.read_text().splitlines():
                if line.strip().lower().startswith("accuracy:"):
                    return float(line.split(":", 1)[1].strip())
        except (OSError, ValueError):
            continue
    return None


if __name__ == "__main__":
    main()
