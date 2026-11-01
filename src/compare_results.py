"""
compare_results.py

Builds the single cross-model comparison table and the error-analysis
figures for the final report, from the per-model result files in results/.

Every model script (svm_baseline.py, cnn_model.py, cnn_lstm_model.py,
metaheuristic_tuning.py, grid_search_tuning.py) writes a
`<name>_results.txt` file with the same layout: an accuracy line, a
scikit-learn classification_report, and a confusion matrix. This script
parses the confusion matrix of each file (the report is recomputed from it,
so the numbers are exactly consistent) and writes:

    results/comparison_table.csv   one row per model, all common metrics
    results/comparison_table.md    the same table in Markdown
    results/error_analysis.txt     top confusions + per-class recall per model
    figures/per_class_f1.png       grouped bar chart of per-class F1
    figures/confusion_matrices.png row-normalised confusion matrices
    figures/tuning_comparison.png  CNN tuning arms: accuracy vs macro-F1

Usage:
    python src/compare_results.py
"""

import csv
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
CLASS_NAMES = ["N", "S", "V", "F", "Q"]  # must match CLASS_TO_IDX order in preprocessing.py

# (display name, results file stem, group). Order = order in the table.
MODELS = [
    ("SVM", "svm", "Model 1"),
    ("Random Forest", "random_forest", "Model 1"),
    ("1D-CNN (Adam, default HPs)", "cnn_baseline", "Model 2"),
    ("1D-CNN (grid-search tuned)", "grid_tuned_cnn", "Model 2 tuning"),
    ("1D-CNN (GWO tuned)", "gwo_tuned_cnn", "Model 2 tuning"),
    ("1D-CNN (WOA tuned)", "woa_tuned_cnn", "Model 2 tuning"),
    ("CNN-LSTM", "cnn_lstm", "Model 3"),
]


def parse_confusion_matrix(path):
    """Read the confusion matrix printed at the end of a results file.

    Args:
        path: Path to a `<name>_results.txt` file.

    Returns:
        numpy int array of shape (5, 5), rows = true class, cols = predicted.
    """
    text = path.read_text()
    block = text.split("Confusion matrix", 1)[1]
    rows = re.findall(r"\[\s*([\d\s]+?)\s*\]", block)
    cm = np.array([[int(v) for v in r.split()] for r in rows])
    assert cm.shape == (5, 5), f"{path.name}: expected 5x5 confusion matrix, got {cm.shape}"
    return cm


def metrics_from_cm(cm):
    """Compute the common evaluation metrics from a confusion matrix.

    Args:
        cm: (5, 5) int array, rows = true, cols = predicted.

    Returns:
        dict with accuracy, macro/weighted F1, macro precision/recall,
        and per-class precision, recall and F1 (arrays of length 5).
    """
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        "accuracy": tp.sum() / cm.sum(),
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "macro_f1": f1.mean(),
        "weighted_f1": (f1 * support).sum() / support.sum(),
        # Q has only 7 test beats, so also report macro-F1 over N/S/V/F.
        "macro_f1_no_q": f1[:4].mean(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
        "n_test": int(cm.sum()),
    }


def load_all():
    """Parse every available results file.

    Returns:
        list of (display_name, group, cm, metrics) for files that exist.
    """
    loaded = []
    for name, stem, group in MODELS:
        path = RESULTS_DIR / f"{stem}_results.txt"
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        cm = parse_confusion_matrix(path)
        loaded.append((name, group, cm, metrics_from_cm(cm)))
    return loaded


def write_tables(loaded):
    """Write comparison_table.csv and comparison_table.md."""
    header = (
        ["Model", "Group", "Test beats", "Accuracy", "Macro P", "Macro R", "Macro F1",
         "Macro F1 (no Q)", "Weighted F1"]
        + [f"F1 {c}" for c in CLASS_NAMES]
        + [f"Recall {c}" for c in CLASS_NAMES]
    )
    rows = []
    for name, group, _, m in loaded:
        rows.append(
            [name, group, m["n_test"]]
            + [f"{m[k]:.4f}" for k in ("accuracy", "macro_precision", "macro_recall",
                                       "macro_f1", "macro_f1_no_q", "weighted_f1")]
            + [f"{v:.4f}" for v in m["f1"]]
            + [f"{v:.4f}" for v in m["recall"]]
        )

    with open(RESULTS_DIR / "comparison_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    md_cols = ["Model", "Accuracy", "Macro F1", "Macro F1 (no Q)", "Weighted F1"] + [
        f"F1 {c}" for c in CLASS_NAMES
    ]
    lines = ["| " + " | ".join(md_cols) + " |", "|" + "---|" * len(md_cols)]
    for name, _, _, m in loaded:
        vals = [m["accuracy"], m["macro_f1"], m["macro_f1_no_q"], m["weighted_f1"], *m["f1"]]
        lines.append("| " + name + " | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    (RESULTS_DIR / "comparison_table.md").write_text(
        "# Test-set comparison (patient-independent DS2 split)\n\n"
        + "\n".join(lines)
        + "\n\nCNN-LSTM is scored on 49,609 beat sequences (the first 4 beats of each"
        " test record have no full 5-beat history); all other models on 49,697 beats.\n"
    )


def write_error_analysis(loaded):
    """Write the top off-diagonal confusions and per-class recall per model."""
    out = []
    for name, _, cm, m in loaded:
        out.append(f"=== {name} ===")
        out.append("Recall per class: " + ", ".join(
            f"{c}={r:.3f}" for c, r in zip(CLASS_NAMES, m["recall"])))
        out.append("Precision per class: " + ", ".join(
            f"{c}={p:.3f}" for c, p in zip(CLASS_NAMES, m["precision"])))
        off = [(cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j])
               for i in range(5) for j in range(5) if i != j]
        off.sort(reverse=True)
        total_err = sum(o[0] for o in off)
        out.append(f"Misclassified beats: {total_err} of {cm.sum()}")
        out.append("Top confusions (true -> predicted):")
        for count, t, p in off[:5]:
            out.append(f"  {t} -> {p}: {count} ({100 * count / total_err:.1f}% of all errors,"
                       f" {100 * count / cm[CLASS_NAMES.index(t)].sum():.1f}% of true {t})")
        out.append("")
    (RESULTS_DIR / "error_analysis.txt").write_text("\n".join(out))


def plot_per_class_f1(loaded):
    """Grouped bar chart of per-class F1 for every model."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    n = len(loaded)
    width = 0.8 / n
    x = np.arange(len(CLASS_NAMES))
    cmap = plt.get_cmap("tab10")
    for k, (name, _, _, m) in enumerate(loaded):
        ax.bar(x + (k - (n - 1) / 2) * width, m["f1"], width, label=name, color=cmap(k))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(CLASS_NAMES, loaded[0][3]["support"])])
    ax.set_ylabel("F1-score (test set)")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 on the patient-independent test set")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_f1.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrices(loaded):
    """Row-normalised confusion matrix (recall view) for every model."""
    n = len(loaded)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, _, cm, _) in zip(axes, loaded):
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.set_title(name, fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Row-normalised confusion matrices (each row sums to 1)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=200)
    plt.close(fig)


def plot_tuning(loaded):
    """Accuracy and macro-F1 of the four CNN tuning arms side by side."""
    arms = [(n, m) for n, g, _, m in loaded if g in ("Model 2", "Model 2 tuning")]
    if not arms:
        return
    labels = [n.replace("1D-CNN ", "").strip("()") for n, _ in arms]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, [m["accuracy"] for _, m in arms], 0.4, label="Accuracy")
    ax.bar(x + 0.2, [m["macro_f1"] for _, m in arms], 0.4, label="Macro F1")
    for i, (_, m) in enumerate(arms):
        ax.text(i - 0.2, m["accuracy"] + 0.01, f"{m['accuracy']:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, m["macro_f1"] + 0.01, f"{m['macro_f1']:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("1D-CNN: default vs grid vs GWO vs WOA tuning (test set)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tuning_comparison.png", dpi=200)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    loaded = load_all()
    write_tables(loaded)
    write_error_analysis(loaded)
    plot_per_class_f1(loaded)
    plot_confusion_matrices(loaded)
    plot_tuning(loaded)
    print((RESULTS_DIR / "comparison_table.md").read_text())
    print(f"Wrote tables to {RESULTS_DIR}/ and figures to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
