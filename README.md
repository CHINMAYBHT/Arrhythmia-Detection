# Arrhythmia Detection (ICT 4442 mini-project)

Classifying MIT-BIH ECG heartbeats into 5 AAMI EC57 classes (Normal,
Supraventricular ectopic, Ventricular ectopic, Fusion, Unknown) by comparing
three model architectures, plus a side experiment on metaheuristic
hyperparameter tuning. See
[documentation/About/project_simple_terms.md](documentation/About/project_simple_terms.md)
for a plain-language description of the goal.

## Models

| # | Model | File | Status |
|---|-------|------|--------|
| 1 | SVM / Random Forest (hand-crafted features: morphology, wavelet, RR-interval) | [src/svm_baseline.py](src/svm_baseline.py) | Implemented, runs locally |
| 2 | 1D-CNN (raw beat windows) | [src/cnn_model.py](src/cnn_model.py) | Implemented, **run on Kaggle** |
| 3 | CNN-LSTM hybrid (5-beat sequences, per-patient) | [src/cnn_lstm_model.py](src/cnn_lstm_model.py) | Implemented, **run on Kaggle** |
| — | GWO / WOA hyperparameter tuning for Model 2 | [src/metaheuristic_tuning.py](src/metaheuristic_tuning.py) | Implemented, **run on Kaggle** |
| — | Grid-search control (same budget as GWO/WOA) | [src/grid_search_tuning.py](src/grid_search_tuning.py) | Implemented, **run on Kaggle** |

The combined comparison table is in [results/comparison_table.md](results/comparison_table.md),
the error analysis in [results/error_analysis.txt](results/error_analysis.txt), figures in
[figures/](figures/), and the final report source in [report/Final_Report.tex](report/Final_Report.tex).

All models share one preprocessing pipeline ([src/preprocessing.py](src/preprocessing.py))
and one patient-independent train/test split, and log accuracy/precision/recall/F1/confusion-matrix
in the same format under `results/`, so they're directly comparable.

## Why local + Kaggle

This local dev environment is Python 3.14 with no GPU, and no stable
TensorFlow wheel exists yet for that Python version (only a `2.22.0rc0`
release candidate). So:

- **Model 1 (SVM/RF)** is pure scikit-learn — run it locally.
- **Models 2, 3, and the GWO/WOA tuning** need TensorFlow/Keras and benefit
  hugely from a GPU — run these on **Kaggle** (free T4/P100 GPU, TF
  preinstalled), using [kaggle_pipeline.ipynb](kaggle_pipeline.ipynb).

## Project structure

```
ad/
├── src/
│   ├── download_data.py        # pulls MIT-BIH via wfdb into data/raw/
│   ├── preprocessing.py        # filtering, beat segmentation, AAMI labels,
│   │                            RR-interval features, train/test split
│   ├── svm_baseline.py         # Model 1: SVM / Random Forest
│   ├── cnn_model.py            # Model 2: 1D-CNN
│   ├── cnn_lstm_model.py       # Model 3: CNN-LSTM hybrid
│   ├── metaheuristic_tuning.py # GWO / WOA hyperparameter search for Model 2
│   ├── grid_search_tuning.py   # grid-search control arm for the tuning experiment
│   └── compare_results.py      # builds comparison table, error analysis, figures
├── kaggle_pipeline.ipynb       # self-contained notebook: run Models 1-3 + tuning on Kaggle
├── data/            (gitignored) raw/ and processed/ MIT-BIH arrays
├── models/          (gitignored) trained model files
├── logs/            (gitignored) per-epoch training logs (CSV)
├── results/         per-model reports, comparison_table.{md,csv}, error_analysis.txt
├── figures/         per-class F1, confusion matrices, tuning comparison (from compare_results.py)
├── report/          final report (LaTeX, compile on Overleaf with report/figures/)
├── documentation/   project guidelines, synopsis, plain-language explainer
└── requirements.txt
```

## Running locally (Model 1)

```bash
pip install -r requirements.txt
python src/download_data.py       # downloads MIT-BIH into data/raw/ (~100MB)
python src/preprocessing.py       # builds data/processed/*.npy
python src/svm_baseline.py        # trains RF + SVM, writes results/*.txt
python src/compare_results.py     # comparison table + error analysis + figures from results/
```

## Running on Kaggle (Models 2, 3, GWO/WOA)

1. Go to [kaggle.com/code](https://kaggle.com/code) → **New Notebook**, or
   upload [kaggle_pipeline.ipynb](kaggle_pipeline.ipynb) directly (File →
   Upload Notebook).
2. In the notebook's Settings (right sidebar): **Accelerator → GPU T4 x2**
   (or P100), **Internet → On**. (One-time phone verification on your
   Kaggle account is required to unlock this if you haven't already.)
3. Run all cells top to bottom. The notebook writes out the current
   `src/*.py` files, downloads MIT-BIH, preprocesses it, and runs every
   model in order.
4. Budget **45-75 minutes** total — the GWO/WOA search cell dominates (see
   [metaheuristic_tuning.py](src/metaheuristic_tuning.py)'s docstring for the
   exact compute-cost accounting: population size × iterations × epochs per
   candidate).
5. Download `outputs.zip` (built in the notebook's last-but-one cell) from
   the Output panel to get `results/`, `models/`, and `logs/` back onto your
   machine.

If you change any `src/*.py` file, regenerate the notebook so it stays in
sync (the notebook embeds each file's exact contents at generation time —
it does not read the repo live).

## Evaluation protocol

- **Patient-independent split**: fixed train/test record lists in
  `preprocessing.py` (`TRAIN_RECORDS` / `TEST_RECORDS`), so no patient
  appears in both splits.
- **Same metrics everywhere**: accuracy, per-class precision/recall/F1, and
  a confusion matrix, via `sklearn.metrics.classification_report` /
  `confusion_matrix`.
- **Class imbalance** (S/F/Q are rare — see `results/random_forest_results.txt`
  / `results/svm_results.txt` from the first pass): handled via SMOTE
  (SVM/RF, training set only) and/or class-weighted loss (all models).
- **Results land in `results/`** with one file per model
  (`<model_name>_results.txt`) so they can be pulled into a single
  comparison table.

## Known caveats

- `cnn_model.py` and `cnn_lstm_model.py` pass the test set as Keras
  `validation_data` (used by EarlyStopping / ReduceLROnPlateau), so their
  test scores are slightly optimistic. A patient-wise validation split is
  listed as future work in the final report.
- Each model was run once with seed 42; differences of a few points are
  within run-to-run variation.
- The GWO/WOA search's internal validation split (used to score candidates
  during the search) is a stratified random split of training beats, not a
  further patient-independent split, since patient IDs aren't currently
  threaded through `preprocessing.py`'s saved arrays. The *final* test
  evaluation stays fully patient-independent regardless.
- `data/`, `models/`, and `logs/` are gitignored — nothing there persists
  across a fresh clone; re-run the pipeline to regenerate them.
