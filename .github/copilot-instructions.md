# Project Context for GitHub Copilot

This is a Deep Learning mini-project (ICT 4442) for classifying cardiac arrhythmias from ECG signals.

## Project Summary
We compare three model architectures on the MIT-BIH Arrhythmia Database to classify heartbeats
into arrhythmia categories, and separately test whether metaheuristic optimization improves
CNN hyperparameter tuning.

## Dataset
- MIT-BIH Arrhythmia Database (PhysioNet), accessed via the `wfdb` library.
- 48 half-hour two-channel ECG recordings from 47 subjects, sampled at 360 Hz.
- Beat-level annotations labeled per the AAMI EC57 standard (Normal, Supraventricular ectopic,
  Ventricular ectopic, Fusion, Unknown).
- Data must be split **patient-independently** (by subject ID) into train/test to avoid data leakage.
- Beats are segmented into fixed-length windows centered on each annotation.

## Models (one per team member)
1. **SVM / Random Forest baseline** — trained on hand-crafted features: RR-intervals, QRS
   morphology, wavelet coefficients (use `PyWavelets`). Uses `scikit-learn`.
2. **1D-CNN** — trained directly on raw ECG signal segments. Built in `tensorflow`/`keras`.
   Tuned first with standard Adam/SGD, later with metaheuristic optimizers (GWO, WOA).
3. **CNN-LSTM hybrid** — combines convolutional feature extraction with an LSTM layer to
   capture beat-to-beat temporal dependencies. Built in `tensorflow`/`keras`.

## Evaluation Protocol (must be identical across all models)
- Same train/test split (patient-independent).
- Metrics: accuracy, precision, recall, F1-score, confusion matrix.
- Results should be saved in a comparable format (e.g. a shared results dict/CSV) so they can
  be combined into one comparison table across all models.

## Folder Structure
- `data/raw/` — original downloaded MIT-BIH files (gitignored)
- `data/processed/` — preprocessed/segmented data (gitignored)
- `src/` — all source code (preprocessing, model scripts, utils)
- `models/` — saved trained model files (gitignored)
- `documentation/` — reports and project docs
- `logs/` — training logs

## Coding Conventions
- Use Python type hints where reasonable.
- Keep preprocessing, feature extraction, training, and evaluation as separate functions/modules
  so each team member's model script can reuse the shared pipeline in `src/`.
- Prefer `scikit-learn` metrics functions (`classification_report`, `confusion_matrix`) for
  consistency across all three models.
- Add a short docstring to every function explaining input/output shapes — ECG signal shapes
  are easy to mix up (samples vs. channels vs. time steps).
- When writing training code, log metrics per epoch so results can go into the interim/final report.

## What NOT to do
- Don't suggest downloading datasets other than MIT-BIH.
- Don't use random (non-patient-independent) train/test splits.
- Don't hardcode file paths — use relative paths from the project root or a config file.
