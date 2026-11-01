# Test-set comparison (patient-independent DS2 split)

| Model | Accuracy | Macro F1 | Macro F1 (no Q) | Weighted F1 | F1 N | F1 S | F1 V | F1 F | F1 Q |
|---|---|---|---|---|---|---|---|---|---|
| SVM | 0.777 | 0.353 | 0.442 | 0.828 | 0.872 | 0.208 | 0.679 | 0.009 | 0.000 |
| Random Forest | 0.860 | 0.393 | 0.491 | 0.881 | 0.924 | 0.192 | 0.785 | 0.064 | 0.000 |
| 1D-CNN (Adam, default HPs) | 0.727 | 0.342 | 0.426 | 0.800 | 0.839 | 0.085 | 0.779 | 0.002 | 0.004 |
| 1D-CNN (grid-search tuned) | 0.703 | 0.339 | 0.422 | 0.783 | 0.819 | 0.090 | 0.777 | 0.004 | 0.003 |
| 1D-CNN (GWO tuned) | 0.751 | 0.323 | 0.403 | 0.807 | 0.853 | 0.053 | 0.702 | 0.002 | 0.003 |
| 1D-CNN (WOA tuned) | 0.763 | 0.347 | 0.433 | 0.824 | 0.865 | 0.069 | 0.794 | 0.002 | 0.003 |
| CNN-LSTM | 0.834 | 0.335 | 0.419 | 0.856 | 0.907 | 0.049 | 0.714 | 0.005 | 0.000 |

CNN-LSTM is scored on 49,609 beat sequences (the first 4 beats of each test record have no full 5-beat history); all other models on 49,697 beats.
