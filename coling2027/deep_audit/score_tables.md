# Saved-score deep audit

Post-hoc descriptive analysis, not new defense validation. Means over seeds 42–44.

| Setting | Score | AUROC | F1@5% | Recall@5% | Poisons left | Budget for 95% recall (oracle diagnostic) |
|---|---|---:|---:|---:|---:|---:|
| imdb N=10000 p=0.005 full | PD_KL | 0.929 | 0.120 | 66.0% | 17.0 | 32.3% |
| imdb N=10000 p=0.005 full | PD_observed | 0.868 | 0.081 | 44.7% | 27.7 | 51.7% |
| imdb N=10000 p=0.005 full | base_label_NLL | 0.928 | 0.119 | 65.3% | 17.3 | 32.5% |
| imdb N=10000 p=0.005 full | adapted_confidence_posthoc | 0.308 | 0.000 | 0.0% | 50.0 | 92.4% |
| imdb N=10000 p=0.01 full | PD_KL | 0.926 | 0.216 | 64.7% | 35.3 | 32.5% |
| imdb N=10000 p=0.01 full | PD_observed | 0.972 | 0.281 | 84.3% | 15.7 | 11.4% |
| imdb N=10000 p=0.01 full | base_label_NLL | 0.926 | 0.214 | 64.3% | 35.7 | 32.6% |
| imdb N=10000 p=0.01 full | adapted_confidence_posthoc | 0.747 | 0.043 | 13.0% | 87.0 | 57.5% |
| imdb N=10000 p=0.05 full | PD_KL | 0.929 | 0.535 | 53.5% | 232.7 | 35.8% |
| imdb N=10000 p=0.05 full | PD_observed | 0.998 | 0.930 | 93.0% | 35.0 | 5.3% |
| imdb N=10000 p=0.05 full | base_label_NLL | 0.928 | 0.529 | 52.9% | 235.3 | 36.1% |
| imdb N=10000 p=0.05 full | adapted_confidence_posthoc | 0.992 | 0.819 | 81.9% | 90.7 | 8.2% |
| imdb N=10000 p=0.05 lora | PD_KL | 0.983 | 0.861 | 86.1% | 69.3 | 12.7% |
| imdb N=10000 p=0.05 lora | PD_observed | 0.997 | 0.945 | 94.5% | 27.7 | 5.5% |
| imdb N=10000 p=0.05 lora | base_label_NLL | 0.928 | 0.529 | 52.9% | 235.3 | 36.1% |
| imdb N=10000 p=0.05 lora | adapted_confidence_posthoc | 0.688 | 0.011 | 1.1% | 494.7 | 50.2% |
| imdb N=1000 p=0.01 full | PD_KL | 0.867 | 0.233 | 70.0% | 3.0 | 81.7% |
| imdb N=1000 p=0.01 full | PD_observed | 0.785 | 0.167 | 50.0% | 5.0 | 96.5% |
| imdb N=1000 p=0.01 full | base_label_NLL | 0.953 | 0.256 | 76.7% | 2.3 | 19.6% |
| imdb N=1000 p=0.01 full | adapted_confidence_posthoc | 0.439 | 0.000 | 0.0% | 10.0 | 99.9% |
| imdb N=1000 p=0.05 full | PD_KL | 0.931 | 0.507 | 50.7% | 24.7 | 32.1% |
| imdb N=1000 p=0.05 full | PD_observed | 0.722 | 0.227 | 22.7% | 38.7 | 77.4% |
| imdb N=1000 p=0.05 full | base_label_NLL | 0.942 | 0.560 | 56.0% | 22.0 | 30.8% |
| imdb N=1000 p=0.05 full | adapted_confidence_posthoc | 0.337 | 0.000 | 0.0% | 50.0 | 99.5% |
| sst2 N=6000 p=0.01 full | PD_KL | 0.929 | 0.191 | 57.2% | 25.7 | 27.3% |
| sst2 N=6000 p=0.01 full | PD_observed | 0.568 | 0.035 | 10.6% | 53.7 | 90.0% |
| sst2 N=6000 p=0.01 full | base_label_NLL | 0.931 | 0.193 | 57.8% | 25.3 | 24.8% |
| sst2 N=6000 p=0.01 full | adapted_confidence_posthoc | 0.220 | 0.000 | 0.0% | 60.0 | 99.5% |
| sst2 N=6000 p=0.05 full | PD_KL | 0.922 | 0.517 | 51.7% | 145.0 | 37.5% |
| sst2 N=6000 p=0.05 full | PD_observed | 0.784 | 0.269 | 26.9% | 219.3 | 71.0% |
| sst2 N=6000 p=0.05 full | base_label_NLL | 0.921 | 0.518 | 51.8% | 144.7 | 38.4% |
| sst2 N=6000 p=0.05 full | adapted_confidence_posthoc | 0.314 | 0.000 | 0.0% | 300.0 | 95.6% |
| sst2 N=6000 p=0.05 lora | PD_KL | 0.876 | 0.449 | 44.9% | 165.3 | 58.3% |
| sst2 N=6000 p=0.05 lora | PD_observed | 0.755 | 0.289 | 28.9% | 213.3 | 71.8% |
| sst2 N=6000 p=0.05 lora | base_label_NLL | 0.921 | 0.518 | 51.8% | 144.7 | 38.4% |
| sst2 N=6000 p=0.05 lora | adapted_confidence_posthoc | 0.180 | 0.000 | 0.0% | 300.0 | 98.0% |

The 95%-recall budget uses poison annotations retrospectively; it is not a deployable threshold.
Changing removal budgets has not been retrained here; the table predicts residual poison counts, not ASR.
