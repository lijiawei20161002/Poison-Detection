# Sentiment removal controls — 26 September 2026

Completed **13/13** planned runs, including 12 new ablations and 1 reproduction control(s).

This follow-up tests whether base-model disagreement or adapted-model confidence alone explains the earlier remediation results. It was motivated by the completed study; its directions, seeds, and budgets were frozen before these new training runs.

All removal arms discard exactly 5% of the same saved poisoned pools and retrain FLAN-T5-small from the same pinned base with the original ten-epoch protocol. Seeds are 42, 43, and 44. The two new scores are base-label NLL (larger is suspicious) and negative adapted-label NLL (higher confidence is suspicious). Neither uses poison labels to rank. Confidence remains a post-hoc ablation; no score direction is chosen per dataset.

## Matched results

Mean ± sample standard deviation across available seeds. ASR and clean accuracy are percentages. The original rows are unchanged reference results, not new reruns. Clean-trained controls use unmodified data; no-removal and clean-trained models have more optimizer steps than filtered models.

| Dataset | Method | Record | Seeds | Poisons removed | Clean removed | ASR (%) | Clean accuracy (%) |
|---|---|---|---:|---:|---:|---:|---:|
| imdb | clean_trained | original | 3 | — | — | 13.8 ± 0.7 | 85.5 ± 0.5 |
| imdb | no_removal | original | 3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 85.2 ± 0.5 |
| imdb | PD_KL | original | 3 | 267.3 ± 9.0 | 232.7 ± 9.0 | 98.7 ± 2.3 | 85.7 ± 0.5 |
| imdb | base_label_NLL | new | 3 | 264.7 ± 10.5 | 235.3 ± 10.5 | 97.8 ± 1.9 | 86.2 ± 1.2 |
| imdb | PD_observed | original | 3 | 465.0 ± 10.4 | 35.0 ± 10.4 | 17.0 ± 0.1 | 85.5 ± 0.8 |
| imdb | adapted_label_confidence_posthoc | new | 3 | 409.3 ± 54.4 | 90.7 ± 54.4 | 76.8 ± 40.3 | 85.0 ± 0.2 |
| imdb | ZScore_released_unigram | original | 3 | 0.0 ± 0.0 | 500.0 ± 0.0 | 100.0 ± 0.0 | 85.5 ± 0.5 |
| imdb | random | original | 3 | 22.7 ± 7.6 | 477.3 ± 7.6 | 100.0 ± 0.0 | 85.6 ± 1.0 |
| imdb | oracle | original | 3 | 500.0 ± 0.0 | 0.0 ± 0.0 | 16.0 ± 1.3 | 85.4 ± 0.3 |
| sst2 | clean_trained | original | 3 | — | — | 23.8 ± 2.2 | 90.7 ± 0.3 |
| sst2 | no_removal | original | 3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 91.0 ± 1.1 | 89.7 ± 0.2 |
| sst2 | PD_KL | original | 3 | 155.0 ± 2.6 | 145.0 ± 2.6 | 48.9 ± 2.6 | 91.0 ± 0.0 |
| sst2 | base_label_NLL | new | 3 | 155.3 ± 1.5 | 144.7 ± 1.5 | 48.4 ± 1.0 | 91.0 ± 0.1 |
| sst2 | PD_observed | original | 3 | 80.7 ± 20.6 | 219.3 ± 20.6 | 76.2 ± 9.1 | 90.3 ± 0.8 |
| sst2 | adapted_label_confidence_posthoc | new | 3 | 0.0 ± 0.0 | 300.0 ± 0.0 | 90.9 ± 0.9 | 89.4 ± 0.1 |
| sst2 | ZScore_released_unigram | original | 3 | 0.0 ± 0.0 | 300.0 ± 0.0 | 93.8 ± 0.3 | 89.3 ± 0.5 |
| sst2 | random | original | 3 | 14.0 ± 4.0 | 286.0 ± 4.0 | 92.1 ± 0.5 | 89.2 ± 0.2 |
| sst2 | oracle | original | 3 | 300.0 ± 0.0 | 0.0 ± 0.0 | 25.1 ± 2.0 | 90.6 ± 0.3 |

## Paired differences

New arm minus the indicated original arm, pairing the same seed and training pool. Negative ASR differences favor the new arm; positive accuracy differences favor the new arm. Overlap is the intersection of the two removal sets divided by their common budget. These three-seed summaries are descriptive, not significance claims.

| Dataset | New arm minus reference | Seeds | ASR difference (pp) | Clean accuracy difference (pp) | Removal-set overlap (%) |
|---|---|---:|---:|---:|---:|
| imdb | base_label_NLL − PD_KL | 3 | -0.9 ± 1.9 | 0.5 ± 1.0 | 99.0 ± 0.2 |
| imdb | adapted_label_confidence_posthoc − PD_observed | 3 | 59.8 ± 40.1 | -0.5 ± 0.8 | 77.8 ± 10.9 |
| sst2 | base_label_NLL − PD_KL | 3 | -0.5 ± 1.7 | 0.0 ± 0.1 | 97.3 ± 0.9 |
| sst2 | adapted_label_confidence_posthoc − PD_observed | 3 | 14.7 ± 8.5 | -0.9 ± 0.9 | 8.8 ± 1.8 |

## Reproduction and artifacts

The SST-2 seed-42 PD-KL replay disagrees with the saved original predictions on **0 clean and 0 triggered examples**. The selected training-example IDs match exactly.

`plan.json` records the frozen source hashes, code hashes, score directions, and jobs. `results/` contains selections, prediction IDs, training histories, manifests, and metrics. Local checkpoints are saved under `../checkpoints/removal_controls/`. `per_seed.csv`, `summary.json`, and `verification.json` provide the complete numerical records.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python coling2027/removal_controls.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python coling2027/summarize_removal_controls.py --require-complete
```

Add `--replay-checkpoints` to reload each saved model on GPU and check every clean/triggered prediction.

This extension covers full fine-tuning on the two existing sentiment settings. It does not validate the GSM8K fixes, separate dataset from attack-family effects, or establish generative defense.
