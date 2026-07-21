# Feature-family ablation — Deltx AI-code detector

Corpus: `droidcollection`. 5 seeds, fixed hyperparameters across all arms.
Families are unequal in size (perplexity 6, stylometric 6, distribution 4); deltas are reported raw.

## Table 1 — Ablation arms

| Arm | #feat | AUROC | AUPRC | ΔAUROC vs full | 95% CI |
|---|---|---|---|---|---|
| `full_16` | 16 | 0.9414 ± 0.0036 | 0.9397 | — | — |
| `drop_perplexity` | 10 | 0.9060 ± 0.0069 | 0.8953 | -0.0354 | ±0.0044 |
| `drop_stylometric` | 10 | 0.7868 ± 0.0148 | 0.8107 | -0.1546 | ±0.0105 |
| `drop_distribution` | 12 | 0.9363 ± 0.0030 | 0.9344 | -0.0052 | ±0.0012 |
| `only_perplexity` | 6 | 0.7486 ± 0.0129 | 0.7664 | -0.1929 | ±0.0091 |
| `only_stylometric` | 6 | 0.8920 ± 0.0078 | 0.8752 | -0.0494 | ±0.0051 |
| `only_distribution` | 4 | 0.6520 ± 0.0207 | 0.6703 | -0.2895 | ±0.0169 |

Read `drop_X` as the cost of removing family X (larger negative Δ = more
necessary). Read `only_X` as how far family X gets on its own
(less negative = more sufficient). A family can be sufficient but not
necessary when families are redundant — compare the two columns.

## Table 2 — Operating point at 5% FPR (threshold chosen on validation)

| Arm | threshold | precision | recall |
|---|---|---|---|
| `full_16` | 0.788 | 0.9288 | 0.7451 |
| `drop_perplexity` | 0.866 | 0.9160 | 0.5532 |
| `drop_stylometric` | 0.800 | 0.8900 | 0.3797 |
| `drop_distribution` | 0.825 | 0.9274 | 0.6857 |
| `only_perplexity` | 0.823 | 0.8745 | 0.3016 |
| `only_stylometric` | 0.876 | 0.9048 | 0.4654 |
| `only_distribution` | 0.762 | 0.7917 | 0.2127 |

## Table 3 — Threshold sensitivity (full 16-feature model)

| threshold | accuracy | precision | recall | F1 | FPR |
|---|---|---|---|---|---|
| 0.05 | 0.7620 | 0.6815 | 0.9846 | 0.8054 | 0.4605 |
| 0.10 | 0.8066 | 0.7306 | 0.9719 | 0.8341 | 0.3586 |
| 0.15 | 0.8322 | 0.7651 | 0.9592 | 0.8511 | 0.2949 |
| 0.20 | 0.8431 | 0.7847 | 0.9462 | 0.8578 | 0.2600 |
| 0.25 | 0.8536 | 0.8045 | 0.9349 | 0.8647 | 0.2276 |
| 0.30 | 0.8627 | 0.8234 | 0.9238 | 0.8706 | 0.1984 |
| 0.35 | 0.8678 | 0.8385 | 0.9116 | 0.8734 | 0.1759 |
| 0.40 | 0.8695 | 0.8487 | 0.8995 | 0.8733 | 0.1605 |
| 0.45 | 0.8699 | 0.8579 | 0.8868 | 0.8720 | 0.1470 |
| 0.50 | 0.8693 | 0.8706 | 0.8678 | 0.8691 | 0.1292 |
| 0.55 | 0.8695 | 0.8809 | 0.8546 | 0.8674 | 0.1157 |
| 0.60 | 0.8688 | 0.8921 | 0.8392 | 0.8647 | 0.1016 |
| 0.65 | 0.8639 | 0.9014 | 0.8173 | 0.8572 | 0.0895 |
| 0.70 | 0.8588 | 0.9102 | 0.7962 | 0.8492 | 0.0786 |
| 0.75 | 0.8527 | 0.9222 | 0.7705 | 0.8394 | 0.0651 |
| 0.80 | 0.8405 | 0.9300 | 0.7365 | 0.8218 | 0.0554 |
| 0.85 | 0.8235 | 0.9397 | 0.6914 | 0.7963 | 0.0443 |
| 0.90 | 0.7949 | 0.9493 | 0.6230 | 0.7520 | 0.0332 |
| 0.95 | 0.7419 | 0.9634 | 0.5030 | 0.6605 | 0.0192 |

The default 0.5 is a reporting convention, not a tuned choice. Deltx
consumes `ai_confidence` as a continuous signal downstream, so the
threshold only affects the reported confusion matrix.
