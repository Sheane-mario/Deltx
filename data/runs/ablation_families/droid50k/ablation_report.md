# Feature-family ablation — Deltx AI-code detector

Corpus: `droidcollection`. 5 seeds, fixed hyperparameters across all arms.
Families are unequal in size (perplexity 6, stylometric 6, distribution 4); deltas are reported raw.

## Table 1 — Ablation arms

| Arm | #feat | AUROC | AUPRC | ΔAUROC vs full | 95% CI |
|---|---|---|---|---|---|
| `full_16` | 16 | 0.9539 ± 0.0019 | 0.9540 | — | — |
| `drop_perplexity` | 10 | 0.9255 ± 0.0020 | 0.9183 | -0.0285 | ±0.0005 |
| `drop_stylometric` | 10 | 0.8105 ± 0.0028 | 0.8309 | -0.1434 | ±0.0009 |
| `drop_distribution` | 12 | 0.9470 ± 0.0016 | 0.9467 | -0.0070 | ±0.0005 |
| `only_perplexity` | 6 | 0.7705 ± 0.0034 | 0.7908 | -0.1834 | ±0.0020 |
| `only_stylometric` | 6 | 0.9059 ± 0.0025 | 0.8931 | -0.0480 | ±0.0007 |
| `only_distribution` | 4 | 0.6654 ± 0.0009 | 0.6828 | -0.2885 | ±0.0017 |

Read `drop_X` as the cost of removing family X (larger negative Δ = more
necessary). Read `only_X` as how far family X gets on its own
(less negative = more sufficient). A family can be sufficient but not
necessary when families are redundant — compare the two columns.

## Table 2 — Operating point at 5% FPR (threshold chosen on validation)

| Arm | threshold | precision | recall |
|---|---|---|---|
| `full_16` | 0.778 | 0.9392 | 0.7664 |
| `drop_perplexity` | 0.825 | 0.9243 | 0.6308 |
| `drop_stylometric` | 0.726 | 0.8930 | 0.4299 |
| `drop_distribution` | 0.791 | 0.9350 | 0.7351 |
| `only_perplexity` | 0.742 | 0.8714 | 0.3606 |
| `only_stylometric` | 0.834 | 0.9130 | 0.5401 |
| `only_distribution` | 0.729 | 0.8149 | 0.2223 |

## Table 3 — Threshold sensitivity (full 16-feature model)

| threshold | accuracy | precision | recall | F1 | FPR |
|---|---|---|---|---|---|
| 0.05 | 0.7493 | 0.6682 | 0.9907 | 0.7981 | 0.4920 |
| 0.10 | 0.8061 | 0.7269 | 0.9807 | 0.8350 | 0.3685 |
| 0.15 | 0.8349 | 0.7631 | 0.9714 | 0.8547 | 0.3016 |
| 0.20 | 0.8533 | 0.7899 | 0.9627 | 0.8678 | 0.2561 |
| 0.25 | 0.8651 | 0.8110 | 0.9521 | 0.8759 | 0.2219 |
| 0.30 | 0.8726 | 0.8273 | 0.9418 | 0.8808 | 0.1966 |
| 0.35 | 0.8784 | 0.8430 | 0.9300 | 0.8844 | 0.1731 |
| 0.40 | 0.8825 | 0.8572 | 0.9179 | 0.8865 | 0.1529 |
| 0.45 | 0.8843 | 0.8691 | 0.9049 | 0.8866 | 0.1363 |
| 0.50 | 0.8854 | 0.8813 | 0.8908 | 0.8860 | 0.1199 |
| 0.55 | 0.8849 | 0.8920 | 0.8758 | 0.8838 | 0.1060 |
| 0.60 | 0.8830 | 0.9026 | 0.8586 | 0.8800 | 0.0927 |
| 0.65 | 0.8794 | 0.9130 | 0.8387 | 0.8743 | 0.0799 |
| 0.70 | 0.8741 | 0.9235 | 0.8157 | 0.8663 | 0.0675 |
| 0.75 | 0.8651 | 0.9331 | 0.7867 | 0.8537 | 0.0564 |
| 0.80 | 0.8523 | 0.9435 | 0.7494 | 0.8353 | 0.0449 |
| 0.85 | 0.8317 | 0.9538 | 0.6972 | 0.8055 | 0.0338 |
| 0.90 | 0.7972 | 0.9666 | 0.6157 | 0.7522 | 0.0213 |
| 0.95 | 0.7291 | 0.9813 | 0.4671 | 0.6329 | 0.0089 |

The default 0.5 is a reporting convention, not a tuned choice. Deltx
consumes `ai_confidence` as a continuous signal downstream, so the
threshold only affects the reported confusion matrix.
