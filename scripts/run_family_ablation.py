#!/usr/bin/env python
"""Feature-family ablation + threshold sensitivity for the Deltx AI-code detector.

Place in ``scripts/`` and run from the repo root:

    python scripts/run_family_ablation.py

Design decisions baked in (see docs/detection/ablation.md):

* **Corpus**: droidcollection only. Pooling corpora confounds feature importance
  with corpus identity (f7_avg_identifier_length is a CodeNet marker, not an AI
  marker), so the ablation runs on a single corpus.
* **Fixed hyperparameters** across every arm, taken from the tuned full-feature
  droid-only run. Re-tuning each arm would mix the feature effect with
  RandomizedSearchCV variance and make the deltas uninterpretable.
* **5 seeds** per arm. Each seed re-draws the class rebalance and the
  train/val/test split, so the spread captures split variance, not just model
  init noise.
* **Threshold** selected on the validation split at a fixed false-positive rate
  (default 5%), never on test. Falsely flagging human code is the costly error.
* Families are unequal (6/6/4). Reported raw; noted in the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PERPLEXITY = [
    "f1_mean_surprisal",
    "f2_surprisal_variance",
    "f3_sequence_perplexity",
    "f4_max_surprisal",
    "f5_low_surprisal_ratio",
    "f6_surprisal_slope",
]
STYLOMETRIC = [
    "f7_avg_identifier_length",
    "f8_identifier_diversity",
    "f9_whitespace_consistency",
    "f10_comment_to_code_ratio",
    "f11_ast_depth_mean",
    "f12_ast_node_type_diversity",
]
DISTRIBUTION = [
    "f13_shannon_entropy",
    "f14_zipf_coefficient_deviation",
    "f15_bigram_repetition_rate",
    "f16_hapax_legomena_ratio",
]

FAMILIES = {
    "perplexity": PERPLEXITY,
    "stylometric": STYLOMETRIC,
    "distribution": DISTRIBUTION,
}
ALL_FEATURES = PERPLEXITY + STYLOMETRIC + DISTRIBUTION

# Tuned on the full-feature droidcollection-only run
# (data/runs/2026-07-19T15-43-36Z_droid-only-*/manifest.json).
FIXED_PARAMS = {
    "subsample": 0.9,
    "n_estimators": 300,
    "min_child_weight": 1,
    "max_depth": 7,
    "learning_rate": 0.05,
    "colsample_bytree": 0.7,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}

SEEDS = [42, 43, 44, 45, 46]
TEST_FRACTION = 0.20
VAL_FRACTION = 0.10
TARGET_FPR = 0.05


def build_arms() -> dict[str, list[str]]:
    """Return the ablation arms as ``{arm_name: feature_columns}``."""
    arms: dict[str, list[str]] = {"full_16": list(ALL_FEATURES)}
    for name, cols in FAMILIES.items():
        arms[f"drop_{name}"] = [c for c in ALL_FEATURES if c not in cols]
        arms[f"only_{name}"] = list(cols)
    return arms


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_frame(path: Path, sources: list[str]) -> pd.DataFrame:
    """Load the Phase B feature parquet, restricted to ``sources``."""
    if not path.is_file():
        raise SystemExit(
            f"Feature file not found: {path}\n"
            "Run Phase B (feature extraction) first, or pass --features."
        )
    frame = pd.read_parquet(path)
    missing = [c for c in (*ALL_FEATURES, "label") if c not in frame.columns]
    if missing:
        raise SystemExit(f"Feature file missing required columns: {missing}")
    if sources and "source_dataset" in frame.columns:
        frame = frame[frame["source_dataset"].isin(sources)]
    if frame.empty:
        raise SystemExit(f"No rows left after filtering to sources={sources}")
    return frame.reset_index(drop=True)


def rebalance(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Downsample the majority class to match the minority, seeded."""
    target = int(frame["label"].value_counts().min())
    parts = [
        group.sample(n=target, random_state=seed)
        for _, group in frame.groupby("label", sort=True)
    ]
    return pd.concat(parts).reset_index(drop=True)


def split(frame: pd.DataFrame, seed: int):
    """Stratified train / validation / test split."""
    train_all, test = train_test_split(
        frame,
        test_size=TEST_FRACTION,
        stratify=frame["label"],
        random_state=seed,
        shuffle=True,
    )
    train, val = train_test_split(
        train_all,
        test_size=VAL_FRACTION,
        stratify=train_all["label"],
        random_state=seed,
        shuffle=True,
    )
    return train, val, test


# --------------------------------------------------------------------------- #
# Threshold selection
# --------------------------------------------------------------------------- #


def threshold_at_fpr(y_true: np.ndarray, proba: np.ndarray, target_fpr: float) -> float:
    """Lowest threshold whose false-positive rate on ``y_true`` is <= target.

    Chosen on validation data only. A low FPR is the operating point that
    matters here: flagging a human-authored file as AI is the expensive error.
    """
    negatives = proba[y_true == 0]
    if negatives.size == 0:
        return 0.5
    # The (1 - target_fpr) quantile of negative scores is the smallest cut that
    # leaves at most target_fpr of negatives above it.
    return float(np.quantile(negatives, 1.0 - target_fpr))


def metrics_at(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Threshold-dependent metrics plus the achieved false-positive rate."""
    pred = (proba >= threshold).astype(int)
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    tp = int(((y_true == 1) & (pred == 1)).sum())
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# --------------------------------------------------------------------------- #
# Core loop
# --------------------------------------------------------------------------- #


def fit_and_score(train, val, test, columns: list[str], seed: int):
    """Fit one arm and return (test_proba, val_proba, y_test, y_val)."""
    model = XGBClassifier(**FIXED_PARAMS, random_state=seed)
    model.fit(train[columns].to_numpy(float), train["label"].to_numpy(int))
    return (
        model.predict_proba(test[columns].to_numpy(float))[:, 1],
        model.predict_proba(val[columns].to_numpy(float))[:, 1],
        test["label"].to_numpy(int),
        val["label"].to_numpy(int),
    )


def run(args: argparse.Namespace) -> None:
    frame = load_frame(args.features, args.sources)
    arms = build_arms()
    rows: list[dict] = []
    sweep_rows: list[dict] = []

    print(f"Corpus: {args.sources or 'all'} — {len(frame)} rows before rebalance")
    print(f"{len(arms)} arms x {len(SEEDS)} seeds = {len(arms) * len(SEEDS)} fits\n")

    for seed in SEEDS:
        balanced = rebalance(frame, seed)
        train, val, test = split(balanced, seed)
        for arm, columns in arms.items():
            proba_test, proba_val, y_test, y_val = fit_and_score(
                train, val, test, columns, seed
            )
            thr = threshold_at_fpr(y_val, proba_val, args.target_fpr)
            row = {
                "arm": arm,
                "seed": seed,
                "n_features": len(columns),
                "n_train": len(train),
                "n_test": len(test),
                "auroc": roc_auc_score(y_test, proba_test),
                "auprc": average_precision_score(y_test, proba_test),
            }
            for key, value in metrics_at(y_test, proba_test, 0.5).items():
                row[f"t50_{key}"] = value
            for key, value in metrics_at(y_test, proba_test, thr).items():
                row[f"tfpr_{key}"] = value
            rows.append(row)
            print(
                f"  seed={seed} {arm:<20s} "
                f"AUROC={row['auroc']:.4f}  "
                f"F1@0.5={row['t50_f1']:.4f}  "
                f"thr@{args.target_fpr:.0%}FPR={thr:.3f} "
                f"(P={row['tfpr_precision']:.3f} R={row['tfpr_recall']:.3f})"
            )

            # Full sweep, full-feature arm only — this is the sensitivity curve.
            if arm == "full_16":
                for t in np.round(np.arange(0.05, 1.0, 0.05), 2):
                    sweep = metrics_at(y_test, proba_test, float(t))
                    sweep.update({"seed": seed})
                    sweep_rows.append(sweep)
        print()

    results = pd.DataFrame(rows)
    sweep = pd.DataFrame(sweep_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "ablation_raw.csv", index=False)
    sweep.to_csv(args.out / "threshold_sweep_raw.csv", index=False)

    summary = summarise(results)
    summary.to_csv(args.out / "ablation_summary.csv", index=False)
    sweep_summary = (
        sweep.groupby("threshold")[
            ["accuracy", "precision", "recall", "f1", "fpr"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    sweep_summary.to_csv(args.out / "threshold_sweep_summary.csv")

    report = render_markdown(summary, sweep_summary, args)
    (args.out / "ablation_report.md").write_text(report, encoding="utf-8")
    (args.out / "config.json").write_text(
        json.dumps(
            {
                "sources": args.sources,
                "seeds": SEEDS,
                "fixed_params": FIXED_PARAMS,
                "target_fpr": args.target_fpr,
                "families": {k: len(v) for k, v in FAMILIES.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report)
    print(f"\nWrote 6 files to {args.out}")


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- 95% CI per arm, with delta-AUROC against the full-feature arm."""
    n = results["seed"].nunique()
    agg = (
        results.groupby("arm")
        .agg(
            n_features=("n_features", "first"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            f1_50_mean=("t50_f1", "mean"),
            prec_fpr_mean=("tfpr_precision", "mean"),
            rec_fpr_mean=("tfpr_recall", "mean"),
            thr_mean=("tfpr_threshold", "mean"),
        )
        .reset_index()
    )
    # Paired delta: same seed, arm vs full_16, so split variance cancels.
    base = results[results["arm"] == "full_16"].set_index("seed")["auroc"]
    deltas = results.assign(delta=lambda d: d["auroc"] - d["seed"].map(base))
    dagg = (
        deltas.groupby("arm")["delta"].agg(["mean", "std"]).reset_index()
    )
    dagg["ci95"] = 1.96 * dagg["std"] / np.sqrt(n)
    agg = agg.merge(
        dagg.rename(columns={"mean": "delta_auroc", "std": "delta_std"}), on="arm"
    )
    order = [
        "full_16",
        "drop_perplexity",
        "drop_stylometric",
        "drop_distribution",
        "only_perplexity",
        "only_stylometric",
        "only_distribution",
    ]
    agg["_o"] = agg["arm"].map({a: i for i, a in enumerate(order)})
    return agg.sort_values("_o").drop(columns="_o").round(4)


def render_markdown(summary, sweep_summary, args) -> str:
    """Paper-ready markdown tables."""
    lines = [
        "# Feature-family ablation — Deltx AI-code detector",
        "",
        f"Corpus: `{', '.join(args.sources) if args.sources else 'all'}`. "
        f"{len(SEEDS)} seeds, fixed hyperparameters across all arms.",
        "Families are unequal in size (perplexity 6, stylometric 6, "
        "distribution 4); deltas are reported raw.",
        "",
        "## Table 1 — Ablation arms",
        "",
        "| Arm | #feat | AUROC | AUPRC | ΔAUROC vs full | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        delta = "—" if r["arm"] == "full_16" else f"{r['delta_auroc']:+.4f}"
        ci = "—" if r["arm"] == "full_16" else f"±{r['ci95']:.4f}"
        lines.append(
            f"| `{r['arm']}` | {int(r['n_features'])} | "
            f"{r['auroc_mean']:.4f} ± {r['auroc_std']:.4f} | "
            f"{r['auprc_mean']:.4f} | {delta} | {ci} |"
        )
    lines += [
        "",
        "Read `drop_X` as the cost of removing family X (larger negative Δ = more",
        "necessary). Read `only_X` as how far family X gets on its own",
        "(less negative = more sufficient). A family can be sufficient but not",
        "necessary when families are redundant — compare the two columns.",
        "",
        f"## Table 2 — Operating point at {args.target_fpr:.0%} FPR "
        "(threshold chosen on validation)",
        "",
        "| Arm | threshold | precision | recall |",
        "|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| `{r['arm']}` | {r['thr_mean']:.3f} | "
            f"{r['prec_fpr_mean']:.4f} | {r['rec_fpr_mean']:.4f} |"
        )
    lines += [
        "",
        "## Table 3 — Threshold sensitivity (full 16-feature model)",
        "",
        "| threshold | accuracy | precision | recall | F1 | FPR |",
        "|---|---|---|---|---|---|",
    ]
    for thr, r in sweep_summary.iterrows():
        lines.append(
            f"| {thr:.2f} | {r[('accuracy', 'mean')]:.4f} | "
            f"{r[('precision', 'mean')]:.4f} | {r[('recall', 'mean')]:.4f} | "
            f"{r[('f1', 'mean')]:.4f} | {r[('fpr', 'mean')]:.4f} |"
        )
    lines += [
        "",
        "The default 0.5 is a reporting convention, not a tuned choice. Deltx",
        "consumes `ai_confidence` as a continuous signal downstream, so the",
        "threshold only affects the reported confusion matrix.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/train_features.parquet"),
        help="Phase B feature parquet.",
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=["droidcollection"],
        help="source_dataset values to keep. Default: droidcollection only.",
    )
    p.add_argument(
        "--target-fpr",
        type=float,
        default=TARGET_FPR,
        help="False-positive rate the validation-selected threshold targets.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/runs/ablation_families"),
        help="Output directory for CSVs and the markdown report.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())