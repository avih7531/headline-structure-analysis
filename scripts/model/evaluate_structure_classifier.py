"""
Evaluate the headline structure classifier against a labeled gold set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, List

import pandas as pd

from headline_structure_classifier import LABELS, classify_dataframe


def _safe_div(numerator: float, denominator: float) -> float:
    """Safely divide with zero-protection for metric calculations."""
    return numerator / denominator if denominator else 0.0


def _compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    """Compute aggregate + per-label classification metrics."""
    per_label = {}
    total_tp = total_fp = total_fn = 0

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_precision = sum(v["precision"] for v in per_label.values()) / len(labels)
    macro_recall = sum(v["recall"] for v in per_label.values()) / len(labels)
    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(labels)
    accuracy = _safe_div(sum(1 for t, p in zip(y_true, y_pred) if t == p), len(y_true))

    micro_precision = _safe_div(total_tp, total_tp + total_fp)
    micro_recall = _safe_div(total_tp, total_tp + total_fn)
    micro_f1 = _safe_div(
        2 * micro_precision * micro_recall, micro_precision + micro_recall
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_label": per_label,
    }


def _build_confusion(y_true: List[str], y_pred: List[str], labels: List[str]) -> pd.DataFrame:
    """Build a label-by-label confusion matrix dataframe."""
    matrix = {(true, pred): 0 for true in labels for pred in labels}
    for true, pred in zip(y_true, y_pred):
        if true in labels and pred in labels:
            matrix[(true, pred)] += 1

    rows = []
    for true in labels:
        row = {"gold_label": true}
        for pred in labels:
            row[pred] = matrix[(true, pred)]
        rows.append(row)
    return pd.DataFrame(rows)


def _majority_baseline(train_labels: List[str], n: int) -> List[str]:
    """Generate majority-class baseline predictions for n examples."""
    majority = Counter(train_labels).most_common(1)[0][0]
    return [majority] * n


def _random_baseline(train_labels: List[str], n: int, seed: int) -> List[str]:
    """Generate class-frequency-weighted random baseline predictions."""
    rng = random.Random(seed)
    dist = Counter(train_labels)
    labels = list(dist.keys())
    weights = [dist[l] for l in labels]
    return rng.choices(labels, weights=weights, k=n)


def _domain_comparison(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Compare predicted structure rates between domestic and world subsets."""
    if "category" not in pred_df.columns:
        return pd.DataFrame()

    results = []
    categories = sorted(pred_df["category"].astype(str).str.lower().unique().tolist())
    if "domestic" not in categories or "world" not in categories:
        return pd.DataFrame()

    domestic = pred_df[pred_df["category"].str.lower() == "domestic"]
    world = pred_df[pred_df["category"].str.lower() == "world"]
    if domestic.empty or world.empty:
        return pd.DataFrame()

    domestic_counts = Counter(domestic["predicted_structure"])
    world_counts = Counter(world["predicted_structure"])
    domestic_n = len(domestic)
    world_n = len(world)

    for label in LABELS:
        d_pct = 100.0 * domestic_counts.get(label, 0) / domestic_n
        w_pct = 100.0 * world_counts.get(label, 0) / world_n
        results.append(
            {
                "label": label,
                "domestic_pct": d_pct,
                "world_pct": w_pct,
                "abs_diff_pct_points": abs(d_pct - w_pct),
            }
        )
    return pd.DataFrame(results).sort_values("abs_diff_pct_points", ascending=False)


def _merge_gold_with_parsed(gold_df: pd.DataFrame, parsed_df: pd.DataFrame) -> pd.DataFrame:
    """Join gold labels with parsed features, filtering unmatched rows."""
    needed_cols = ["headline", "gold_label", "split"]
    missing = [c for c in needed_cols if c not in gold_df.columns]
    if missing:
        raise ValueError(f"Gold file missing required columns: {missing}")

    labeled = gold_df[gold_df["gold_label"].astype(str).str.strip() != ""].copy()
    if labeled.empty:
        raise ValueError("No labeled rows in gold file.")

    merged = labeled.merge(parsed_df, on="headline", how="left", suffixes=("", "_parsed"))
    merged = merged[merged["tokens"].notna()].copy()
    if merged.empty:
        raise ValueError("No labeled headlines could be matched to parsed data.")
    return merged


def _evaluate_subset(subset_df: pd.DataFrame, train_labels: List[str], seed: int) -> Dict:
    """Evaluate rule-based model and baselines on one split subset."""
    y_true = subset_df["gold_label"].tolist()
    y_rule = subset_df["predicted_structure"].tolist()
    y_majority = _majority_baseline(train_labels, len(subset_df))
    y_random = _random_baseline(train_labels, len(subset_df), seed=seed)

    return {
        "n": len(subset_df),
        "rule_based": _compute_metrics(y_true, y_rule, LABELS),
        "majority_baseline": _compute_metrics(y_true, y_majority, LABELS),
        "random_baseline": _compute_metrics(y_true, y_random, LABELS),
    }


def main() -> None:
    """CLI entrypoint for full structure-model evaluation outputs."""
    parser = argparse.ArgumentParser(description="Evaluate rule-based structure classifier.")
    parser.add_argument("--parsed-input", default="data/headlines_parsed.json")
    parser.add_argument("--gold-input", default="data/gold_headlines_annotation_split.csv")
    parser.add_argument("--output-dir", default="data/evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fallback-to-suggested",
        action="store_true",
        help="Use suggested_label when gold_label is empty.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    parsed_df = pd.read_json(args.parsed_input)
    gold_df = pd.read_csv(args.gold_input).fillna("")
    if args.fallback_to_suggested and "suggested_label" in gold_df.columns:
        empty_mask = gold_df["gold_label"].str.strip() == ""
        gold_df.loc[empty_mask, "gold_label"] = gold_df.loc[empty_mask, "suggested_label"]

    merged = _merge_gold_with_parsed(gold_df, parsed_df)
    merged = classify_dataframe(merged)

    if "split" not in merged.columns or (merged["split"].str.strip() == "").all():
        raise ValueError(
            "Gold dataset must include train/dev/test in split column. "
            "Run split_gold_dataset.py first."
        )

    train_df = merged[merged["split"] == "train"].copy()
    dev_df = merged[merged["split"] == "dev"].copy()
    test_df = merged[merged["split"] == "test"].copy()
    if train_df.empty or dev_df.empty or test_df.empty:
        raise ValueError("Need non-empty train/dev/test splits for evaluation.")

    train_labels = train_df["gold_label"].tolist()
    eval_report = {
        "train_distribution": dict(Counter(train_labels)),
        "dev": _evaluate_subset(dev_df, train_labels, seed=args.seed),
        "test": _evaluate_subset(test_df, train_labels, seed=args.seed + 1),
    }

    eval_path = os.path.join(args.output_dir, "classifier_eval.json")
    with open(eval_path, "w", encoding="utf-8") as handle:
        json.dump(eval_report, handle, indent=2)

    pred_path = os.path.join(args.output_dir, "gold_predictions.csv")
    merged[
        [
            "headline",
            "category",
            "gold_label",
            "split",
            "predicted_structure",
            "matched_rules",
        ]
    ].to_csv(pred_path, index=False)

    conf_df = _build_confusion(
        test_df["gold_label"].tolist(), test_df["predicted_structure"].tolist(), LABELS
    )
    conf_path = os.path.join(args.output_dir, "confusion_matrix_test.csv")
    conf_df.to_csv(conf_path, index=False)

    full_pred = classify_dataframe(parsed_df)
    domain_df = _domain_comparison(full_pred)
    if not domain_df.empty:
        domain_path = os.path.join(args.output_dir, "domain_structure_comparison.csv")
        domain_df.to_csv(domain_path, index=False)

    print(f"[save] evaluation report: {eval_path}")
    print(f"[save] gold predictions: {pred_path}")
    print(f"[save] test confusion matrix: {conf_path}")
    if not domain_df.empty:
        print("[save] domain comparison: data/evaluation/domain_structure_comparison.csv")
        top = domain_df.head(3)[["label", "abs_diff_pct_points"]].values.tolist()
        print("[summary] top domestic-vs-world structural gaps (pp):", top)

    test_macro = eval_report["test"]["rule_based"]["macro_f1"]
    dev_macro = eval_report["dev"]["rule_based"]["macro_f1"]
    test_acc = eval_report["test"]["rule_based"]["accuracy"]
    print(f"[summary] rule-based dev macro F1={dev_macro:.3f} | test macro F1={test_macro:.3f} | test acc={test_acc:.3f}")


if __name__ == "__main__":
    main()
