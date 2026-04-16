"""
Evaluate classifier on currently manually-labeled rows only.
Useful for incremental, honest progress tracking before all 200 are done.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import pandas as pd

from evaluate_structure_classifier import _build_confusion, _compute_metrics
from headline_structure_classifier import LABELS, classify_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on currently labeled subset.")
    parser.add_argument("--parsed-input", default="data/headlines_parsed.json")
    parser.add_argument("--gold-input", default="data/gold_headlines_annotation.csv")
    parser.add_argument("--output-dir", default="data/evaluation_partial")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    parsed_df = pd.read_json(args.parsed_input)
    gold_df = pd.read_csv(args.gold_input).fillna("")
    labeled = gold_df[gold_df["gold_label"].str.strip() != ""].copy()
    if labeled.empty:
        raise ValueError("No manual gold labels found yet.")

    merged = labeled.merge(parsed_df, on="headline", how="left")
    merged = merged[merged["tokens"].notna()].copy()
    if merged.empty:
        raise ValueError("No labeled rows matched parsed dataset.")

    pred_df = classify_dataframe(merged)
    y_true = pred_df["gold_label"].tolist()
    y_pred = pred_df["predicted_structure"].tolist()

    metrics = _compute_metrics(y_true, y_pred, LABELS)
    report = {
        "n_labeled": len(pred_df),
        "label_distribution": dict(Counter(y_true)),
        "metrics": metrics,
    }

    with open(os.path.join(args.output_dir, "partial_eval.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    conf = _build_confusion(y_true, y_pred, LABELS)
    conf.to_csv(os.path.join(args.output_dir, "partial_confusion_matrix.csv"), index=False)

    print(f"Labeled rows evaluated: {len(pred_df)}")
    print(f"Partial macro F1: {metrics['macro_f1']:.3f}")
    print(f"Partial accuracy: {metrics['accuracy']:.3f}")
    print("Saved: data/evaluation_partial/partial_eval.json")


if __name__ == "__main__":
    main()
