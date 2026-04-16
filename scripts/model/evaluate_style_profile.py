"""
Evaluate multi-dimensional style profile predictions against manual gold labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import pandas as pd

# Ensure project root importability when executed as script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.model.headline_style_profiler import profile_dataframe


STYLE_DIMS = [
    ("lead_frame", "gold_lead_frame", "lead_frame"),
    ("agency_style", "gold_agency_style", "agency_style"),
    ("density_band", "gold_density_band", "density_band"),
    ("rhetorical_mode", "gold_rhetorical_mode", "rhetorical_mode"),
]


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def _metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    accuracy = _safe_div(sum(1 for t, p in zip(y_true, y_pred) if t == p), len(y_true))

    per_label = {}
    f1_sum = 0.0
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in y_true if t == label),
        }
        f1_sum += f1

    macro_f1 = _safe_div(f1_sum, len(labels))
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_label": per_label}


def _eval_dim(df: pd.DataFrame, gold_col: str, pred_col: str) -> Dict:
    sub = df[df[gold_col].astype(str).str.strip() != ""].copy()
    labels = sorted(sub[gold_col].astype(str).unique().tolist())
    y_true = sub[gold_col].astype(str).tolist()
    y_pred = sub[pred_col].astype(str).tolist()
    return _metrics(y_true, y_pred, labels)


def _eval_group(df: pd.DataFrame) -> Dict:
    result = {"n": len(df)}
    for dim_name, gold_col, pred_col in STYLE_DIMS:
        result[dim_name] = _eval_dim(df, gold_col, pred_col)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate style profile dimensions.")
    parser.add_argument("--parsed-input", default="data/headlines_parsed.json")
    parser.add_argument("--gold-style-input", default="data/gold_headlines_style_manual.csv")
    parser.add_argument("--split-input", default="data/gold_headlines_full_manual_split.csv")
    parser.add_argument("--output-dir", default="data/evaluation_style")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    parsed = pd.read_json(args.parsed_input).fillna("")
    profiled = profile_dataframe(parsed)

    gold = pd.read_csv(args.gold_style_input).fillna("")
    split_df = pd.read_csv(args.split_input).fillna("")
    if "split" not in split_df.columns:
        raise ValueError("Split file must include split column.")

    split_lookup = split_df[["headline", "split"]].drop_duplicates("headline")
    merged = gold.merge(
        profiled[
            [
                "headline",
                "lead_frame",
                "agency_style",
                "density_band",
                "rhetorical_mode",
            ]
        ],
        on="headline",
        how="left",
    ).merge(split_lookup, on="headline", how="left")

    merged.to_csv(os.path.join(args.output_dir, "style_predictions_vs_gold.csv"), index=False)

    all_eval = _eval_group(merged)
    dev_eval = _eval_group(merged[merged["split"] == "dev"])
    test_eval = _eval_group(merged[merged["split"] == "test"])

    report = {"all": all_eval, "dev": dev_eval, "test": test_eval}
    out_path = os.path.join(args.output_dir, "style_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved style evaluation: {out_path}")
    print("All-labeled dimension scores:")
    for dim, _, _ in STYLE_DIMS:
        print(
            f"  {dim:16} accuracy={all_eval[dim]['accuracy']:.3f} "
            f"macro_f1={all_eval[dim]['macro_f1']:.3f}"
        )
    print("Held-out test dimension scores:")
    for dim, _, _ in STYLE_DIMS:
        print(
            f"  {dim:16} accuracy={test_eval[dim]['accuracy']:.3f} "
            f"macro_f1={test_eval[dim]['macro_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
