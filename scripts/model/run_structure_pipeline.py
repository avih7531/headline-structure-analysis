"""
Run the full structure-classifier workflow on existing parsed headlines.

This script intentionally does NOT collect or parse new headlines.
It only uses data/headlines_parsed.json that already exists.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _run(cmd: list[str]) -> None:
    """Execute one subprocess command and fail fast on errors."""
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _has_manual_gold_labels(path: str) -> bool:
    """Return True when a gold file exists and has non-empty gold labels."""
    if not os.path.exists(path):
        return False
    df = pd.read_csv(path).fillna("")
    if "gold_label" not in df.columns:
        return False
    return (df["gold_label"].str.strip() != "").any()


def _annotation_rows(path: str) -> int:
    """Count rows in an annotation file, returning 0 when absent."""
    if not os.path.exists(path):
        return 0
    df = pd.read_csv(path)
    return len(df)


def _generate_readme_graphs(
    structure_pred_path: str,
    structure_eval_path: str,
    style_eval_path: str,
    images_dir: str = "images",
) -> None:
    """Generate compact README-ready performance/analysis charts in images/."""
    os.makedirs(images_dir, exist_ok=True)

    # 1) Structure label distribution (from evaluated gold predictions).
    if os.path.exists(structure_pred_path):
        pred_df = pd.read_csv(structure_pred_path).fillna("")
        counts = (
            pred_df["predicted_structure"]
            .value_counts()
            .reindex(
                [
                    "simple_clause",
                    "noun_phrase_fragment",
                    "coordination",
                    "passive_clause",
                    "question_form",
                    "other",
                ],
                fill_value=0,
            )
        )
        plt.figure(figsize=(9, 5))
        bars = plt.bar(counts.index, counts.values, color="#5E81AC")
        plt.title("Structure Label Distribution")
        plt.ylabel("Count")
        plt.xticks(rotation=20, ha="right")
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{int(h)}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        out = os.path.join(images_dir, "structure_label_distribution.png")
        plt.savefig(out, dpi=180)
        plt.close()
        print(f"[save] graph: {out}")

    # 2) Model performance summary (accuracy + macro F1 bars).
    if os.path.exists(structure_eval_path) and os.path.exists(style_eval_path) and os.path.exists(structure_pred_path):
        with open(structure_eval_path, "r", encoding="utf-8") as f:
            structure_eval = json.load(f)
        with open(style_eval_path, "r", encoding="utf-8") as f:
            style_eval = json.load(f)

        pred_df = pd.read_csv(structure_pred_path).fillna("")
        labels = [
            "question_form",
            "passive_clause",
            "coordination",
            "noun_phrase_fragment",
            "simple_clause",
            "other",
        ]
        y_true = pred_df["gold_label"].astype(str).tolist()
        y_pred = pred_df["predicted_structure"].astype(str).tolist()
        structure_all_acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
        f1s = []
        for label in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            f1s.append(f1)
        structure_all_macro = sum(f1s) / len(f1s) if f1s else 0.0

        rows = [
            ("structure_all", structure_all_acc, structure_all_macro),
            (
                "structure_test",
                structure_eval["test"]["rule_based"]["accuracy"],
                structure_eval["test"]["rule_based"]["macro_f1"],
            ),
        ]
        for dim in ["lead_frame", "agency_style", "density_band", "rhetorical_mode"]:
            rows.append(
                (
                    f"{dim}_test",
                    style_eval["test"][dim]["accuracy"],
                    style_eval["test"][dim]["macro_f1"],
                )
            )

        names = [r[0] for r in rows]
        acc_vals = [r[1] for r in rows]
        f1_vals = [r[2] for r in rows]
        x = list(range(len(rows)))
        width = 0.4

        plt.figure(figsize=(11, 5))
        plt.bar([i - width / 2 for i in x], acc_vals, width=width, color="#97B67C", label="Accuracy")
        plt.bar([i + width / 2 for i in x], f1_vals, width=width, color="#E7C173", label="Macro F1")
        plt.ylim(0, 1.05)
        plt.title("Model Performance Summary")
        plt.ylabel("Score")
        plt.xticks(x, names, rotation=20, ha="right")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(images_dir, "model_performance_summary.png")
        plt.savefig(out, dpi=180)
        plt.close()
        print(f"[save] graph: {out}")


def main() -> None:
    """CLI entrypoint that orchestrates classification, splitting, and evaluation."""
    print("[start] run_structure_pipeline", flush=True)
    parser = argparse.ArgumentParser(description="Run full structure classifier pipeline.")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-manual-gold",
        action="store_true",
        default=True,
        help="Fail if no manual gold labels are present.",
    )
    parser.add_argument(
        "--allow-bootstrap-fallback",
        action="store_true",
        help="Allow suggested-label fallback when manual gold is absent.",
    )
    parser.add_argument(
        "--regenerate-annotation",
        action="store_true",
        help="Force regeneration of data/gold_headlines_annotation.csv.",
    )
    args = parser.parse_args()

    parsed_path = "data/headlines_parsed.json"
    if not os.path.exists(parsed_path):
        raise FileNotFoundError(
            f"{parsed_path} is missing. Collect/parse first, then rerun this script."
        )

    classifier_script = os.path.join(SCRIPT_DIR, "headline_structure_classifier.py")
    create_gold_script = os.path.join(SCRIPT_DIR, "create_gold_annotation_set.py")
    split_script = os.path.join(SCRIPT_DIR, "split_gold_dataset.py")
    evaluate_script = os.path.join(SCRIPT_DIR, "evaluate_structure_classifier.py")
    style_profiler_script = os.path.join(SCRIPT_DIR, "headline_style_profiler.py")
    style_eval_script = os.path.join(SCRIPT_DIR, "evaluate_style_profile.py")

    _run([sys.executable, classifier_script])
    _run([sys.executable, style_profiler_script])

    full_gold_path = "data/gold_headlines_full_manual.csv"
    default_gold_path = "data/gold_headlines_annotation.csv"
    gold_path = full_gold_path if os.path.exists(full_gold_path) else default_gold_path
    existing_rows = _annotation_rows(gold_path)
    should_generate = args.regenerate_annotation or existing_rows == 0

    if should_generate:
        _run(
            [
                sys.executable,
                create_gold_script,
                "--sample-size",
                str(args.sample_size),
                "--seed",
                str(args.seed),
            ]
        )
        # If we generated a new annotation file, evaluation should use it.
        gold_path = default_gold_path
    else:
        print(
            f"[data] using existing annotation file ({gold_path}, {existing_rows} rows). "
            "Pass --regenerate-annotation to overwrite it.",
            flush=True,
        )

    has_gold = _has_manual_gold_labels(gold_path)

    strict_mode = args.require_manual_gold and not args.allow_bootstrap_fallback
    if strict_mode and not has_gold:
        raise RuntimeError(
            "No manual gold labels found in data/gold_headlines_annotation.csv. "
            "Fill gold_label before running strict evaluation."
        )

    split_output = (
        "data/gold_headlines_full_manual_split.csv"
        if os.path.basename(gold_path) == "gold_headlines_full_manual.csv"
        else "data/gold_headlines_annotation_split.csv"
    )

    split_cmd = [
        sys.executable,
        split_script,
        "--input",
        gold_path,
        "--output",
        split_output,
        "--seed",
        str(args.seed),
    ]
    eval_cmd = [
        sys.executable,
        evaluate_script,
        "--gold-input",
        split_output,
        "--seed",
        str(args.seed),
    ]

    if not has_gold:
        if not args.allow_bootstrap_fallback:
            raise RuntimeError(
                "No manual gold labels found and bootstrap fallback is disabled. "
                "Run scripts/model/annotate_gold_cli.py first, or pass --allow-bootstrap-fallback."
            )
        # Bootstrap mode for end-to-end sanity checks only.
        split_cmd.append("--use-suggested-if-empty")
        eval_cmd.append("--fallback-to-suggested")
        print(
            "[warn] no manual gold labels detected. "
            "Running bootstrap evaluation using suggested labels.",
            flush=True,
        )

    _run(split_cmd)
    _run(eval_cmd)

    style_gold_path = "data/gold_headlines_style_manual.csv"
    if os.path.exists(style_gold_path):
        _run(
            [
                sys.executable,
                style_eval_script,
                "--gold-style-input",
                style_gold_path,
                "--split-input",
                split_output,
            ]
        )

    _generate_readme_graphs(
        structure_pred_path="data/evaluation/gold_predictions.csv",
        structure_eval_path="data/evaluation/classifier_eval.json",
        style_eval_path="data/evaluation_style/style_eval.json",
        images_dir="images",
    )

    print("[done] pipeline complete", flush=True)
    if has_gold:
        print("[mode] evaluation used manual gold labels", flush=True)
    else:
        print("[mode] evaluation used suggested labels as temporary stand-in", flush=True)


if __name__ == "__main__":
    main()
