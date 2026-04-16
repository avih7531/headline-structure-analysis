"""
Run the full structure-classifier workflow on existing parsed headlines.

This script intentionally does NOT collect or parse new headlines.
It only uses data/headlines_parsed.json that already exists.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _run(cmd: list[str]) -> None:
    """Execute one subprocess command and fail fast on errors."""
    print(f"[run] {' '.join(cmd)}")
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


def main() -> None:
    """CLI entrypoint that orchestrates classification, splitting, and evaluation."""
    print("[start] run_structure_pipeline")
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
            "Pass --regenerate-annotation to overwrite it."
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
            "Running bootstrap evaluation using suggested labels."
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

    print("[done] pipeline complete")
    if has_gold:
        print("[mode] evaluation used manual gold labels")
    else:
        print("[mode] evaluation used suggested labels as temporary stand-in")


if __name__ == "__main__":
    main()
