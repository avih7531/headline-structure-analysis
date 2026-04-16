"""
Interactive CLI for manual headline-structure annotation.

Usage:
  python annotate_gold_cli.py
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import pandas as pd


LABEL_KEYS: Dict[str, str] = {
    "q": "question_form",
    "p": "passive_clause",
    "c": "coordination",
    "n": "noun_phrase_fragment",
    "s": "simple_clause",
    "o": "other",
}


def _print_legend() -> None:
    print("\nLabel keys:")
    for key, label in LABEL_KEYS.items():
        print(f"  {key}: {label}")
    print("  enter: keep current label")
    print("  b: back one item")
    print("  x: save and exit\n")


def main() -> None:
    """CLI entrypoint for keyboard-driven manual gold annotation."""
    parser = argparse.ArgumentParser(description="Manual annotation CLI for gold labels.")
    parser.add_argument("--input", default="data/gold_headlines_annotation.csv")
    parser.add_argument("--annotator", default="")
    parser.add_argument(
        "--show-suggestion",
        action="store_true",
        help="Show suggested_label during annotation (off by default to reduce bias).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File not found: {args.input}")

    df = pd.read_csv(args.input).fillna("")
    required_cols = {"headline", "gold_label", "suggested_label", "annotator"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    unlabeled = df[df["gold_label"].str.strip() == ""]
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Unlabeled rows remaining: {len(unlabeled)}")
    _print_legend()

    indices = df.index.tolist()
    i = 0
    while i < len(indices):
        idx = indices[i]
        row = df.loc[idx]

        print("=" * 90)
        print(f"Item {i + 1}/{len(indices)} | row_id={idx} | current={row['gold_label'] or '<empty>'}")
        print(f"Headline: {row['headline']}")
        if args.show_suggestion:
            print(f"Suggested: {row['suggested_label']}")

        choice = input("Label [q/p/c/n/s/o, enter/b/x]: ").strip().lower()

        if choice == "x":
            break
        if choice == "b":
            i = max(0, i - 1)
            continue
        if choice == "":
            i += 1
            continue
        if choice in LABEL_KEYS:
            df.at[idx, "gold_label"] = LABEL_KEYS[choice]
            if args.annotator:
                df.at[idx, "annotator"] = args.annotator
            i += 1
            continue

        print("Invalid input. Try again.")

    df.to_csv(args.input, index=False)
    labeled_count = (df["gold_label"].str.strip() != "").sum()
    print(f"\nSaved {args.input}")
    print(f"Labeled rows: {labeled_count}/{len(df)}")


if __name__ == "__main__":
    main()
