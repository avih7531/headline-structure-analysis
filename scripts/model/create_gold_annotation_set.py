"""
Create a manually annotatable gold-standard sample of headlines.

This script does not recollect data. It samples from existing
data/headlines_parsed.json and pre-fills suggested labels from the
rule-based classifier to speed up annotation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

import pandas as pd

from headline_structure_classifier import classify_record


def _load_parsed(path: str) -> pd.DataFrame:
    """Load parsed headline records from JSON into dataframe form."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parsed data not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return pd.DataFrame(data)


def _stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Sample rows with light category balancing when category exists."""
    if sample_size >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if "category" not in df.columns:
        return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    rng = random.Random(seed)
    grouped: Dict[str, List[int]] = {}
    for idx, row in df.iterrows():
        key = str(row.get("category", "unknown")).lower()
        grouped.setdefault(key, []).append(idx)

    cats = list(grouped.keys())
    if not cats:
        return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    # Equal allocation, then remainder by available capacity.
    base = sample_size // len(cats)
    remainder = sample_size % len(cats)

    chosen_indices: List[int] = []
    for cat in cats:
        pool = grouped[cat][:]
        rng.shuffle(pool)
        take = min(base, len(pool))
        chosen_indices.extend(pool[:take])
        grouped[cat] = pool[take:]

    # Fill remainder from all categories with remaining capacity.
    remaining_pool = []
    for pool in grouped.values():
        remaining_pool.extend(pool)
    rng.shuffle(remaining_pool)
    chosen_indices.extend(remaining_pool[:remainder])

    # Safety fill if category balancing undershot due to low counts.
    if len(chosen_indices) < sample_size:
        missing = sample_size - len(chosen_indices)
        leftovers = [idx for idx in df.index if idx not in set(chosen_indices)]
        rng.shuffle(leftovers)
        chosen_indices.extend(leftovers[:missing])

    return df.loc[chosen_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    """CLI entrypoint to build a manual annotation sheet with suggestions."""
    parser = argparse.ArgumentParser(description="Create a gold annotation sample.")
    parser.add_argument("--input", default="data/headlines_parsed.json")
    parser.add_argument("--output", default="data/gold_headlines_annotation.csv")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    parsed_df = _load_parsed(args.input)
    sample_df = _stratified_sample(parsed_df, args.sample_size, args.seed)

    rows = []
    for idx, row in sample_df.iterrows():
        label, rules = classify_record(row.to_dict())
        rows.append(
            {
                "item_id": idx + 1,
                "headline": row.get("headline", ""),
                "category": row.get("category", ""),
                "source": row.get("source", ""),
                "collected_at": row.get("collected_at", ""),
                "suggested_label": label,
                "suggested_rules": ";".join(rules),
                "gold_label": "",
                "split": "",
                "annotator": "",
                "notes": "",
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"Saved annotation sheet: {args.output}")
    print(f"Rows: {len(out_df)}")
    print("Reminder: fill gold_label manually, then run split_gold_dataset.py")


if __name__ == "__main__":
    main()
