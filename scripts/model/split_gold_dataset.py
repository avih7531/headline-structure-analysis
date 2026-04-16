"""
Create train/dev/test splits from a manually labeled gold dataset.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Tuple

import pandas as pd


def _allocate_counts(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Convert split ratios into concrete train/dev/test counts."""
    train_r, dev_r, _ = ratios
    train_n = int(n * train_r)
    dev_n = int(n * dev_r)
    test_n = n - train_n - dev_n

    # Ensure at least one dev/test when possible.
    if n >= 3:
        if dev_n == 0:
            dev_n = 1
            train_n = max(train_n - 1, 1)
        if test_n == 0:
            test_n = 1
            train_n = max(train_n - 1, 1)

    # Rebalance in case of edge effects.
    total = train_n + dev_n + test_n
    if total != n:
        train_n += n - total
    return train_n, dev_n, test_n


def _split_indices(indices: List[int], ratios: Tuple[float, float, float], rng: random.Random) -> Dict[int, str]:
    """Assign one label-homogeneous index list into train/dev/test splits."""
    idxs = indices[:]
    rng.shuffle(idxs)
    train_n, dev_n, test_n = _allocate_counts(len(idxs), ratios)

    split_map: Dict[int, str] = {}
    for idx in idxs[:train_n]:
        split_map[idx] = "train"
    for idx in idxs[train_n : train_n + dev_n]:
        split_map[idx] = "dev"
    for idx in idxs[train_n + dev_n : train_n + dev_n + test_n]:
        split_map[idx] = "test"
    return split_map


def main() -> None:
    """CLI entrypoint to create stratified train/dev/test assignments."""
    parser = argparse.ArgumentParser(description="Split gold labels into train/dev/test.")
    parser.add_argument("--input", default="data/gold_headlines_annotation.csv")
    parser.add_argument("--output", default="data/gold_headlines_annotation_split.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--use-suggested-if-empty",
        action="store_true",
        help="Fallback to suggested_label where gold_label is empty.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input).fillna("")
    if "gold_label" not in df.columns:
        raise ValueError("Input file must include a gold_label column.")

    if args.use_suggested_if_empty and "suggested_label" in df.columns:
        empty_mask = df["gold_label"].str.strip() == ""
        df.loc[empty_mask, "gold_label"] = df.loc[empty_mask, "suggested_label"]

    labeled_df = df[df["gold_label"].str.strip() != ""].copy()
    if labeled_df.empty:
        raise ValueError("No labeled rows found. Fill gold_label before splitting.")

    ratios = (args.train_ratio, args.dev_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Ratios must sum to 1.0")

    rng = random.Random(args.seed)
    split_map: Dict[int, str] = {}

    for _, group in labeled_df.groupby("gold_label"):
        group_indices = group.index.tolist()
        split_map.update(_split_indices(group_indices, ratios, rng))

    df["split"] = df.get("split", "")
    for idx, split in split_map.items():
        df.at[idx, "split"] = split

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)

    base_dir = os.path.dirname(args.output) or "."
    labeled_out = df[df["gold_label"].str.strip() != ""].copy()
    labeled_out[labeled_out["split"] == "train"].to_csv(
        os.path.join(base_dir, "gold_train.csv"), index=False
    )
    labeled_out[labeled_out["split"] == "dev"].to_csv(
        os.path.join(base_dir, "gold_dev.csv"), index=False
    )
    labeled_out[labeled_out["split"] == "test"].to_csv(
        os.path.join(base_dir, "gold_test.csv"), index=False
    )

    print(f"Saved split dataset: {args.output}")
    print("Split counts:")
    print(labeled_out["split"].value_counts())


if __name__ == "__main__":
    main()
