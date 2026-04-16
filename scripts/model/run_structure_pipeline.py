"""
Run the full structure-classifier workflow on existing parsed headlines.

This script intentionally does NOT collect or parse new headlines.
It only uses data/headlines_parsed.json that already exists.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SANITY_HEADLINES = [
    "Donald Trump Replaces Joseph Biden as President of US; mixed reactions among citizens",
    "Nine killed in second Turkish school shooting in two days",
    "At CPAC, many Republicans stand by Trump on Iran. But they're divided on how the war could end.",
]


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


def _parse_seed_list(seed_csv: str) -> list[int]:
    """Parse comma-separated seeds into a de-duplicated integer list."""
    seeds: list[int] = []
    for raw in seed_csv.split(","):
        raw = raw.strip()
        if not raw:
            continue
        val = int(raw)
        if val not in seeds:
            seeds.append(val)
    return seeds


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return mean and sample-standard-deviation for a numeric series."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _run_seed_sweep(
    split_script: str,
    evaluate_script: str,
    style_eval_script: str,
    gold_path: str,
    style_gold_path: str | None,
    seeds: list[int],
) -> None:
    """Run repeated split/evaluation over multiple seeds and save aggregate stats."""
    if len(seeds) < 3:
        print("[sweep] skipped (need at least 3 seeds for stability)", flush=True)
        return

    sweep_root = "data/evaluation_seed_sweep"
    os.makedirs(sweep_root, exist_ok=True)
    rows = []

    for seed in seeds:
        split_path = os.path.join(sweep_root, f"gold_split_seed_{seed}.csv")
        structure_dir = os.path.join(sweep_root, f"seed_{seed}", "structure")
        style_dir = os.path.join(sweep_root, f"seed_{seed}", "style")

        _run(
            [
                sys.executable,
                split_script,
                "--input",
                gold_path,
                "--output",
                split_path,
                "--seed",
                str(seed),
            ]
        )
        _run(
            [
                sys.executable,
                evaluate_script,
                "--gold-input",
                split_path,
                "--seed",
                str(seed),
                "--output-dir",
                structure_dir,
            ]
        )

        row: dict[str, float | int] = {"seed": seed}
        with open(os.path.join(structure_dir, "classifier_eval.json"), "r", encoding="utf-8") as handle:
            s_eval = json.load(handle)
        row["structure_test_accuracy"] = s_eval["test"]["rule_based"]["accuracy"]
        row["structure_test_macro_f1"] = s_eval["test"]["rule_based"]["macro_f1"]
        row["structure_dev_macro_f1"] = s_eval["dev"]["rule_based"]["macro_f1"]

        if style_gold_path and os.path.exists(style_gold_path):
            _run(
                [
                    sys.executable,
                    style_eval_script,
                    "--gold-style-input",
                    style_gold_path,
                    "--split-input",
                    split_path,
                    "--output-dir",
                    style_dir,
                ]
            )
            with open(os.path.join(style_dir, "style_eval.json"), "r", encoding="utf-8") as handle:
                st_eval = json.load(handle)
            row["style_test_rhetorical_macro_f1"] = st_eval["test"]["rhetorical_mode"]["macro_f1"]
            row["style_test_agency_macro_f1"] = st_eval["test"]["agency_style"]["macro_f1"]

        rows.append(row)

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(os.path.join(sweep_root, "seed_metrics.csv"), index=False)

    summary: dict[str, dict[str, float]] = {"n_seeds": len(seeds)}  # type: ignore[assignment]
    for metric in [c for c in sweep_df.columns if c != "seed"]:
        mean, std = _mean_std(sweep_df[metric].astype(float).tolist())
        summary[metric] = {"mean": mean, "std": std}

    with open(os.path.join(sweep_root, "seed_sweep_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    struct_mean = summary["structure_test_macro_f1"]["mean"]
    struct_std = summary["structure_test_macro_f1"]["std"]
    print(
        f"[sweep] structure test macro F1 over {len(seeds)} seeds: {struct_mean:.3f} +/- {struct_std:.3f}",
        flush=True,
    )
    print(f"[save] seed sweep summary: {os.path.join(sweep_root, 'seed_sweep_summary.json')}", flush=True)


def _generate_seed_stability_graph(
    seed_metrics_path: str,
    seed_summary_path: str,
    images_dir: str = "images",
) -> None:
    """Generate a compact graph visualizing split-seed performance variance."""
    if not (os.path.exists(seed_metrics_path) and os.path.exists(seed_summary_path)):
        return

    df = pd.read_csv(seed_metrics_path)
    if df.empty or "seed" not in df.columns:
        return

    with open(seed_summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    os.makedirs(images_dir, exist_ok=True)
    seeds = df["seed"].astype(int).tolist()
    struct_test_f1 = df["structure_test_macro_f1"].astype(float).tolist()
    struct_test_acc = df["structure_test_accuracy"].astype(float).tolist()

    mean_f1 = summary.get("structure_test_macro_f1", {}).get("mean", 0.0)
    std_f1 = summary.get("structure_test_macro_f1", {}).get("std", 0.0)
    mean_acc = summary.get("structure_test_accuracy", {}).get("mean", 0.0)
    std_acc = summary.get("structure_test_accuracy", {}).get("std", 0.0)

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True)
    x = range(len(seeds))

    # Top: structure test macro F1 by seed with mean +/- std band.
    axes[0].plot(x, struct_test_f1, marker="o", linewidth=1.8, color="#5E81AC", label="Seed score")
    axes[0].axhline(mean_f1, color="#a97ea1", linestyle="--", linewidth=1.5, label=f"Mean {mean_f1:.3f}")
    axes[0].axhspan(max(0.0, mean_f1 - std_f1), min(1.0, mean_f1 + std_f1), color="#88C0D0", alpha=0.2, label=f"+/-1 SD ({std_f1:.3f})")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Seed Stability: How Much Split Luck Moves Scores")
    axes[0].legend(loc="lower left", fontsize=8)

    # Bottom: structure test accuracy by seed with mean +/- std band.
    axes[1].plot(x, struct_test_acc, marker="o", linewidth=1.8, color="#97b67c", label="Seed score")
    axes[1].axhline(mean_acc, color="#a97ea1", linestyle="--", linewidth=1.5, label=f"Mean {mean_acc:.3f}")
    axes[1].axhspan(max(0.0, mean_acc - std_acc), min(1.0, mean_acc + std_acc), color="#e7c173", alpha=0.25, label=f"+/-1 SD ({std_acc:.3f})")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([str(s) for s in seeds])
    axes[1].set_xlabel("Random split seed")
    axes[1].legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    out = os.path.join(images_dir, "seed_stability_summary.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] graph: {out}", flush=True)


def _write_sanity_cases(output_path: str = "data/evaluation/sanity_cases.json") -> None:
    """Run fixed stress-test headlines and save end-to-end predictions."""
    from scripts.model.headline_structure_classifier import classify_record
    from scripts.model.headline_style_profiler import profile_record
    from scripts.pipeline.parse_headlines import load_spacy_model, parse_headline

    nlp = load_spacy_model()
    rows = []
    for headline in SANITY_HEADLINES:
        parsed = parse_headline(headline, nlp)
        record = {"headline": headline, **parsed}
        structure, _ = classify_record(record)
        style = profile_record(record)
        rows.append(
            {
                "headline": headline,
                "parse_variant": parsed.get("parse_variant", "original"),
                "parse_quality": parsed.get("parse_quality", 0),
                "predicted_structure": structure,
                "lead_frame": style["lead_frame"],
                "agency_style": style["agency_style"],
                "density_band": style["density_band"],
                "rhetorical_mode": style["rhetorical_mode"],
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"[save] sanity-case predictions: {output_path}", flush=True)


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

        # 3) Performance heatmap (compact + intuitive gradient).
        heatmap_rows = [
            ("Structure (all)", structure_all_acc, structure_all_macro),
            (
                "Structure (test)",
                structure_eval["test"]["rule_based"]["accuracy"],
                structure_eval["test"]["rule_based"]["macro_f1"],
            ),
            ("Lead frame (test)", style_eval["test"]["lead_frame"]["accuracy"], style_eval["test"]["lead_frame"]["macro_f1"]),
            ("Agency style (test)", style_eval["test"]["agency_style"]["accuracy"], style_eval["test"]["agency_style"]["macro_f1"]),
            ("Density band (test)", style_eval["test"]["density_band"]["accuracy"], style_eval["test"]["density_band"]["macro_f1"]),
            (
                "Rhetorical mode (test)",
                style_eval["test"]["rhetorical_mode"]["accuracy"],
                style_eval["test"]["rhetorical_mode"]["macro_f1"],
            ),
        ]
        row_labels = [r[0] for r in heatmap_rows]
        matrix = [[r[1], r[2]] for r in heatmap_rows]
        col_labels = ["Accuracy", "Macro F1"]
        tui_rgy = LinearSegmentedColormap.from_list(
            "tui_rgy",
            ["#b74e58", "#e7c173", "#97b67c"],
            N=256,
        )

        plt.figure(figsize=(6.4, 4.8))
        im = plt.imshow(matrix, cmap=tui_rgy, vmin=0.0, vmax=1.0, aspect="auto")
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("Score (0 to 1)")
        plt.title("Performance Heatmap")
        plt.xticks(range(len(col_labels)), col_labels)
        plt.yticks(range(len(row_labels)), row_labels)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                text_color = "white" if value < 0.45 else "black"
                plt.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=9)
        plt.tight_layout()
        out = os.path.join(images_dir, "performance_heatmap.png")
        plt.savefig(out, dpi=180, bbox_inches="tight")
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
    parser.add_argument(
        "--sweep-seeds",
        default="13,42,87,123,202",
        help="Comma-separated seeds for split-stability evaluation sweep.",
    )
    parser.add_argument(
        "--skip-seed-sweep",
        action="store_true",
        help="Disable multi-seed stability sweep.",
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

    if not args.skip_seed_sweep:
        seeds = _parse_seed_list(args.sweep_seeds)
        _run_seed_sweep(
            split_script=split_script,
            evaluate_script=evaluate_script,
            style_eval_script=style_eval_script,
            gold_path=gold_path,
            style_gold_path=style_gold_path if os.path.exists(style_gold_path) else None,
            seeds=seeds,
        )

    _generate_readme_graphs(
        structure_pred_path="data/evaluation/gold_predictions.csv",
        structure_eval_path="data/evaluation/classifier_eval.json",
        style_eval_path="data/evaluation_style/style_eval.json",
        images_dir="images",
    )
    _write_sanity_cases()
    _generate_seed_stability_graph(
        seed_metrics_path="data/evaluation_seed_sweep/seed_metrics.csv",
        seed_summary_path="data/evaluation_seed_sweep/seed_sweep_summary.json",
        images_dir="images",
    )

    print("[done] pipeline complete", flush=True)
    if has_gold:
        print("[mode] evaluation used manual gold labels", flush=True)
    else:
        print("[mode] evaluation used suggested labels as temporary stand-in", flush=True)


if __name__ == "__main__":
    main()
