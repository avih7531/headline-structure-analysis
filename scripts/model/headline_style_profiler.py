"""
Headline style profiler.

Builds a richer per-headline profile on top of structural classification:
- structure label
- lead frame
- agency style
- information density
- rhetorical mode
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List

import pandas as pd

# Ensure project root importability when executed as script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.model.headline_structure_classifier import classify_record


CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}
FUNCTION_POS = {"DET", "PRON", "ADP", "AUX", "CCONJ", "SCONJ", "PART"}
LIVE_CUES = (" live", "live:", "live updates", "update", "breaking")
EXPLAINER_CUES = (
    "what to know",
    "special report",
    "analysis",
    "explainer",
    "why ",
    "how ",
)


def _non_punct(tokens: List[Dict]) -> List[Dict]:
    return [tok for tok in tokens if not tok.get("is_punct", False)]


def predict_lead_frame(tokens: List[Dict]) -> str:
    clean = _non_punct(tokens)
    if not clean:
        return "other_lead"
    pos = clean[0].get("pos", "")
    if pos in {"NOUN", "PROPN", "PRON"}:
        return "actor_first"
    if pos in {"VERB", "AUX"}:
        return "action_first"
    if pos in {"ADV", "ADJ", "NUM", "DET", "ADP"}:
        return "context_first"
    return "other_lead"


def predict_agency_style(tokens: List[Dict], structure_label: str) -> str:
    if structure_label != "passive_clause":
        return "active_or_nonpassive"
    dep_set = {tok.get("dep", "") for tok in tokens}
    has_agent = "agent" in dep_set or any(str(tok.get("text", "")).lower() == "by" for tok in tokens)
    return "passive_with_agent" if has_agent else "passive_agent_omitted"


def compute_density(tokens: List[Dict]) -> tuple[float, str]:
    content_count = 0
    function_count = 0
    for tok in _non_punct(tokens):
        pos = tok.get("pos", "")
        if pos in CONTENT_POS:
            content_count += 1
        elif pos in FUNCTION_POS:
            function_count += 1
    total = content_count + function_count
    score = content_count / total if total else 0.0
    if score >= 0.70:
        band = "high_density"
    elif score >= 0.50:
        band = "medium_density"
    else:
        band = "low_density"
    return score, band


def predict_rhetorical_mode(headline: str, structure_label: str) -> str:
    lower = headline.lower()
    if structure_label == "question_form":
        return "question_hook"
    if any(cue in lower for cue in LIVE_CUES):
        return "live_or_alert"
    if ":" in headline and any(cue in lower for cue in ("report", "analysis", "update")):
        return "analysis_explainer"
    if any(cue in lower for cue in EXPLAINER_CUES):
        return "analysis_explainer"
    return "straight_report"


def profile_record(record: Dict) -> Dict:
    structure_label, matched_rules = classify_record(record)
    tokens = record.get("tokens", [])
    headline = record.get("headline", "")

    lead_frame = predict_lead_frame(tokens)
    agency_style = predict_agency_style(tokens, structure_label)
    density_score, density_band = compute_density(tokens)
    rhetorical_mode = predict_rhetorical_mode(headline, structure_label)

    return {
        "predicted_structure": structure_label,
        "matched_rules": ";".join(matched_rules),
        "lead_frame": lead_frame,
        "agency_style": agency_style,
        "density_score": round(density_score, 4),
        "density_band": density_band,
        "rhetorical_mode": rhetorical_mode,
        "style_signature": f"{structure_label}|{lead_frame}|{rhetorical_mode}",
    }


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    records = df.to_dict(orient="records")
    profiled_rows = [profile_record(rec) for rec in records]
    out = df.copy()
    for key in profiled_rows[0].keys():
        out[key] = [row[key] for row in profiled_rows]
    return out


def _print_summary(df: pd.DataFrame) -> None:
    print("Saved style profiles with the following distributions:")
    for col in ["predicted_structure", "lead_frame", "agency_style", "density_band", "rhetorical_mode"]:
        counts = Counter(df[col])
        print(f"\n{col}:")
        for label, count in counts.most_common():
            pct = 100 * count / len(df)
            print(f"  {label:24} {count:3} ({pct:4.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate headline style profiles.")
    parser.add_argument("--input", default="data/headlines_parsed.json")
    parser.add_argument("--output", default="data/headline_style_profiles.csv")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    df = pd.DataFrame(data)

    profiled = profile_dataframe(df)
    profiled.to_csv(args.output, index=False)
    print(f"Saved profiles: {args.output}")
    _print_summary(profiled)


if __name__ == "__main__":
    main()
