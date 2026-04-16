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
ACTOR_ENTITY_LABELS = {"PERSON", "ORG", "NORP"}
EVENT_NOUN_CLUES = {
    "bombing",
    "attack",
    "strike",
    "blast",
    "explosion",
    "shooting",
    "clash",
    "conflict",
    "war",
    "crash",
    "earthquake",
    "flood",
    "fire",
    "protest",
    "election",
    "trial",
    "hearing",
    "investigation",
}
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
    """Return non-punctuation tokens for feature extraction."""
    return [tok for tok in tokens if not tok.get("is_punct", False)]


def _starts_with_actor_entity(tokens: List[Dict], entities: List[Dict]) -> bool:
    """Return True when first token anchors a likely human/institutional actor."""
    clean = _non_punct(tokens)
    if not clean:
        return False

    first_text = str(clean[0].get("text", "")).strip()
    first_pos = clean[0].get("pos", "")
    if first_pos == "PRON":
        return True
    if first_pos == "PROPN":
        return True

    for ent in entities:
        ent_text = str(ent.get("text", "")).strip()
        ent_label = str(ent.get("label", "")).strip()
        if not ent_text or ent_label not in ACTOR_ENTITY_LABELS:
            continue
        if ent_text.split()[0] == first_text:
            return True
    return False


def _is_event_head(tokens: List[Dict]) -> bool:
    """Return True when first token appears to be an event noun."""
    clean = _non_punct(tokens)
    if not clean:
        return False
    first = clean[0]
    first_text = str(first.get("text", "")).lower()
    first_lemma = str(first.get("lemma", "")).lower()
    first_pos = first.get("pos", "")
    if first_pos != "NOUN":
        return False
    if first_text in EVENT_NOUN_CLUES or first_lemma in EVENT_NOUN_CLUES:
        return True
    return first_text.endswith("ing") and len(first_text) > 4


def predict_lead_frame(tokens: List[Dict], entities: List[Dict]) -> str:
    """Classify whether headline opens with actor/entity, event, action, or context."""
    clean = _non_punct(tokens)
    if not clean:
        return "other_lead"

    if _starts_with_actor_entity(tokens, entities):
        return "actor_entity_first"
    if _is_event_head(tokens):
        return "event_first"

    pos = clean[0].get("pos", "")
    if pos in {"VERB", "AUX"}:
        return "action_first"
    if pos in {"ADV", "ADJ", "NUM", "DET", "ADP"}:
        return "context_first"
    if pos == "NOUN":
        return "actor_entity_first"
    return "other_lead"


def predict_agency_style(tokens: List[Dict], structure_label: str) -> str:
    """Estimate agency framing, with explicit handling for passive forms."""
    if structure_label != "passive_clause":
        return "active_or_nonpassive"
    dep_set = {tok.get("dep", "") for tok in tokens}
    has_agent = "agent" in dep_set or any(str(tok.get("text", "")).lower() == "by" for tok in tokens)
    return "passive_with_agent" if has_agent else "passive_agent_omitted"


def compute_density(tokens: List[Dict]) -> tuple[float, str]:
    """Compute lexical information density score and discretized band."""
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
    """Infer high-level rhetorical mode from form and discourse cues."""
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
    """Build a complete style profile for a single parsed headline."""
    structure_label, matched_rules = classify_record(record)
    tokens = record.get("tokens", [])
    entities = record.get("entities", [])
    headline = record.get("headline", "")

    lead_frame = predict_lead_frame(tokens, entities)
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
    """Vectorized wrapper applying style profiling across a dataframe."""
    records = df.to_dict(orient="records")
    profiled_rows = [profile_record(rec) for rec in records]
    out = df.copy()
    for key in profiled_rows[0].keys():
        out[key] = [row[key] for row in profiled_rows]
    return out


def _print_summary(df: pd.DataFrame) -> None:
    """Print compact distribution summaries for generated profile fields."""
    print("[summary] style profile distributions:")
    for col in ["predicted_structure", "lead_frame", "agency_style", "density_band", "rhetorical_mode"]:
        counts = Counter(df[col])
        top_label, top_count = counts.most_common(1)[0]
        pct = 100 * top_count / len(df)
        print(f"  - {col:18} top={top_label} ({top_count}, {pct:.1f}%)")


def main() -> None:
    """CLI entrypoint for generating headline style profile CSV output."""
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
    print(f"[save] style profiles: {args.output}")
    _print_summary(profiled)


if __name__ == "__main__":
    main()
