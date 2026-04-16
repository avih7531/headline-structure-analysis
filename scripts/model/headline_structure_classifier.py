"""
Rule-based structural classifier for news headlines.

This module assigns a single structural label to each parsed headline using
spaCy POS/dependency outputs that already exist in data/headlines_parsed.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd


LABELS = [
    "question_form",
    "passive_clause",
    "coordination",
    "noun_phrase_fragment",
    "simple_clause",
    "other",
]


WH_WORDS = {"who", "what", "when", "where", "why", "how", "which"}
AUX_START = {
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "has",
    "have",
    "had",
}

BE_AUX = {"is", "are", "was", "were", "be", "been", "being", "get", "gets", "got"}
COORD_WORDS = (" and ", " but ", " while ", " yet ")
VERB_CLUES = {
    "says",
    "say",
    "warns",
    "threatens",
    "pushes",
    "rejects",
    "declares",
    "announces",
    "kills",
    "hits",
    "calls",
    "moves",
    "leaves",
    "appoints",
    "demands",
    "faces",
    "finalizes",
    "resigns",
    "slams",
    "hunts",
    "laughs",
    "heads",
    "breaks",
    "collides",
    "amends",
    "signs",
    "set",
    "becomes",
}
PASSIVE_PARTICIPLE_CLUES = {"accused", "charged", "detained", "disbarred", "earmarked"}
PASSIVE_ADJECTIVAL_EXCEPTIONS = {"petrified"}


def _non_punct_tokens(tokens: List[Dict]) -> List[Dict]:
    """Return tokens excluding punctuation-only entries."""
    return [tok for tok in tokens if not tok.get("is_punct", False)]


def _has_question_form(headline: str, tokens: List[Dict]) -> bool:
    """Detect interrogative headline form via punctuation/start cues."""
    clean_tokens = _non_punct_tokens(tokens)
    if not clean_tokens:
        return False

    first = clean_tokens[0]["text"].lower()
    has_verb = any(tok.get("pos") in {"VERB", "AUX"} for tok in clean_tokens)
    return ("?" in headline) or (
        has_verb and (first in WH_WORDS or first in AUX_START)
    )


def _has_passive(headline: str, tokens: List[Dict]) -> bool:
    """Detect passive constructions using dependency and lexical heuristics."""
    root_token = next((tok for tok in tokens if tok.get("dep") == "ROOT"), None)
    root_text = str(root_token.get("text", "")).lower() if root_token else ""

    if root_text in PASSIVE_ADJECTIVAL_EXCEPTIONS:
        return False

    dep_set = {tok.get("dep") for tok in tokens}
    if "nsubjpass" in dep_set:
        return True

    # Headlines like "Nine killed ..." (ellipsis of "were killed").
    if len(tokens) >= 2 and tokens[0].get("tag") == "CD" and tokens[1].get("tag") == "VBN":
        return True

    # Fallback pattern: be-aux + VBN in sequence (avoid have+VBN perfect).
    for idx in range(len(tokens) - 1):
        aux_text = str(tokens[idx].get("text", "")).lower()
        prev_text = str(tokens[idx - 1].get("text", "")).lower() if idx > 0 else ""
        if (
            tokens[idx].get("pos") == "AUX"
            and aux_text in BE_AUX
            and tokens[idx + 1].get("tag") == "VBN"
            and prev_text not in {"after", "before", "while", "as"}
        ):
            return True

    # spaCy often marks non-root passive fragments with auxpass.
    if "auxpass" in dep_set:
        if root_token and root_token.get("pos") in {"VERB", "AUX"}:
            return True

    # Nominal roots often carry passive participial modifiers ("organizer accused ...").
    if root_token and root_token.get("pos") in {"NOUN", "PROPN"}:
        for tok in tokens:
            if (
                tok.get("dep") == "acl"
                and tok.get("tag") == "VBN"
                and str(tok.get("text", "")).lower() in PASSIVE_PARTICIPLE_CLUES
                and not _has_verb_clue(tokens, headline)
            ):
                return True
    return False


def _has_coordination(tokens: List[Dict], headline: str) -> bool:
    """Detect coordinated multi-clause/list headline structures."""
    if ";" in headline:
        return True
    if headline.count(",") >= 2 and ", and " in headline.lower():
        return True

    early_tokens = tokens[:6]
    if "," in headline and sum(
        1 for tok in early_tokens if str(tok.get("text", "")).lower() in VERB_CLUES
    ) >= 2:
        return True

    dep_set = {tok.get("dep") for tok in tokens}
    if "cc" in dep_set and any(
        tok.get("dep") == "conj" and tok.get("pos") in {"VERB", "AUX"} for tok in tokens
    ):
        return True

    return False


def _has_verb_clue(tokens: List[Dict], headline: str) -> bool:
    """Infer finite-clause likelihood from curated lexical verb clues."""
    text_words = re.findall(r"[A-Za-z']+", headline.lower())
    if any(word in VERB_CLUES for word in text_words):
        return True
    return any(str(tok.get("text", "")).lower() in VERB_CLUES for tok in tokens)


def _is_noun_phrase_fragment(tokens: List[Dict], root_pos: str, headline: str) -> bool:
    """Detect nominal headline fragments without a finite clause."""
    clean_tokens = _non_punct_tokens(tokens)
    if not clean_tokens:
        return False

    has_finite_verb = any(
        tok.get("tag") in {"VBD", "VBP", "VBZ", "MD"} for tok in clean_tokens
    )
    has_any_verb = any(tok.get("pos") in {"VERB", "AUX"} for tok in clean_tokens)
    root_token = next((tok for tok in clean_tokens if tok.get("dep") == "ROOT"), None)

    if _has_verb_clue(tokens, headline):
        return False
    if not has_any_verb:
        return True

    # Common nominal headline pattern: participial modifier but no finite clause.
    if not has_finite_verb and root_pos in {"NOUN", "PROPN", "ADJ"}:
        return True
    if not has_finite_verb and root_token and root_token.get("tag") == "VBG":
        return True
    if not has_finite_verb and " to " in f" {headline.lower()} " and root_token and root_token.get("tag") == "VB":
        return True

    if (
        len(clean_tokens) >= 3
        and "." in str(clean_tokens[0].get("text", ""))
        and str(clean_tokens[1].get("text", "")).lower() == "said"
        and str(clean_tokens[2].get("text", "")).lower() == "to"
    ):
        return True

    if ":" in headline:
        prefix = headline.split(":", 1)[0].strip().lower()
        if len(prefix.split()) <= 6 and any(marker in prefix for marker in {"live", "update"}):
            return True

    return False


def _is_simple_clause(tokens: List[Dict], root_pos: str, headline: str) -> bool:
    """Detect canonical finite subject-verb clause headlines."""
    dep_set = {tok.get("dep") for tok in tokens}
    has_subject = "nsubj" in dep_set or "csubj" in dep_set
    has_finite = any(tok.get("tag") in {"VBD", "VBP", "VBZ"} for tok in tokens)
    has_modal = any(tok.get("tag") == "MD" for tok in tokens)
    if root_pos in {"VERB", "AUX"} and has_subject and has_finite:
        return True
    if root_pos in {"VERB", "AUX"} and has_subject and has_modal:
        return True
    if headline.lower().startswith(("here's ", "here is ")):
        return True
    if _has_verb_clue(tokens, headline):
        return True
    return False


def classify_record(record: Dict) -> Tuple[str, List[str]]:
    """Classify a single parsed headline record and return matched rules."""
    headline = record.get("headline", "")
    tokens = record.get("tokens", [])
    root_pos = record.get("root_pos", "") or ""
    matched_rules: List[str] = []

    if _has_question_form(headline, tokens):
        matched_rules.append("question_punctuation_or_interrogative_start")
        return "question_form", matched_rules

    if _has_passive(headline, tokens):
        matched_rules.append("passive_dependency_or_aux_vbn")
        return "passive_clause", matched_rules

    if _has_coordination(tokens, headline):
        matched_rules.append("coordination_cc_conj_pattern")
        return "coordination", matched_rules

    if _is_noun_phrase_fragment(tokens, root_pos, headline):
        matched_rules.append("no_finite_clause_nominal_fragment")
        return "noun_phrase_fragment", matched_rules

    if _is_simple_clause(tokens, root_pos, headline):
        matched_rules.append("finite_subject_verb_clause")
        return "simple_clause", matched_rules

    matched_rules.append("fallback_other")
    return "other", matched_rules


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply structure classification to all rows in a dataframe."""
    records = df.to_dict(orient="records")
    labels = []
    rules = []
    for rec in records:
        label, matched_rules = classify_record(rec)
        labels.append(label)
        rules.append(";".join(matched_rules))

    out = df.copy()
    out["predicted_structure"] = labels
    out["matched_rules"] = rules
    return out


def load_parsed_dataframe(path: str) -> pd.DataFrame:
    """Load parsed headline JSON into a pandas dataframe."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parsed file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return pd.DataFrame(data)


def main() -> None:
    """CLI entrypoint for batch structure classification."""
    parser = argparse.ArgumentParser(description="Classify headline structures.")
    parser.add_argument(
        "--input",
        default="data/headlines_parsed.json",
        help="Path to parsed headline JSON.",
    )
    parser.add_argument(
        "--output",
        default="data/headline_structure_predictions.csv",
        help="Path to output CSV predictions.",
    )
    args = parser.parse_args()

    df = load_parsed_dataframe(args.input)
    classified = classify_dataframe(df)
    classified.to_csv(args.output, index=False)

    counts = Counter(classified["predicted_structure"])
    print("Saved predictions:", args.output)
    print("Label distribution:")
    for label in LABELS:
        print(f"  {label:22} {counts.get(label, 0)}")


if __name__ == "__main__":
    main()
