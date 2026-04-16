"""
News Headline Parser
Parses headlines using spaCy to extract linguistic features.
"""

import spacy
import pandas as pd
import os
import re
import subprocess
import sys
from collections import Counter
from typing import Any, Dict, Iterable


DEFAULT_SPACY_MODELS = [
    "en_core_web_trf",
    "en_core_web_lg",
    "en_core_web_md",
    "en_core_web_sm",
]


def _try_load_model(model_name: str):
    """Attempt to load one spaCy model, returning None on failure."""
    try:
        return spacy.load(model_name)
    except OSError:
        return None


def _download_model(model_name: str) -> bool:
    """Download one spaCy model package via subprocess."""
    try:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def load_spacy_model(preferred_models: Iterable[str] | None = None):
    """
    Load the strongest available spaCy English model.
    
    Returns:
        spaCy nlp object
    """
    # Suppress the pydantic v1 warning for Python 3.14
    import warnings
    warnings.filterwarnings('ignore', message='.*Pydantic V1.*')
    
    models = list(preferred_models or DEFAULT_SPACY_MODELS)

    for model_name in models:
        nlp = _try_load_model(model_name)
        if nlp is not None:
            print(f"[init] spaCy model loaded: {model_name}")
            return nlp

    print("[init] no preferred spaCy model found locally; attempting download")
    # Download only practical fallback models first.
    for model_name in ["en_core_web_md", "en_core_web_sm"]:
        if model_name not in models:
            continue
        if _download_model(model_name):
            nlp = _try_load_model(model_name)
            if nlp is not None:
                print(f"[init] spaCy model downloaded and loaded: {model_name}")
                return nlp

    raise OSError(
        "Could not load/download any spaCy model. "
        "Try: python -m spacy download en_core_web_md"
    )


def normalize_headline_text(headline_text: str) -> str:
    """Normalize punctuation/spacing to improve parser robustness."""
    text = (headline_text or "").strip()
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "--")
        .replace('\\"', '"')
        .replace("\\'", "'")
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_quality(doc) -> int:
    """Score parse plausibility for headline syntax selection."""
    non_punct = [t for t in doc if not t.is_punct]
    if not non_punct:
        return -99

    score = 0
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    root_pos = root.pos_ if root is not None else ""
    if root_pos in {"VERB", "AUX"}:
        score += 4
    elif root_pos in {"NOUN", "PROPN"}:
        score -= 1

    dep_set = {t.dep_ for t in non_punct}
    if "nsubj" in dep_set or "nsubjpass" in dep_set:
        score += 2
    if "dobj" in dep_set or "obj" in dep_set:
        score += 1
    if any(t.pos_ in {"VERB", "AUX"} for t in non_punct):
        score += 2
    if len(non_punct) >= 5 and all(t.pos_ in {"PROPN", "NOUN", "ADP", "DET"} for t in non_punct):
        score -= 2
    return score


def _should_try_lowercase_variant(doc) -> bool:
    """Decide when lowercase fallback parsing may recover syntax."""
    non_punct = [t for t in doc if not t.is_punct]
    if len(non_punct) < 4:
        return False

    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    root_pos = root.pos_ if root is not None else ""
    has_verb = any(t.pos_ in {"VERB", "AUX"} for t in non_punct)
    many_title_tokens = sum(1 for t in non_punct if t.text[:1].isupper()) >= max(3, len(non_punct) // 2)
    return (not has_verb or root_pos in {"PROPN", "NOUN"}) and many_title_tokens


def _extract_doc_features(syntax_doc, entity_doc) -> Dict[str, Any]:
    """Extract parser features from syntax doc and entities from entity doc."""
    tokens = []
    for token in syntax_doc:
        tokens.append({
            'text': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,
            'tag': token.tag_,
            'dep': token.dep_,
            'is_stop': token.is_stop,
            'is_punct': token.is_punct
        })

    pos_sequence = [token.pos_ for token in syntax_doc]
    pos_pattern = ' '.join(pos_sequence)
    dep_structure = [(token.text, token.dep_, token.head.text) for token in syntax_doc]
    dep_relations = [token.dep_ for token in syntax_doc]
    dep_pattern = ' '.join(dep_relations)
    root_tokens = [token for token in syntax_doc if token.dep_ == 'ROOT']
    root = root_tokens[0].text if root_tokens else None
    root_pos = root_tokens[0].pos_ if root_tokens else None
    noun_chunks = [chunk.text for chunk in syntax_doc.noun_chunks]
    entities = [{'text': ent.text, 'label': ent.label_} for ent in entity_doc.ents]

    return {
        'tokens': tokens,
        'pos_sequence': pos_sequence,
        'pos_pattern': pos_pattern,
        'dep_structure': dep_structure,
        'dep_pattern': dep_pattern,
        'entities': entities,
        'root': root,
        'root_pos': root_pos,
        'noun_chunks': noun_chunks,
        'num_tokens': len(syntax_doc)
    }


def parse_headline(headline_text: str, nlp) -> Dict[str, Any]:
    """
    Parse a single headline using spaCy.
    
    Args:
        headline_text: The headline text string
        nlp: spaCy nlp object
    
    Returns:
        Dictionary containing parsed information
    """
    normalized_text = normalize_headline_text(headline_text)
    base_doc = nlp(normalized_text)
    best_doc = base_doc
    best_score = _parse_quality(base_doc)

    # Fallback parse pass for title-cased headlines that lose verb structure.
    if _should_try_lowercase_variant(base_doc):
        lower_doc = nlp(normalized_text.lower())
        lower_score = _parse_quality(lower_doc)
        if lower_score >= best_score + 2:
            best_doc = lower_doc
            best_score = lower_score

    parsed = _extract_doc_features(best_doc, base_doc)
    parsed["parse_quality"] = best_score
    parsed["parse_variant"] = "lowercase_fallback" if best_doc is not base_doc else "original"
    return parsed




def analyze_patterns(parsed_df: pd.DataFrame) -> None:
    """Print a compact snapshot of dominant POS patterns."""
    pattern_counts = Counter(parsed_df['pos_pattern'])
    print(f"[patterns] unique POS patterns: {len(pattern_counts)}")
    for pattern, count in pattern_counts.most_common(5):
        percentage = (count / len(parsed_df)) * 100
        print(f"[patterns] {count:3d} ({percentage:5.1f}%) | {pattern}")


def save_parsed_data(parsed_df: pd.DataFrame, output_dir: str = 'data') -> None:
    """
    Save all parsed headlines.
    
    Args:
        parsed_df: DataFrame with all parsed headlines
        output_dir: Directory to save output
    """
    json_path = os.path.join(output_dir, 'headlines_parsed.json')
    parsed_df.to_json(json_path, orient='records', indent=2, date_format="iso")
    print(f"[save] parsed dataset: {json_path}")
    print(f"[save] total parsed rows: {len(parsed_df)}")


def main() -> None:
    """CLI entrypoint that incrementally parses new headlines with spaCy."""
    import argparse

    parser = argparse.ArgumentParser(description="Parse headlines with spaCy.")
    parser.add_argument(
        "--reparse-all",
        action="store_true",
        help="Reparse all headlines (ignore previously parsed cache).",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_SPACY_MODELS),
        help="Comma-separated model preference order.",
    )
    args = parser.parse_args()

    print("[start] parse_headlines")
    
    headlines_file = 'data/headlines.json'
    parsed_file = 'data/headlines_parsed.json'
    
    if not os.path.exists(headlines_file):
        print(f"[error] missing input: {headlines_file}")
        print("[hint] run scripts/pipeline/collect_headlines.py first")
        return
    
    # Load all headlines
    all_headlines_df = pd.read_json(headlines_file)
    print(f"[load] total headlines: {len(all_headlines_df)}")
    
    # Check which headlines have already been parsed
    if os.path.exists(parsed_file) and not args.reparse_all:
        parsed_df_existing = pd.read_json(parsed_file)
        print(f"[load] already parsed: {len(parsed_df_existing)}")
        
        # Find unparsed headlines by comparing headline text
        parsed_headlines = set(parsed_df_existing['headline'])
        unparsed_df = all_headlines_df[~all_headlines_df['headline'].isin(parsed_headlines)]
        
        print(f"[plan] new headlines to parse: {len(unparsed_df)}")
        
        if len(unparsed_df) == 0:
            print("[done] nothing new to parse")
            return
    else:
        print("[plan] reparsing full dataset")
        unparsed_df = all_headlines_df
        parsed_df_existing = None
    
    # Parse only the unparsed headlines
    print("[init] loading spaCy model")
    model_pref = [m.strip() for m in args.models.split(",") if m.strip()]
    nlp = load_spacy_model(model_pref)
    
    print(f"[run] parsing {len(unparsed_df)} headlines")
    parsed_data = []
    
    for idx, row in unparsed_df.iterrows():
        if idx % 10 == 0:
            print(f"  Processing {len(parsed_data)}/{len(unparsed_df)}...", end='\r')
        
        headline = row['headline']
        parsed = parse_headline(headline, nlp)
        
        # Combine original data with parsed data
        result = {
            'headline': headline,
            'category': row.get('category', ''),
            'source': row.get('source', ''),
            'collected_at': row.get('collected_at', ''),
            'pos_pattern': parsed['pos_pattern'],
            'dep_pattern': parsed['dep_pattern'],
            'num_tokens': parsed['num_tokens'],
            'root': parsed['root'],
            'root_pos': parsed['root_pos'],
            'noun_chunks': parsed['noun_chunks'],
            'entities': parsed['entities'],
            'tokens': parsed['tokens'],
            'dep_structure': parsed['dep_structure']
        }
        parsed_data.append(result)
    
    print(f"[run] processed {len(unparsed_df)}/{len(unparsed_df)}")
    
    # Create DataFrame of newly parsed data
    new_parsed_df = pd.DataFrame(parsed_data)
    
    # Combine with existing parsed data if any
    if parsed_df_existing is not None:
        combined_parsed_df = pd.concat([parsed_df_existing, new_parsed_df], ignore_index=True)
        print(f"[merge] combined parsed rows: {len(combined_parsed_df)}")
    else:
        combined_parsed_df = new_parsed_df
    
    # Analyze patterns (on full dataset)
    analyze_patterns(combined_parsed_df)
    
    # Save results
    save_parsed_data(combined_parsed_df)
    
    print("[done] parse_headlines complete")


if __name__ == '__main__':
    main()
