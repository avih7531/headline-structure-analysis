"""
Headline Structure Analysis
Analyzes grammatical patterns and structures in news headlines.
"""

import json
import pandas as pd
import glob
import os
import sys
from collections import Counter, defaultdict
import re

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from scripts.model.headline_structure_classifier import LABELS, classify_dataframe
    from scripts.model.headline_style_profiler import profile_dataframe
except ImportError:
    LABELS = []
    classify_dataframe = None
    profile_dataframe = None


def load_latest_parsed_data():
    """Load the parsed headlines dataset."""
    parsed_file = 'data/headlines_parsed.json'
    
    if not os.path.exists(parsed_file):
        print(f"ERROR: {parsed_file} not found!")
        print("Please run collect_headlines.py and parse_headlines.py first.")
        return None
    
    print(f"Loading: {parsed_file}")
    with open(parsed_file, 'r') as f:
        data = json.load(f)
    
    return pd.DataFrame(data)


def simplify_pos_pattern(pos_pattern, collapse_repetitions=False):
    """
    Simplify POS patterns by grouping similar structures.
    PROPN -> N (noun-like)
    NOUN -> N
    VERB -> V
    AUX -> V (auxiliary verbs)
    ADJ -> A
    ADV -> ADV
    NUM -> NUM
    PUNCT -> (removed)
    Others -> X
    
    If collapse_repetitions=True, collapses consecutive identical tokens:
    N N N -> N+, V V -> V+, etc.
    """
    mapping = {
        'PROPN': 'N',
        'NOUN': 'N',
        'VERB': 'V',
        'AUX': 'V',
        'ADJ': 'A',
        'ADV': 'ADV',
        'NUM': 'NUM',
        'DET': 'DET',
        'ADP': 'P',  # Preposition
        'PART': 'PART',
        'CCONJ': 'C',
        'SCONJ': 'C',
        'PRON': 'PRON',
    }
    
    # Split pattern and map
    tokens = pos_pattern.split()
    simplified = []
    for token in tokens:
        if token == 'PUNCT':
            continue  # Skip punctuation
        simplified.append(mapping.get(token, 'X'))
    
    if not collapse_repetitions:
        return ' '.join(simplified)
    
    # Collapse consecutive repetitions
    if not simplified:
        return ''
    
    collapsed = [simplified[0]]
    for token in simplified[1:]:
        if token != collapsed[-1]:
            collapsed.append(token)
        # If same as previous, skip (already have it)
    
    # Add + to indicate "one or more" for readability
    result = []
    i = 0
    while i < len(simplified):
        current = simplified[i]
        count = 1
        while i + count < len(simplified) and simplified[i + count] == current:
            count += 1
        
        if count > 1:
            result.append(f"{current}+")
        else:
            result.append(current)
        i += count
    
    return ' '.join(result)


def describe_structure_pattern(pattern):
    """
    Convert collapsed pattern to human-readable description.
    """
    descriptions = {
        'N+': 'Noun Phrase',
        'V+': 'Verb Phrase',
        'A+': 'Adjectives',
        'P': 'Prep',
        'DET': 'Det',
        'C': 'Conj',
        'PART': 'Particle',
        'ADV': 'Adverb',
        'NUM': 'Number',
        'PRON': 'Pronoun',
    }
    
    parts = pattern.split()
    readable = []
    for part in parts:
        readable.append(descriptions.get(part, part))
    
    return ' → '.join(readable)


def identify_structure_type(row):
    """
    Identify the headline structure type based on patterns.
    """
    pattern = row['pos_pattern']
    root_pos = row.get('root_pos', '')
    tokens = pattern.split()
    
    # Question headline
    if tokens and tokens[0] in ['SCONJ', 'ADV'] and 'VERB' in tokens:
        # Starts with How, When, Where, Why, etc.
        return 'Question'
    
    # Verb-led headline (active)
    if root_pos == 'VERB' and tokens and tokens[0] in ['VERB', 'AUX']:
        return 'Verb-Led Active'
    
    # Noun phrase headline (no main verb)
    if 'VERB' not in tokens and 'AUX' not in tokens:
        return 'Noun Phrase'
    
    # Subject-Verb structure
    if root_pos == 'VERB':
        return 'Subject-Verb'
    
    # Noun-centered (verb exists but root is noun)
    if root_pos in ['NOUN', 'PROPN']:
        return 'Noun-Centered'
    
    return 'Other'


def analyze_verb_patterns(df):
    """Analyze how verbs are used in headlines."""
    print("\n" + "=" * 70)
    print("VERB USAGE PATTERNS")
    print("=" * 70)
    
    # Headlines with verbs
    has_verb = df['pos_pattern'].str.contains('VERB|AUX')
    print(f"\nHeadlines with verbs: {has_verb.sum()} / {len(df)} ({has_verb.sum()/len(df)*100:.1f}%)")
    print(f"Noun-only headlines: {(~has_verb).sum()} / {len(df)} ({(~has_verb).sum()/len(df)*100:.1f}%)")
    
    # Verb tense analysis (from fine-grained tags)
    verb_headlines = df[has_verb].copy()
    
    # Extract verb forms from tokens
    verb_forms = defaultdict(int)
    for _, row in verb_headlines.iterrows():
        for token in row['tokens']:
            if token['pos'] in ['VERB', 'AUX']:
                tag = token['tag']
                # VBD = past, VBZ/VBP = present, VBN = past participle, VBG = gerund
                verb_forms[tag] += 1
    
    print("\nVerb forms used:")
    for tag, count in sorted(verb_forms.items(), key=lambda x: x[1], reverse=True)[:10]:
        tag_name = {
            'VBD': 'Past tense',
            'VBZ': 'Present 3rd person',
            'VBP': 'Present non-3rd',
            'VBN': 'Past participle',
            'VBG': 'Gerund/progressive',
            'VB': 'Base form',
        }.get(tag, tag)
        print(f"  {tag:5} ({tag_name:20}): {count:3}")


def analyze_length_patterns(df):
    """Analyze headline length and its relation to structure."""
    print("\n" + "=" * 70)
    print("LENGTH ANALYSIS")
    print("=" * 70)
    
    print(f"\nToken count statistics:")
    print(f"  Mean:   {df['num_tokens'].mean():.1f} tokens")
    print(f"  Median: {df['num_tokens'].median():.1f} tokens")
    print(f"  Min:    {df['num_tokens'].min()} tokens")
    print(f"  Max:    {df['num_tokens'].max()} tokens")
    
    # Length distribution
    print("\nLength distribution:")
    length_bins = pd.cut(df['num_tokens'], bins=[0, 8, 12, 16, 100], 
                         labels=['Short (≤8)', 'Medium (9-12)', 'Long (13-16)', 'Very Long (17+)'])
    print(length_bins.value_counts().sort_index())
    
    # Length vs structure type
    df_copy = df.copy()
    df_copy['structure_type'] = df_copy.apply(identify_structure_type, axis=1)
    
    print("\nAverage length by structure type:")
    for struct_type in df_copy['structure_type'].unique():
        avg_len = df_copy[df_copy['structure_type'] == struct_type]['num_tokens'].mean()
        count = (df_copy['structure_type'] == struct_type).sum()
        print(f"  {struct_type:20}: {avg_len:5.1f} tokens (n={count})")


def analyze_proper_noun_density(df):
    """Analyze the use of proper nouns (names, places, organizations)."""
    print("\n" + "=" * 70)
    print("PROPER NOUN ANALYSIS")
    print("=" * 70)
    
    # Calculate PROPN density
    propn_counts = []
    for _, row in df.iterrows():
        propn_count = row['pos_pattern'].split().count('PROPN')
        propn_counts.append(propn_count / row['num_tokens'])
    
    df_copy = df.copy()
    df_copy['propn_density'] = propn_counts
    
    print(f"\nProper noun density:")
    print(f"  Mean: {df_copy['propn_density'].mean()*100:.1f}% of tokens are proper nouns")
    print(f"  Headlines with NO proper nouns: {(df_copy['propn_density'] == 0).sum()}")
    print(f"  Headlines with >50% proper nouns: {(df_copy['propn_density'] > 0.5).sum()}")
    
    # Top headlines by PROPN density
    print("\nExamples of proper-noun heavy headlines:")
    top_propn = df_copy.nlargest(3, 'propn_density')
    for _, row in top_propn.iterrows():
        print(f"  ({row['propn_density']*100:.0f}% PROPN) {row['headline'][:70]}")


def create_macro_structure(pattern):
    """
    Create ultra-simplified macro structure by grouping:
    - N, DET, NUM, PRON → NP (noun phrase)
    - V, PART → VP (verb phrase)
    - A, ADV → MOD (modifier)
    - P → P (preposition)
    - C → CONJ (conjunction)
    """
    tokens = pattern.split()
    macro = []
    
    for token in tokens:
        # Remove + suffix if present
        base_token = token.rstrip('+')
        
        if base_token in ['N', 'DET', 'NUM', 'PRON']:
            macro.append('NP')
        elif base_token in ['V', 'PART']:
            macro.append('VP')
        elif base_token in ['A', 'ADV']:
            macro.append('MOD')
        elif base_token == 'P':
            macro.append('P')
        elif base_token == 'C':
            macro.append('CONJ')
        else:
            macro.append(base_token)
    
    # Collapse consecutive identical macro tokens
    if not macro:
        return ''
    
    result = [macro[0]]
    for token in macro[1:]:
        if token != result[-1]:
            result.append(token)
    
    return ' '.join(result)


def analyze_structural_templates(df):
    """Find common structural templates."""
    print("\n" + "=" * 70)
    print("STRUCTURAL TEMPLATES (Macro Patterns)")
    print("=" * 70)
    
    # Create macro structures
    df_copy = df.copy()
    df_copy['collapsed_pattern'] = df_copy['pos_pattern'].apply(
        lambda x: simplify_pos_pattern(x, collapse_repetitions=True)
    )
    df_copy['macro_structure'] = df_copy['collapsed_pattern'].apply(create_macro_structure)
    
    # Count macro structures
    macro_counts = Counter(df_copy['macro_structure'])
    
    print(f"\nTotal unique detailed POS patterns: {len(df['pos_pattern'].unique())}")
    print(f"Total unique macro structures: {len(macro_counts)}")
    print("\nLegend: NP = Noun Phrase, VP = Verb Phrase, MOD = Modifier, P = Preposition")
    
    print("\n" + "-" * 70)
    print("Most Common Headline Structures:")
    print("-" * 70)
    
    for i, (template, count) in enumerate(macro_counts.most_common(12), 1):
        pct = count / len(df) * 100
        
        print(f"\n{i}. {template}")
        print(f"   Frequency: {count} headlines ({pct:.1f}%)")
        
        # Show examples
        examples = df_copy[df_copy['macro_structure'] == template].head(3)
        print(f"   Examples:")
        for _, ex in examples.iterrows():
            print(f"     • {ex['headline']}")
    
    print("\n" + "=" * 70)
    
    # Also show the detailed collapsed patterns for top macro structures
    print("\n" + "=" * 70)
    print("DETAILED BREAKDOWN: Top 3 Macro Structures")
    print("=" * 70)
    
    for i, (macro_template, count) in enumerate(macro_counts.most_common(3), 1):
        print(f"\n{i}. Macro: {macro_template} ({count} headlines)")
        print("   Detailed variations:")
        
        # Get all collapsed patterns for this macro structure
        subset = df_copy[df_copy['macro_structure'] == macro_template]
        detail_counts = Counter(subset['collapsed_pattern'])
        
        for detail_pattern, detail_count in detail_counts.most_common(5):
            print(f"     - {detail_pattern} ({detail_count} times)")
            example = subset[subset['collapsed_pattern'] == detail_pattern].iloc[0]
            print(f"       Ex: {example['headline']}")
    
    print("\n" + "=" * 70)


def analyze_structure_types(df):
    """Categorize headlines by structure type."""
    print("\n" + "=" * 70)
    print("HEADLINE STRUCTURE TYPES")
    print("=" * 70)
    
    df_copy = df.copy()
    df_copy['structure_type'] = df_copy.apply(identify_structure_type, axis=1)
    
    # Count types
    type_counts = df_copy['structure_type'].value_counts()
    
    print("\nHeadline types:")
    for struct_type, count in type_counts.items():
        pct = count / len(df) * 100
        print(f"\n  {struct_type:20}: {count:2} ({pct:4.1f}%)")
        
        # Show examples
        examples = df_copy[df_copy['structure_type'] == struct_type].head(2)
        for _, ex in examples.iterrows():
            print(f"    • {ex['headline'][:65]}")


def analyze_root_words(df):
    """Analyze the root (main) words of headlines."""
    print("\n" + "=" * 70)
    print("ROOT WORD ANALYSIS")
    print("=" * 70)
    
    # Root POS distribution
    root_pos_counts = df['root_pos'].value_counts()
    
    print("\nRoot word part-of-speech:")
    for pos, count in root_pos_counts.items():
        pct = count / len(df) * 100
        print(f"  {pos:10}: {count:2} ({pct:4.1f}%)")
    
    # Most common root verbs
    verb_roots = df[df['root_pos'] == 'VERB']['root'].value_counts()
    if len(verb_roots) > 0:
        print("\nMost common root verbs:")
        for word, count in verb_roots.head(10).items():
            print(f"  {word:15}: {count}")
    
    # Most common root nouns
    noun_roots = df[df['root_pos'].isin(['NOUN', 'PROPN'])]['root'].value_counts()
    if len(noun_roots) > 0:
        print("\nMost common root nouns:")
        for word, count in noun_roots.head(10).items():
            print(f"  {word:15}: {count}")


def analyze_building_blocks(df):
    """Analyze common structural building blocks regardless of full pattern."""
    print("\n" + "=" * 70)
    print("STRUCTURAL BUILDING BLOCKS")
    print("=" * 70)
    print("\n(Patterns that appear across different headlines)")
    
    df_copy = df.copy()
    df_copy['collapsed_pattern'] = df_copy['pos_pattern'].apply(
        lambda x: simplify_pos_pattern(x, collapse_repetitions=True)
    )
    df_copy['macro_structure'] = df_copy['collapsed_pattern'].apply(create_macro_structure)
    
    # Analyze opening patterns (first 2-3 elements)
    print("\nOpening structures (how headlines begin):")
    opening_counts = defaultdict(int)
    for macro in df_copy['macro_structure']:
        parts = macro.split()
        if len(parts) >= 2:
            opening = ' '.join(parts[:2])
            opening_counts[opening] += 1
    
    for opening, count in sorted(opening_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
        pct = count / len(df) * 100
        print(f"  {opening:20} : {count:2} headlines ({pct:4.1f}%)")
        examples = df_copy[df_copy['macro_structure'].str.startswith(opening)].head(2)
        for _, ex in examples.iterrows():
            print(f"    → {ex['headline'][:60]}...")
    
    # Analyze ending patterns
    print("\nEnding structures (how headlines conclude):")
    ending_counts = defaultdict(int)
    for macro in df_copy['macro_structure']:
        parts = macro.split()
        if len(parts) >= 2:
            ending = ' '.join(parts[-2:])
            ending_counts[ending] += 1
    
    for ending, count in sorted(ending_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
        pct = count / len(df) * 100
        print(f"  {ending:20} : {count:2} headlines ({pct:4.1f}%)")
    
    # Analyze core patterns (NP-VP relationships)
    print("\nCore NP-VP patterns:")
    
    # Simple NP VP structure
    simple_subj_verb = sum(1 for m in df_copy['macro_structure'] if m.startswith('NP VP'))
    print(f"  NP VP (Subject-Verb opening): {simple_subj_verb} ({simple_subj_verb/len(df)*100:.1f}%)")
    
    # VP first (verb-led)
    verb_led = sum(1 for m in df_copy['macro_structure'] if m.startswith('VP'))
    print(f"  VP-led (action-first):        {verb_led} ({verb_led/len(df)*100:.1f}%)")
    
    # MOD first (modifier-led)
    mod_led = sum(1 for m in df_copy['macro_structure'] if m.startswith('MOD'))
    print(f"  MOD-led (descriptor-first):   {mod_led} ({mod_led/len(df)*100:.1f}%)")
    
    # Contains VP P NP (verb + prep phrase)
    verb_prep = sum(1 for m in df_copy['macro_structure'] if 'VP P NP' in m)
    print(f"  Contains VP P NP:             {verb_prep} ({verb_prep/len(df)*100:.1f}%)")
    
    # Analyze complexity (number of phrases)
    print("\nStructural complexity:")
    complexities = []
    for macro in df_copy['macro_structure']:
        # Count number of NP + VP units
        num_phrases = macro.count('NP') + macro.count('VP')
        complexities.append(num_phrases)
    
    avg_complexity = sum(complexities) / len(complexities)
    print(f"  Average NP+VP units per headline: {avg_complexity:.1f}")
    
    simple_headlines = sum(1 for c in complexities if c <= 3)
    complex_headlines = sum(1 for c in complexities if c >= 5)
    print(f"  Simple structures (≤3 units): {simple_headlines} ({simple_headlines/len(df)*100:.1f}%)")
    print(f"  Complex structures (≥5 units): {complex_headlines} ({complex_headlines/len(df)*100:.1f}%)")


def analyze_dependency_templates(df):
    """Analyze dependency relation patterns - more stable than POS."""
    print("\n" + "=" * 70)
    print("DEPENDENCY STRUCTURE TEMPLATES")
    print("=" * 70)
    print("\n(Dependency relations are more stable than POS patterns)")
    
    # Extract simplified dependency patterns (non-punctuation)
    simplified_deps = []
    for _, row in df.iterrows():
        deps = []
        for token in row['tokens']:
            if not token['is_punct']:
                deps.append(token['dep'])
        simplified_deps.append(' '.join(deps))
    
    df_copy = df.copy()
    df_copy['dep_simple'] = simplified_deps
    
    # Count dependency patterns
    dep_counts = Counter(df_copy['dep_simple'])
    
    print(f"\nTotal unique dependency patterns: {len(dep_counts)}")
    print("\nMost common dependency structures:")
    
    for i, (dep_pattern, count) in enumerate(dep_counts.most_common(10), 1):
        pct = count / len(df) * 100
        print(f"\n{i}. [{count:2} headlines, {pct:4.1f}%]")
        print(f"   Pattern: {dep_pattern[:80]}...")
        
        # Show examples
        examples = df_copy[df_copy['dep_simple'] == dep_pattern].head(2)
        for _, ex in examples.iterrows():
            print(f"     • {ex['headline'][:65]}...")
    
    # Analyze core dependency structures (first 3 non-punct deps)
    print("\n" + "-" * 70)
    print("Core dependency openings (first 3 relations):")
    print("-" * 70)
    
    core_patterns = defaultdict(int)
    for dep_simple in simplified_deps:
        parts = dep_simple.split()
        if len(parts) >= 3:
            core = ' '.join(parts[:3])
            core_patterns[core] += 1
    
    for core, count in sorted(core_patterns.items(), key=lambda x: x[1], reverse=True)[:12]:
        pct = count / len(df) * 100
        print(f"  {core:30} : {count:2} ({pct:4.1f}%)")


def analyze_named_entities(df):
    """Analyze named entity types in headlines."""
    print("\n" + "=" * 70)
    print("NAMED ENTITY ANALYSIS")
    print("=" * 70)
    
    # Count entity types
    entity_type_counts = defaultdict(int)
    headlines_with_entities = 0
    total_entities = 0
    
    for _, row in df.iterrows():
        entities = row.get('entities', [])
        if entities:
            headlines_with_entities += 1
        for ent in entities:
            entity_type_counts[ent['label']] += 1
            total_entities += 1
    
    print(f"\nHeadlines with named entities: {headlines_with_entities} / {len(df)} ({headlines_with_entities/len(df)*100:.1f}%)")
    print(f"Total named entities found: {total_entities}")
    print(f"Average entities per headline: {total_entities/len(df):.1f}")
    
    print("\nNamed entity type distribution:")
    entity_labels = {
        'PERSON': 'People',
        'ORG': 'Organizations',
        'GPE': 'Geo-political entities (countries, cities)',
        'DATE': 'Dates',
        'NORP': 'Nationalities/groups',
        'CARDINAL': 'Numbers',
        'ORDINAL': 'Ordinals (1st, 2nd)',
        'LOC': 'Locations',
        'EVENT': 'Events',
        'MONEY': 'Money amounts',
    }
    
    for ent_type, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_entities * 100
        description = entity_labels.get(ent_type, ent_type)
        print(f"  {ent_type:12} ({description:40}): {count:3} ({pct:5.1f}%)")
    
    # Show examples
    print("\nExample headlines by entity type:")
    for ent_type in list(entity_type_counts.keys())[:5]:
        for _, row in df.iterrows():
            entities = row.get('entities', [])
            if any(e['label'] == ent_type for e in entities):
                ent_texts = [e['text'] for e in entities if e['label'] == ent_type]
                print(f"  {ent_type:10} → {row['headline'][:55]}...")
                print(f"             Entities: {', '.join(ent_texts[:3])}")
                break


def analyze_compression_ratio(df):
    """Analyze content word vs function word ratio."""
    print("\n" + "=" * 70)
    print("HEADLINE COMPRESSION RATIO")
    print("=" * 70)
    
    content_pos = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM'}
    function_pos = {'DET', 'PRON', 'ADP', 'AUX', 'CCONJ', 'SCONJ', 'PART'}
    
    compression_ratios = []
    
    for _, row in df.iterrows():
        content_count = 0
        function_count = 0
        
        for token in row['tokens']:
            if token['is_punct']:
                continue
            if token['pos'] in content_pos:
                content_count += 1
            elif token['pos'] in function_pos:
                function_count += 1
        
        total_non_punct = content_count + function_count
        if total_non_punct > 0:
            ratio = content_count / total_non_punct
            compression_ratios.append(ratio)
        else:
            compression_ratios.append(0)
    
    df_copy = df.copy()
    df_copy['compression_ratio'] = compression_ratios
    
    avg_ratio = sum(compression_ratios) / len(compression_ratios)
    
    print(f"\nContent word ratio (higher = more information-dense):")
    print(f"  Average: {avg_ratio*100:.1f}% content words")
    print(f"  Range: {min(compression_ratios)*100:.1f}% - {max(compression_ratios)*100:.1f}%")
    
    # Distribution
    high_compression = sum(1 for r in compression_ratios if r >= 0.7)
    medium_compression = sum(1 for r in compression_ratios if 0.5 <= r < 0.7)
    low_compression = sum(1 for r in compression_ratios if r < 0.5)
    
    print(f"\nCompression distribution:")
    print(f"  High (≥70% content):   {high_compression} headlines ({high_compression/len(df)*100:.1f}%)")
    print(f"  Medium (50-69% content): {medium_compression} headlines ({medium_compression/len(df)*100:.1f}%)")
    print(f"  Low (<50% content):    {low_compression} headlines ({low_compression/len(df)*100:.1f}%)")
    
    # Show most compressed headlines
    print("\nMost information-dense headlines:")
    top_compressed = df_copy.nlargest(3, 'compression_ratio')
    for _, row in top_compressed.iterrows():
        print(f"  ({row['compression_ratio']*100:.0f}% content) {row['headline']}")


def detect_passive_voice(tokens):
    """Detect passive voice construction (AUX + VBN)."""
    for i in range(len(tokens) - 1):
        if tokens[i]['pos'] == 'AUX' and tokens[i+1]['tag'] == 'VBN':
            return True
    return False


def analyze_voice(df):
    """Analyze passive vs active voice in headlines."""
    print("\n" + "=" * 70)
    print("PASSIVE VS ACTIVE VOICE")
    print("=" * 70)
    
    passive_count = 0
    passive_examples = []
    
    for _, row in df.iterrows():
        if detect_passive_voice(row['tokens']):
            passive_count += 1
            if len(passive_examples) < 5:
                passive_examples.append(row['headline'])
    
    active_count = len(df) - passive_count
    
    print(f"\nVoice distribution:")
    print(f"  Active voice:  {active_count} headlines ({active_count/len(df)*100:.1f}%)")
    print(f"  Passive voice: {passive_count} headlines ({passive_count/len(df)*100:.1f}%)")
    
    if passive_examples:
        print("\nPassive voice examples:")
        for ex in passive_examples:
            print(f"  • {ex}")
    
    print("\nInsight: News headlines strongly prefer active voice for directness and impact.")


def analyze_information_order(df):
    """Analyze whether headlines lead with actor, action, or context."""
    print("\n" + "=" * 70)
    print("INFORMATION ORDER (Who/Action/Context)")
    print("=" * 70)
    
    actor_first = 0
    action_first = 0
    context_first = 0
    other = 0
    
    actor_examples = []
    action_examples = []
    context_examples = []
    
    for _, row in df.iterrows():
        tokens = row['tokens']
        if not tokens:
            other += 1
            continue
        
        # Remove punctuation
        non_punct_tokens = [t for t in tokens if not t['is_punct']]
        if not non_punct_tokens:
            other += 1
            continue
        
        first_token = non_punct_tokens[0]
        
        # Actor-first: starts with NOUN, PROPN, PRON (subject)
        if first_token['pos'] in ['NOUN', 'PROPN', 'PRON']:
            actor_first += 1
            if len(actor_examples) < 3:
                actor_examples.append(row['headline'])
        
        # Action-first: starts with VERB, AUX
        elif first_token['pos'] in ['VERB', 'AUX']:
            action_first += 1
            if len(action_examples) < 3:
                action_examples.append(row['headline'])
        
        # Context-first: starts with ADV, ADJ, NUM, DET
        elif first_token['pos'] in ['ADV', 'ADJ', 'NUM', 'DET']:
            context_first += 1
            if len(context_examples) < 3:
                context_examples.append(row['headline'])
        
        else:
            other += 1
    
    print(f"\nHeadline opening strategy:")
    print(f"  Actor-first:   {actor_first} headlines ({actor_first/len(df)*100:.1f}%)")
    print(f"  Action-first:  {action_first} headlines ({action_first/len(df)*100:.1f}%)")
    print(f"  Context-first: {context_first} headlines ({context_first/len(df)*100:.1f}%)")
    print(f"  Other:         {other} headlines ({other/len(df)*100:.1f}%)")
    
    print("\nExamples:")
    if actor_examples:
        print("\n  Actor-first (Who):")
        for ex in actor_examples:
            print(f"    • {ex}")
    
    if action_examples:
        print("\n  Action-first (What happened):")
        for ex in action_examples:
            print(f"    • {ex}")
    
    if context_examples:
        print("\n  Context-first (When/Where/How):")
        for ex in context_examples:
            print(f"    • {ex}")


def _content_word_ratio(tokens):
    """Compute content word ratio for a token list."""
    content_pos = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM'}
    function_pos = {'DET', 'PRON', 'ADP', 'AUX', 'CCONJ', 'SCONJ', 'PART'}

    content_count = 0
    function_count = 0

    for token in tokens:
        if token.get('is_punct'):
            continue
        pos = token.get('pos')
        if pos in content_pos:
            content_count += 1
        elif pos in function_pos:
            function_count += 1

    total = content_count + function_count
    return (content_count / total) if total > 0 else 0


def _information_opening_counts(df):
    """Return opening strategy counts for a dataframe subset."""
    counts = {'actor_first': 0, 'action_first': 0, 'context_first': 0, 'other': 0}

    for _, row in df.iterrows():
        tokens = row.get('tokens', [])
        non_punct_tokens = [t for t in tokens if not t.get('is_punct', False)]

        if not non_punct_tokens:
            counts['other'] += 1
            continue

        first_pos = non_punct_tokens[0].get('pos', '')
        if first_pos in ['NOUN', 'PROPN', 'PRON']:
            counts['actor_first'] += 1
        elif first_pos in ['VERB', 'AUX']:
            counts['action_first'] += 1
        elif first_pos in ['ADV', 'ADJ', 'NUM', 'DET']:
            counts['context_first'] += 1
        else:
            counts['other'] += 1

    return counts


def _top_entity_type(df):
    """Return most common named entity label for subset."""
    entity_counts = Counter()
    for _, row in df.iterrows():
        for ent in row.get('entities', []):
            label = ent.get('label')
            if label:
                entity_counts[label] += 1

    if not entity_counts:
        return 'None', 0
    return entity_counts.most_common(1)[0]


def analyze_domestic_vs_world(df):
    """Compare structural patterns between domestic and world headlines."""
    print("\n" + "=" * 70)
    print("DOMESTIC VS WORLD COMPARISON")
    print("=" * 70)

    if 'category' not in df.columns:
        print("\nNo 'category' field found in parsed data.")
        print("Run collect_headlines.py and parse_headlines.py again to include categories.")
        return

    df_copy = df.copy()
    df_copy['category'] = df_copy['category'].astype(str).str.strip().str.lower()

    domestic_df = df_copy[df_copy['category'] == 'domestic']
    world_df = df_copy[df_copy['category'] == 'world']

    print(f"\nDomestic headlines: {len(domestic_df)}")
    print(f"World headlines:    {len(world_df)}")

    if len(domestic_df) == 0 or len(world_df) == 0:
        print("\nNeed both domestic and world headlines to run this comparison.")
        return

    def category_metrics(subset):
        content_ratios = [_content_word_ratio(row.get('tokens', [])) for _, row in subset.iterrows()]
        passive_count = sum(1 for _, row in subset.iterrows() if detect_passive_voice(row.get('tokens', [])))
        opening = _information_opening_counts(subset)
        top_entity, top_entity_count = _top_entity_type(subset)

        collapsed = subset['pos_pattern'].apply(lambda x: simplify_pos_pattern(x, collapse_repetitions=True))
        macro = collapsed.apply(create_macro_structure)
        top_macros = Counter(macro).most_common(3)

        return {
            'avg_tokens': subset['num_tokens'].mean(),
            'avg_content_ratio': sum(content_ratios) / len(content_ratios),
            'passive_pct': (passive_count / len(subset)) * 100,
            'actor_first_pct': (opening['actor_first'] / len(subset)) * 100,
            'action_first_pct': (opening['action_first'] / len(subset)) * 100,
            'context_first_pct': (opening['context_first'] / len(subset)) * 100,
            'avg_entities': sum(len(row.get('entities', [])) for _, row in subset.iterrows()) / len(subset),
            'top_entity': top_entity,
            'top_entity_count': top_entity_count,
            'top_macros': top_macros,
        }

    domestic = category_metrics(domestic_df)
    world = category_metrics(world_df)

    print("\nCore metrics:")
    print(f"  Avg length (tokens):    Domestic {domestic['avg_tokens']:.1f} | World {world['avg_tokens']:.1f}")
    print(f"  Content-word ratio:     Domestic {domestic['avg_content_ratio']*100:.1f}% | World {world['avg_content_ratio']*100:.1f}%")
    print(f"  Passive voice:          Domestic {domestic['passive_pct']:.1f}% | World {world['passive_pct']:.1f}%")
    print(f"  Actor-first openings:   Domestic {domestic['actor_first_pct']:.1f}% | World {world['actor_first_pct']:.1f}%")
    print(f"  Action-first openings:  Domestic {domestic['action_first_pct']:.1f}% | World {world['action_first_pct']:.1f}%")
    print(f"  Context-first openings: Domestic {domestic['context_first_pct']:.1f}% | World {world['context_first_pct']:.1f}%")
    print(f"  Avg entities/headline:  Domestic {domestic['avg_entities']:.2f} | World {world['avg_entities']:.2f}")

    print("\nTop named entity type:")
    print(f"  Domestic: {domestic['top_entity']} ({domestic['top_entity_count']} total)")
    print(f"  World:    {world['top_entity']} ({world['top_entity_count']} total)")

    print("\nTop 3 macro structures by category:")
    print("  Domestic:")
    for template, count in domestic['top_macros']:
        print(f"    • {template} ({count})")
    print("  World:")
    for template, count in world['top_macros']:
        print(f"    • {template} ({count})")

    print("\nLargest directional differences:")
    differences = [
        ("Length", abs(domestic['avg_tokens'] - world['avg_tokens']), "tokens"),
        ("Content ratio", abs(domestic['avg_content_ratio'] - world['avg_content_ratio']) * 100, "pp"),
        ("Passive voice", abs(domestic['passive_pct'] - world['passive_pct']), "pp"),
        ("Actor-first", abs(domestic['actor_first_pct'] - world['actor_first_pct']), "pp"),
        ("Context-first", abs(domestic['context_first_pct'] - world['context_first_pct']), "pp"),
    ]

    for metric, diff, unit in sorted(differences, key=lambda x: x[1], reverse=True)[:3]:
        print(f"  • {metric}: {diff:.1f} {unit}")


def analyze_model_label_distribution(df):
    """Run classifier on parsed headlines and summarize label patterns."""
    print("\n" + "=" * 70)
    print("MODEL LABEL DISTRIBUTION")
    print("=" * 70)

    if classify_dataframe is None:
        print("\nModel classifier import failed.")
        print("Run from project root so scripts.model can be imported.")
        return

    classified = classify_dataframe(df.copy())
    total = len(classified)

    print(f"\nClassified headlines: {total}")
    counts = Counter(classified['predicted_structure'])
    print("\nOverall structural labels:")
    for label in (LABELS or sorted(counts.keys())):
        count = counts.get(label, 0)
        pct = (count / total * 100) if total else 0
        print(f"  {label:22}: {count:3} ({pct:4.1f}%)")

    # Simple, publishable blanket statements from model outputs.
    top_label, top_count = counts.most_common(1)[0]
    print("\nModel-backed summary statements:")
    print(f"  • {top_count/total*100:.1f}% of headlines are classified as {top_label}.")
    print(f"  • {counts.get('question_form', 0)/total*100:.1f}% are question-form headlines.")
    print(f"  • {counts.get('passive_clause', 0)/total*100:.1f}% are passive-clause headlines.")

    if 'category' not in classified.columns:
        return

    classified['category'] = classified['category'].astype(str).str.strip().str.lower()
    dom = classified[classified['category'] == 'domestic']
    wor = classified[classified['category'] == 'world']
    if dom.empty or wor.empty:
        return

    print("\nDomestic vs World by model labels:")
    diffs = []
    for label in (LABELS or sorted(counts.keys())):
        dom_pct = 100 * (dom['predicted_structure'] == label).sum() / len(dom)
        wor_pct = 100 * (wor['predicted_structure'] == label).sum() / len(wor)
        diffs.append((abs(dom_pct - wor_pct), label, dom_pct, wor_pct))
        print(f"  {label:22}: Domestic {dom_pct:5.1f}% | World {wor_pct:5.1f}%")

    top_diffs = sorted(diffs, reverse=True)[:3]
    print("\nLargest model label gaps (domestic vs world):")
    for diff, label, dom_pct, wor_pct in top_diffs:
        print(f"  • {label}: {diff:.1f} pp ({dom_pct:.1f}% vs {wor_pct:.1f}%)")


def analyze_style_profile_story(df):
    """Produce richer story-level outputs from multi-signal style profiles."""
    print("\n" + "=" * 70)
    print("STYLE PROFILE STORY LAYER")
    print("=" * 70)

    if profile_dataframe is None:
        print("\nStyle profiler import failed.")
        return

    profiled = profile_dataframe(df.copy())
    total = len(profiled)
    if total == 0:
        print("\nNo rows available.")
        return

    def pct(mask):
        return 100 * mask.sum() / total

    print("\nCore style dimensions:")
    for col in ["lead_frame", "agency_style", "density_band", "rhetorical_mode"]:
        print(f"  {col}:")
        counts = profiled[col].value_counts()
        for label, count in counts.items():
            print(f"    - {label:24} {count:3} ({count/total*100:4.1f}%)")

    print("\nNarrative-ready blanket statements:")
    print(f"  • {pct(profiled['lead_frame'] == 'actor_first'):.1f}% of headlines lead with an actor.")
    print(f"  • {pct(profiled['density_band'] == 'high_density'):.1f}% are high-density compressed headlines.")
    print(
        f"  • {pct(profiled['agency_style'] == 'passive_agent_omitted'):.1f}% use passive framing "
        "without naming an explicit agent."
    )
    print(f"  • {pct(profiled['rhetorical_mode'] == 'straight_report'):.1f}% are straight-report rhetorical mode.")

    top_signatures = Counter(profiled["style_signature"]).most_common(5)
    print("\nMost common style signatures:")
    for sig, count in top_signatures:
        print(f"  • {sig} ({count}, {count/total*100:.1f}%)")

    if "category" in profiled.columns:
        profiled["category"] = profiled["category"].astype(str).str.strip().str.lower()
        dom = profiled[profiled["category"] == "domestic"]
        wor = profiled[profiled["category"] == "world"]
        if not dom.empty and not wor.empty:
            print("\nDomestic vs World (style dimensions):")
            for col in ["lead_frame", "density_band", "rhetorical_mode"]:
                dom_top = dom[col].value_counts(normalize=True).idxmax()
                dom_pct = dom[col].value_counts(normalize=True).max() * 100
                wor_top = wor[col].value_counts(normalize=True).idxmax()
                wor_pct = wor[col].value_counts(normalize=True).max() * 100
                print(f"  • {col}: Domestic top={dom_top} ({dom_pct:.1f}%), World top={wor_top} ({wor_pct:.1f}%)")


def generate_insights(df):
    """Generate key insights about headline structures."""
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    insights = []
    
    # Insight 1: Brevity
    avg_length = df['num_tokens'].mean()
    insights.append(f"Headlines are concise, averaging {avg_length:.1f} words, optimized for quick scanning.")
    
    # Insight 2: Verb usage
    has_verb = df['pos_pattern'].str.contains('VERB|AUX').sum()
    verb_pct = has_verb / len(df) * 100
    if verb_pct > 70:
        insights.append(f"{verb_pct:.0f}% of headlines use verbs, creating dynamic, action-oriented language.")
    
    # Insight 3: Proper nouns
    propn_counts = [row['pos_pattern'].split().count('PROPN') for _, row in df.iterrows()]
    avg_propn = sum(propn_counts) / len(propn_counts)
    insights.append(f"Headlines average {avg_propn:.1f} proper nouns, anchoring stories to specific people/places.")
    
    # Insight 4: Structure diversity
    unique_patterns = len(df['pos_pattern'].unique())
    if unique_patterns == len(df):
        insights.append(f"Each headline uses a unique grammatical structure, showing high linguistic diversity.")
    else:
        insights.append(f"Headlines show moderate structural variety with {unique_patterns} unique patterns across {len(df)} headlines.")
    
    # Insight 5: Root word analysis
    root_pos_counts = df['root_pos'].value_counts()
    top_root_pos = root_pos_counts.index[0] if len(root_pos_counts) > 0 else None
    if top_root_pos:
        count = root_pos_counts.iloc[0]
        pct = count / len(df) * 100
        pos_name = {'VERB': 'verbs', 'NOUN': 'nouns', 'PROPN': 'proper nouns'}.get(top_root_pos, top_root_pos)
        insights.append(f"Most headlines ({pct:.0f}%) center around {pos_name}, defining the main focus.")
    
    print()
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")


def main():
    """Main analysis function."""
    print("=" * 70)
    print("NEWS HEADLINE STRUCTURAL ANALYSIS")
    print("=" * 70)
    print()
    
    # Load data
    df = load_latest_parsed_data()
    if df is None:
        return
    
    print(f"Loaded {len(df)} headlines\n")
    
    # Run analyses
    analyze_structure_types(df)
    analyze_dependency_templates(df)
    analyze_named_entities(df)
    analyze_domestic_vs_world(df)
    analyze_model_label_distribution(df)
    analyze_style_profile_story(df)
    analyze_information_order(df)
    analyze_compression_ratio(df)
    analyze_voice(df)
    analyze_structural_templates(df)
    analyze_building_blocks(df)
    analyze_length_patterns(df)
    analyze_verb_patterns(df)
    analyze_proper_noun_density(df)
    analyze_root_words(df)
    
    # Generate insights
    generate_insights(df)
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
