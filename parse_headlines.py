"""
News Headline Parser
Parses headlines using spaCy to extract linguistic features.
"""

import spacy
import pandas as pd
import json
import os
import sys
from collections import Counter


def load_spacy_model():
    """
    Load the spaCy English model with Python 3.14 compatibility.
    
    Returns:
        spaCy nlp object
    """
    # Suppress the pydantic v1 warning for Python 3.14
    import warnings
    warnings.filterwarnings('ignore', message='.*Pydantic V1.*')
    
    try:
        nlp = spacy.load('en_core_web_sm')
        print("✓ Loaded spaCy model: en_core_web_sm")
        return nlp
    except OSError:
        print("spaCy model 'en_core_web_sm' not found!")
        print("Attempting to download...")
        try:
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                         check=True, capture_output=True)
            nlp = spacy.load('en_core_web_sm')
            print("✓ Downloaded and loaded spaCy model")
            return nlp
        except Exception as e:
            print(f"ERROR: Could not download spaCy model: {e}")
            print("\nPlease run manually:")
            print("  python -m spacy download en_core_web_sm")
            raise


def parse_headline(headline_text, nlp):
    """
    Parse a single headline using spaCy.
    
    Args:
        headline_text: The headline text string
        nlp: spaCy nlp object
    
    Returns:
        Dictionary containing parsed information
    """
    doc = nlp(headline_text)
    
    # Extract tokens with POS tags and dependencies
    tokens = []
    for token in doc:
        tokens.append({
            'text': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,      # Universal POS
            'tag': token.tag_,      # Fine-grained POS
            'dep': token.dep_,      # Dependency relation
            'is_stop': token.is_stop,
            'is_punct': token.is_punct
        })
    
    # Extract POS sequence (using universal POS)
    pos_sequence = [token.pos_ for token in doc]
    pos_pattern = ' '.join(pos_sequence)
    
    # Extract dependency structure
    dep_structure = [(token.text, token.dep_, token.head.text) for token in doc]
    
    # Extract dependency relation sequence (from root outward)
    dep_relations = [token.dep_ for token in doc]
    dep_pattern = ' '.join(dep_relations)
    
    # Identify root (main verb or predicate)
    root_tokens = [token for token in doc if token.dep_ == 'ROOT']
    root = root_tokens[0].text if root_tokens else None
    root_pos = root_tokens[0].pos_ if root_tokens else None
    
    # Get noun chunks (useful for analysis)
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    
    # Extract named entities
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_
        })
    
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
        'num_tokens': len(doc)
    }




def analyze_patterns(parsed_df):
    """
    Analyze and display common POS patterns.
    
    Args:
        parsed_df: DataFrame with parsed headlines
    """
    print("\n" + "=" * 60)
    print("PATTERN ANALYSIS")
    print("=" * 60)
    
    # Count POS patterns
    pattern_counts = Counter(parsed_df['pos_pattern'])
    
    print(f"\nTotal unique patterns: {len(pattern_counts)}")
    print(f"\nTop 20 Most Common POS Patterns:")
    print("-" * 60)
    
    for pattern, count in pattern_counts.most_common(20):
        percentage = (count / len(parsed_df)) * 100
        print(f"{count:3d} ({percentage:5.2f}%) | {pattern}")
    
    # Show examples for top patterns
    print("\n" + "=" * 60)
    print("EXAMPLES OF TOP PATTERNS")
    print("=" * 60)
    
    for pattern, count in pattern_counts.most_common(5):
        print(f"\nPattern: {pattern} ({count} occurrences)")
        examples = parsed_df[parsed_df['pos_pattern'] == pattern]['headline'].head(3)
        for ex in examples:
            print(f"  • {ex}")


def save_parsed_data(parsed_df, output_dir='data'):
    """
    Save all parsed headlines.
    
    Args:
        parsed_df: DataFrame with all parsed headlines
        output_dir: Directory to save output
    """
    json_path = os.path.join(output_dir, 'headlines_parsed.json')
    parsed_df.to_json(json_path, orient='records', indent=2)
    print(f"\nSaved parsed data to: {json_path}")
    print(f"Total parsed headlines in dataset: {len(parsed_df)}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("News Headline Parser")
    print("=" * 60)
    print()
    
    headlines_file = 'data/headlines.json'
    parsed_file = 'data/headlines_parsed.json'
    
    if not os.path.exists(headlines_file):
        print(f"ERROR: {headlines_file} not found!")
        print("Please run collect_headlines.py first.")
        return
    
    # Load all headlines
    all_headlines_df = pd.read_json(headlines_file)
    print(f"Found {len(all_headlines_df)} total headlines in dataset")
    
    # Check which headlines have already been parsed
    if os.path.exists(parsed_file):
        parsed_df_existing = pd.read_json(parsed_file)
        print(f"Found {len(parsed_df_existing)} already parsed headlines")
        
        # Find unparsed headlines by comparing headline text
        parsed_headlines = set(parsed_df_existing['headline'])
        unparsed_df = all_headlines_df[~all_headlines_df['headline'].isin(parsed_headlines)]
        
        print(f"Need to parse {len(unparsed_df)} new headlines")
        
        if len(unparsed_df) == 0:
            print("\nAll headlines already parsed! Run collect_headlines.py to get new data.")
            return
    else:
        print("No existing parsed data found. Parsing all headlines...")
        unparsed_df = all_headlines_df
        parsed_df_existing = None
    
    print()
    
    # Parse only the unparsed headlines
    print("\nInitializing spaCy parser...")
    nlp = load_spacy_model()
    
    print(f"\nParsing {len(unparsed_df)} headlines...")
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
    
    print(f"  Processing {len(unparsed_df)}/{len(unparsed_df)}... Done!")
    
    # Create DataFrame of newly parsed data
    new_parsed_df = pd.DataFrame(parsed_data)
    
    # Combine with existing parsed data if any
    if parsed_df_existing is not None:
        combined_parsed_df = pd.concat([parsed_df_existing, new_parsed_df], ignore_index=True)
        print(f"\nCombined with existing data: {len(combined_parsed_df)} total parsed headlines")
    else:
        combined_parsed_df = new_parsed_df
    
    # Analyze patterns (on full dataset)
    analyze_patterns(combined_parsed_df)
    
    # Save results
    save_parsed_data(combined_parsed_df)
    
    print("\n" + "=" * 60)
    print("Parsing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
