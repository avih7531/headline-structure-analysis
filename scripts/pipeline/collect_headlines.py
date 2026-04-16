"""
Collect and persist headline data from configured RSS feeds.

The collector merges newly fetched headlines into ``data/headlines.json``,
deduplicating by ``headline`` + ``source`` to keep corpus growth predictable
across repeated runs.
"""

import feedparser
import pandas as pd
from datetime import datetime
import os
from typing import Dict, List, Tuple


# RSS Feed URLs
DOMESTIC_RSS_URL = "https://news.google.com/rss/topics/CAAqIggKIhxDQkFTRHdvSkwyMHZNRGxqTjNjd0VnSmxiaWdBUAE?hl=en-US&gl=US&ceid=US%3Aen"
WORLD_RSS_URL = "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US%3Aen"


def clean_headline(title: str) -> Tuple[str, str]:
    """
    Clean headline by removing source attribution and fixing unicode.
    
    Args:
        title: Raw headline title
    
    Returns:
        Tuple of (cleaned_headline, source)
    """
    import re
    
    # First, fix unicode issues and escape sequences
    # Replace smart quotes with regular quotes
    title = title.replace('\u2018', "'")  # left single quote
    title = title.replace('\u2019', "'")  # right single quote
    title = title.replace('\u201c', '"')  # left double quote
    title = title.replace('\u201d', '"')  # right double quote
    title = title.replace('\u2013', '-')  # en dash
    title = title.replace('\u2014', '--') # em dash
    
    # Fix escaped quotes from JSON/HTML
    title = title.replace('\\"', '"')
    title = title.replace("\\'", "'")
    
    # Split by dashes/pipes to handle multiple segments
    segments = re.split(r'\s*[-–—|]\s*', title)
    
    # Metadata keywords to filter out
    metadata_keywords = ['live updates', 'live update', 'breaking', 'breaking news', 
                        'update', 'updates', 'latest', 'developing']
    
    # Filter out metadata segments
    cleaned_segments = []
    source = 'Unknown'
    
    for seg in segments:
        seg_lower = seg.lower().strip()
        # Skip empty segments
        if not seg_lower:
            continue
        # Skip metadata
        if seg_lower in metadata_keywords:
            continue
        # Last segment is likely the source if it's capitalized and short-ish
        if seg == segments[-1] and len(seg.split()) <= 4 and seg[0].isupper():
            source = seg.strip()
        else:
            cleaned_segments.append(seg.strip())
    
    # Rejoin the main headline
    headline = ' '.join(cleaned_segments) if cleaned_segments else title.strip()
    
    return headline, source


def fetch_headlines_from_feed(feed_url: str, category: str) -> List[Dict]:
    """
    Fetch headlines from a single RSS feed.
    
    Args:
        feed_url: URL of the RSS feed
        category: Category label (e.g., 'domestic', 'world')
    
    Returns:
        List of dictionaries containing headline data
    """
    print(f"[collect] fetching {category} feed")
    
    feed = feedparser.parse(feed_url)
    headlines = []
    
    for entry in feed.entries:
        # Clean headline and extract source
        cleaned_headline, source = clean_headline(entry.title)
        
        headline_data = {
            'headline': cleaned_headline,
            'category': category,
            'published': entry.get('published', ''),
            'link': entry.get('link', ''),
            'source': source,
            'collected_at': datetime.now().isoformat()
        }
        headlines.append(headline_data)
    
    print(f"[collect] {category}: {len(headlines)} headlines")
    return headlines


def collect_all_headlines() -> pd.DataFrame:
    """
    Collect headlines from all configured RSS feeds.
    
    Returns:
        DataFrame containing all collected headlines
    """
    all_headlines = []
    
    # Fetch from domestic news
    domestic_headlines = fetch_headlines_from_feed(DOMESTIC_RSS_URL, 'domestic')
    all_headlines.extend(domestic_headlines)
    
    # Fetch from world news
    world_headlines = fetch_headlines_from_feed(WORLD_RSS_URL, 'world')
    all_headlines.extend(world_headlines)
    
    # Create DataFrame
    df = pd.DataFrame(all_headlines)
    
    print(f"[collect] total fetched: {len(df)}")
    return df


def save_headlines(df: pd.DataFrame, output_dir: str = 'data') -> Tuple[str, int]:
    """
    Append headlines to the main headlines.json file.
    
    Args:
        df: DataFrame containing new headlines
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = os.path.join(output_dir, 'headlines.json')
    
    # Load existing headlines if file exists
    if os.path.exists(json_path):
        print(f"[save] loading existing dataset: {json_path}")
        existing_df = pd.read_json(json_path)
        print(f"[save] existing rows: {len(existing_df)}")
        
        # Combine with new headlines
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        
        # Remove duplicates based on headline text and source
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['headline', 'source'], keep='first')
        duplicates_removed = before_dedup - len(combined_df)
        
        if duplicates_removed > 0:
            print(f"[save] duplicates removed: {duplicates_removed}")
        
        print(f"[save] merged total rows: {len(combined_df)}")
    else:
        print(f"[save] creating dataset: {json_path}")
        combined_df = df
    
    # Save combined data
    combined_df.to_json(json_path, orient='records', indent=2)
    print(f"[save] wrote {len(df)} new rows (dataset total: {len(combined_df)})")
    
    return json_path, len(df)


def main() -> None:
    """CLI entrypoint for RSS headline collection."""
    print("[start] collect_headlines")
    
    # Collect headlines
    headlines_df = collect_all_headlines()
    
    # Save to file
    _, new_count = save_headlines(headlines_df)
    
    domestic_n = len(headlines_df[headlines_df['category'] == 'domestic'])
    world_n = len(headlines_df[headlines_df['category'] == 'world'])
    print(f"[done] new rows: {new_count} (domestic={domestic_n}, world={world_n})")
    print("[next] run scripts/pipeline/parse_headlines.py")


if __name__ == '__main__':
    main()
