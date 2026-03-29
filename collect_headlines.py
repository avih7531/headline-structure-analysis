"""
News Headline Collection Script
Fetches headlines from RSS feeds and saves them for analysis.
"""

import feedparser
import pandas as pd
from datetime import datetime
import json
import os


# RSS Feed URLs
DOMESTIC_RSS_URL = "https://news.google.com/rss/topics/CAAqIggKIhxDQkFTRHdvSkwyMHZNRGxqTjNjd0VnSmxiaWdBUAE?hl=en-US&gl=US&ceid=US%3Aen"
WORLD_RSS_URL = "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US%3Aen"


def clean_headline(title):
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


def fetch_headlines_from_feed(feed_url, category):
    """
    Fetch headlines from a single RSS feed.
    
    Args:
        feed_url: URL of the RSS feed
        category: Category label (e.g., 'domestic', 'world')
    
    Returns:
        List of dictionaries containing headline data
    """
    print(f"Fetching {category} headlines from RSS feed...")
    
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
    
    print(f"  → Collected {len(headlines)} headlines from {category}")
    return headlines


def collect_all_headlines():
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
    
    print(f"\nTotal headlines collected: {len(df)}")
    return df


def save_headlines(df, output_dir='data'):
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
        print(f"Loading existing headlines from {json_path}...")
        existing_df = pd.read_json(json_path)
        print(f"  → Found {len(existing_df)} existing headlines")
        
        # Combine with new headlines
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        
        # Remove duplicates based on headline text and source
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['headline', 'source'], keep='first')
        duplicates_removed = before_dedup - len(combined_df)
        
        if duplicates_removed > 0:
            print(f"  → Removed {duplicates_removed} duplicate headlines")
        
        print(f"  → Total headlines after merge: {len(combined_df)}")
    else:
        print(f"Creating new headlines file: {json_path}")
        combined_df = df
    
    # Save combined data
    combined_df.to_json(json_path, orient='records', indent=2)
    print(f"Saved {len(df)} new headlines (total: {len(combined_df)} in dataset)")
    
    return json_path, len(df)


def main():
    """Main execution function."""
    print("=" * 60)
    print("News Headline Collection Script")
    print("=" * 60)
    print()
    
    # Collect headlines
    headlines_df = collect_all_headlines()
    
    # Display sample
    print("\nSample headlines:")
    print("-" * 60)
    for idx, row in headlines_df.head(10).iterrows():
        print(f"{row['category'].upper():10} | {row['headline']}")
    print("-" * 60)
    
    # Save to file
    print()
    json_path, new_count = save_headlines(headlines_df)
    
    print()
    print("Collection complete!")
    print(f"New headlines collected: {new_count}")
    print(f"  Domestic: {len(headlines_df[headlines_df['category'] == 'domestic'])}")
    print(f"  World: {len(headlines_df[headlines_df['category'] == 'world'])}")
    print(f"\nRun parse_headlines.py to parse new headlines.")


if __name__ == '__main__':
    main()
