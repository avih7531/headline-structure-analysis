2# Structural Patterns in News Headlines

An NLP project analyzing the grammatical structure of news headlines to determine which syntactic patterns occur most frequently.

## Project Overview

This project analyzes how news organizations structure headlines for maximum impact and information density. Using RSS feeds from Google News, we collect headlines across multiple sources and topics, parse them with spaCy, and identify structural templates and linguistic patterns.

## Key Findings

### Information Density
- Headlines achieve **76.3% content word ratio** (vs ~40% in normal prose)
- **72.9%** of headlines have ≥70% content words
- Ruthlessly eliminate function words while maintaining clarity

### Voice & Agency
- **95.7% active voice**: creates immediacy and directness
- Only **4.3% passive voice**, which used strategically when actor is de-emphasized. 

### Information Order
- **81.4% Actor-first** (WHO): "Trump delays...", "FBI hunts..."
- **14.3% Context-first** (WHEN/WHERE): "Early strikes...", "In Arizona..."
- **2.9% Action-first** (WHAT): "Growing evidence..."

**Key Observation**: Journalism follows strict **WHO → WHAT → WHERE** hierarchy

### Named Entity Distribution
- **94.3%** of headlines contain named entities (avg 2.3 per headline)
- **GPE (38.4%)**: Countries, cities (Iran, U.S., Alabama)
- **ORG (24.4%)**: Organizations (FBI, GOP, Supreme Court)
- **PERSON (12.8%)**: People (Trump, Burton, Epstein)

### Structural Patterns
- **61.4%** use Subject-Verb opening (NP VP)
- **41.4%** end with prepositional phrases (P NP) for context
- Average **13.9 words** with **6.4 phrase units** for dense information packing

## Installation

### Prerequisites
- Python 3.12.x (managed via pyenv)
- pyenv installed

### Setup

```bash
# Clone the repository
git clone <https://github.com/avih7531/headline-structure-analysis>
cd headline-structure-analysis

# Python version will auto-switch to 3.12.8 via .python-version, activate virtual environment
python -m venv venv
source venv/bin/activate

# If using Git Bash on Windows, replace the second command to...
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage

### 1. Collect Headlines

```bash
python collect_headlines.py
```

Fetches headlines from Google News RSS feeds (Domestic & World categories). Headlines are cleaned, deduplicated, and saved to `data/headlines.json`.

### 2. Parse Headlines

```bash
python parse_headlines.py
```

Parses headlines using spaCy to extract:
- POS tags and patterns
- Dependency structures
- Named entities
- Noun chunks
- Root words

Only parses new (unparsed) headlines. Results saved to `data/headlines_parsed.json`.

### 3. Analyze Structures

```bash
python analyze_structure.py
```

Generates comprehensive analysis including:
- **Dependency structure templates** (most stable patterns)
- **Named entity distributions**
- **Information order analysis** (actor/action/context)
- **Compression ratios** (content vs function words)
- **Passive vs active voice detection**
- **Structural templates and building blocks**

## Project Structure

```
.
├── collect_headlines.py      # RSS feed collection & cleaning
├── parse_headlines.py         # spaCy NLP parsing
├── analyze_structure.py       # Structural analysis & insights
├── requirements.txt           # Python dependencies
├── .python-version           # Python 3.12.8 (via pyenv)
└── data/
    ├── headlines.json        # Raw collected headlines (accumulating)
    └── headlines_parsed.json # Parsed headlines with NLP features (accumulating)
```

## Data Sources

- **Domestic News**: Google News US Domestic RSS
- **World News**: Google News World RSS

Headlines are automatically cleaned to:
- Remove source attribution (e.g., "- CNN", "- BBC")
- Remove metadata (e.g., "- Live Updates")
- Fix unicode characters (smart quotes → regular quotes)
- Deduplicate across collections

## Methodology

### Data Collection
- RSS feeds parsed using `feedparser`
- Automatic cleaning and normalization
- Incremental collection with deduplication

### Preprocessing
- Tokenization and POS tagging via spaCy
- Dependency parsing
- Named Entity Recognition (NER)
- Lemmatization

### Structure Extraction
- **POS Patterns**: Detailed grammatical sequences
- **Dependency Templates**: Syntactic relationship patterns
- **Macro Structures**: Simplified NP/VP/MOD patterns
- **Building Blocks**: Common opening/ending structures

### Analysis
- Frequency distributions
- Statistical measures (means, medians, distributions)
- Compression ratios
- Voice detection (active/passive)
- Entity type analysis

## Why This Project is Doable

-  **Data is Easy to Obtain** - RSS feeds provide unlimited headlines
-  **Simplistic Dataset** - 70 headlines yield meaningful patterns
-  **Automatic Annotation** - Fully automated pipeline
-  **Quick Iteration** - Collect → Parse → Analyze in minutes
-  **Incremental Growth** - Dataset automatically grows 

## Key Insights

1. **Brevity with Density**: Average 13.9 words but 6.4 phrase units - packed information
2. **Present Tense Preference**: 42 present tense verbs vs 10 past tense - creates immediacy
3. **Verb-Centered**: 67% of headlines center on verbs (not nouns) for action/impact
4. **Proper Noun Heavy**: 25% of tokens are proper nouns - specificity and credibility
5. **Standardized Opening**: 61% use NP VP structure - journalism's default template

## Future Work/Next Steps

- Collect larger dataset over multiple days/weeks
- Compare structural patterns across news categories
- Analyze temporal trends in headline structures
- Compare domestic vs world news structures
- Study correlation between structure and engagement

## Technologies

- **Python 3.12.8** (via pyenv)
- **spaCy 3.7+** - NLP parsing and analysis
- **pandas** - Data manipulation
- **feedparser** - RSS feed parsing

## Author

NLP Course Project - Spring 2026
Group 5: Victor Derani, Avi Herman, Kylie Lin


## License

MIT
