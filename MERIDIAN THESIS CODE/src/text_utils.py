# text_utils.py
# Text normalisation, ticker mapping, and near-duplicate detection.

import re
import json
import os
import csv
from collections import defaultdict

from config import DEDUP_THRESHOLD, TICKER_UNIVERSE_FILE

# ── Ticker universe ───────────────────────────────────────────────────────────

def load_ticker_universe(path: str = TICKER_UNIVERSE_FILE) -> dict:
    """Returns {ticker: [alias1, ...]} from the ticker universe CSV."""
    universe = {}
    if not os.path.exists(path):
        # Fallback: small hardcoded set for testing
        universe = {
            "AAPL":  ["apple"],
            "TSLA":  ["tesla"],
            "NVDA":  ["nvidia"],
            "MSFT":  ["microsoft"],
            "AMZN":  ["amazon"],
            "META":  ["meta", "facebook"],
            "GOOGL": ["google", "alphabet"],
            "AMD":   ["amd", "advanced micro"],
            "NFLX":  ["netflix"],
            "SPY":   ["spy", "s&p", "sp500"],
        }
        return universe

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row["ticker"].strip().upper()
            name   = row.get("name", "").strip().lower()
            universe[ticker] = [name] if name else []
    return universe


# Ambiguous tickers that require alias co-occurrence to match
AMBIGUOUS_TICKERS = {"A", "B", "C", "D", "F", "T", "X", "K", "M", "O", "S", "U", "V"}


# ── Text normalisation ────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase, strip URLs/punctuation, collapse whitespace."""
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^\w\s\$]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokenise(text: str) -> list:
    """Splits normalised text into tokens."""
    return text.split()


# ── Ticker matching ───────────────────────────────────────────────────────────

def map_tickers(text: str, universe: dict) -> list:
    """Returns tickers mentioned in the post text."""
    norm = normalise(text)
    tokens = set(tokenise(norm))
    matched = []

    for ticker, aliases in universe.items():
        ticker_lower = ticker.lower()
        dollar_form  = "$" + ticker_lower

        direct_match = (ticker_lower in tokens) or (dollar_form in tokens)
        if not direct_match:
            continue

        # Ambiguous tickers require alias co-occurrence
        if ticker in AMBIGUOUS_TICKERS:
            alias_found = any(alias in norm for alias in aliases)
            if not alias_found:
                continue

        matched.append(ticker)

    return matched


# ── Near-duplicate detection ──────────────────────────────────────────────────

def jaccard_similarity(tokens_a: set, tokens_b: set) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|."""
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = len(tokens_a & tokens_b)
    union        = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def flag_duplicates(posts: list, threshold: float = DEDUP_THRESHOLD) -> list:
    """Marks near-duplicate posts by Jaccard similarity to earlier posts."""
    token_sets = [set(tokenise(normalise(p["text"]))) for p in posts]

    for i, post in enumerate(posts):
        post["is_duplicate"] = False
        for j in range(i):
            if posts[j]["is_duplicate"]:
                continue   # Don't compare against already-flagged duplicates
            sim = jaccard_similarity(token_sets[i], token_sets[j])
            if sim >= threshold:
                post["is_duplicate"] = True
                break

    return posts


# ── Language detection (simple heuristic) ────────────────────────────────────

# Common non-English function words used as a fast language filter.
_NON_ENGLISH_MARKERS = {
    "es": {"que", "con", "del", "los", "las", "una", "por", "para", "como", "pero"},
    "de": {"und", "der", "die", "das", "ist", "ein", "eine", "nicht", "von", "mit"},
    "fr": {"les", "des", "une", "dans", "est", "sur", "qui", "par", "aussi", "mais"},
    "pt": {"que", "com", "uma", "para", "por", "não", "mais", "como", "isso", "seu"},
}
_ENGLISH_THRESHOLD = 2   # If >= this many non-English markers found, flag as non-English


def is_english(text: str) -> bool:
    """Heuristic English detection based on non-English function words."""
    tokens = set(tokenise(normalise(text)))
    for lang, markers in _NON_ENGLISH_MARKERS.items():
        overlap = len(tokens & markers)
        if overlap >= _ENGLISH_THRESHOLD:
            return False
    return True


# ── Full processing pipeline for one day's raw posts ─────────────────────────

def process_raw_posts(raw_posts: list, universe: dict) -> list:
    """Normalise, match tickers, flag duplicates and language for a day's posts."""
    # Step 1: flag duplicates across the full day batch
    raw_posts = flag_duplicates(raw_posts)

    processed = []
    for post in raw_posts:
        norm_text  = normalise(post["text"])
        tickers    = map_tickers(post["text"], universe)
        english    = is_english(post["text"])

        processed.append({
            **post,
            "normalised_text": norm_text,
            "tickers":         tickers,
            "is_english":      english,
        })

    return processed
