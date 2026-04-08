# features.py
# Computes all ticker-day features: tone, attention, credibility, and aggregates.

import math
import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime

from config import (
    TRIM_RATE, FINBERT_MODEL, BURST_WINDOW_MINUTES,
    CRED_ALPHA, CRED_BETA, CRED_GAMMA,
    ATTN_WINDOW, PROCESSED_DATA_DIR, PARAMS_PATH
)

# ── FinBERT (lazy load) ───────────────────────────────────────────────────────

_finbert_pipeline = None

def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )
    return _finbert_pipeline


# ── Loughran-McDonald lexicon (lazy load) ─────────────────────────────────────

_lm_pos = None
_lm_neg = None

def _get_lm_lexicon():
    """Loads Loughran-McDonald positive/negative word lists."""
    global _lm_pos, _lm_neg
    if _lm_pos is None:
        def _load(path):
            if os.path.exists(path):
                with open(path) as f:
                    return {line.strip().lower() for line in f if line.strip()}
            return set()
        _lm_pos = _load("data/lm_positive.txt")
        _lm_neg = _load("data/lm_negative.txt")
    return _lm_pos, _lm_neg


# ── Post-level sentiment ──────────────────────────────────────────────────────

def score_post_finbert(text: str) -> dict:
    """Runs FinBERT on a single post, returns probabilities and sp = P(pos) - P(neg)."""
    pipe = _get_finbert()
    results = pipe(text[:512])[0]   # truncate to model max length
    probs = {r["label"].lower(): r["score"] for r in results}
    sp = probs.get("positive", 0.0) - probs.get("negative", 0.0)
    return {**probs, "sp": sp}


def score_post_lm(text: str) -> float:
    """Loughran-McDonald lexicon score: (pos - neg) / (pos + neg + eps)."""
    lm_pos, lm_neg = _get_lm_lexicon()
    tokens = text.lower().split()
    pos = sum(1 for t in tokens if t in lm_pos)
    neg = sum(1 for t in tokens if t in lm_neg)
    return (pos - neg) / (pos + neg + 1e-9)


# ── Ticker-day aggregation ────────────────────────────────────────────────────

def _trimmed_mean(values: list, trim: float = TRIM_RATE) -> float:
    """Trimmed mean: discard top/bottom `trim` fraction before averaging."""
    if not values:
        return 0.0
    arr = sorted(values)
    k   = int(len(arr) * trim)
    trimmed = arr[k: len(arr) - k] if k > 0 else arr
    return float(np.mean(trimmed)) if trimmed else 0.0


def _herfindahl(posts: list) -> float:
    """Herfindahl-Hirschman author concentration index."""
    total = len(posts)
    if total == 0:
        return 0.0
    counts = Counter(p["author"] for p in posts)
    return sum((c / total) ** 2 for c in counts.values())


def _entropy(probs_list: list) -> float:
    """Average Shannon entropy over sentiment distributions."""
    if not probs_list:
        return math.log(3)   # maximum entropy
    total_entropy = 0.0
    for probs in probs_list:
        h = 0.0
        for c in ["positive", "negative", "neutral"]:
            p = probs.get(c, 1/3) + 1e-9
            h -= p * math.log(p)
        total_entropy += h
    return total_entropy / len(probs_list)


def _burstiness(posts: list, window_minutes: int = BURST_WINDOW_MINUTES) -> float:
    """Fraction of posts in the most active time window."""
    if not posts:
        return 0.0
    total = len(posts)
    if total == 1:
        return 1.0

    timestamps = sorted(p["created_utc"] for p in posts)
    window_sec = window_minutes * 60
    max_in_window = 0

    # Sliding window: count posts within [t, t + window_sec]
    right = 0
    for left in range(total):
        while right < total and timestamps[right] - timestamps[left] <= window_sec:
            right += 1
        max_in_window = max(max_in_window, right - left)

    return max_in_window / total


# ── Attention: rolling z-score ────────────────────────────────────────────────

def compute_attention_shock(volume_series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of post volume (backward-looking only)."""
    mu    = volume_series.shift(1).rolling(window, min_periods=1).mean()
    sigma = volume_series.shift(1).rolling(window, min_periods=1).std().fillna(1.0)
    sigma = sigma.replace(0, 1.0)   # avoid division by zero on flat periods
    return (volume_series - mu) / sigma


# ── Main feature computation ──────────────────────────────────────────────────

def compute_features(processed_posts: list,
                     date_str: str,
                     history_df: pd.DataFrame = None,
                     window: int = ATTN_WINDOW,
                     alpha: float = CRED_ALPHA,
                     beta: float  = CRED_BETA,
                     gamma: float = CRED_GAMMA,
                     skip_finbert: bool = False) -> pd.DataFrame:
    """Computes all ticker-day features from processed posts; returns one row per ticker."""

    # Group posts by ticker (English-only for FinBERT; all for volume counts)
    ticker_posts = defaultdict(list)
    for post in processed_posts:
        for ticker in post["tickers"]:
            ticker_posts[ticker].append(post)

    if not ticker_posts:
        return pd.DataFrame()

    total_posts_day = sum(len(v) for v in ticker_posts.values())
    rows = []

    for ticker, posts in ticker_posts.items():
        n_posts  = len(posts)
        eng_posts = [p for p in posts if p.get("is_english", True)
                     and not p.get("is_duplicate", False)]

        # ── Tone: FinBERT ───────────────────────────────────────────────────────
        sp_scores      = []
        probs_list     = []
        lm_scores      = []
        upvote_weights = []
        upvote_sp      = []

        if not skip_finbert:
            for post in eng_posts:
                try:
                    result = score_post_finbert(post["normalised_text"])
                    sp = result["sp"]
                    sp_scores.append(sp)
                    probs_list.append({
                        "positive": result.get("positive", 0),
                        "negative": result.get("negative", 0),
                        "neutral":  result.get("neutral",  0),
                    })
                    lm_scores.append(score_post_lm(post["normalised_text"]))
                    w = math.sqrt(max(0, post.get("score", 0)) + 1)
                    upvote_weights.append(w)
                    upvote_sp.append(sp * w)
                except Exception:
                    continue   # Skip posts that fail inference; don't crash the pipeline
        else:
            for post in eng_posts:
                lm_scores.append(score_post_lm(post["normalised_text"]))

        tone_finbert         = _trimmed_mean(sp_scores)   # 0.0 when skip_finbert
        tone_lm              = float(np.mean(lm_scores)) if lm_scores else 0.0
        upvote_weighted_tone = (sum(upvote_sp) / sum(upvote_weights)
                                if upvote_weights else 0.0)
        sentiment_vol        = float(np.std(sp_scores)) if len(sp_scores) > 1 else 0.0

        # ── Relative share of attention ─────────────────────────────────────────
        rel_share = n_posts / total_posts_day if total_posts_day > 0 else 0.0

        # ── Breadth ─────────────────────────────────────────────────────────────
        unique_authors = len({p["author"] for p in posts})
        breadth = unique_authors / (n_posts + 1e-9)

        # ── Author concentration ──────────────────────────────────────────────
        herfindahl = _herfindahl(posts)

        # ── Agreement ──────────────────────────────────────────────────────────
        H        = _entropy(probs_list)
        H_max    = math.log(3)   # maximum entropy for 3 classes
        agreement = 1.0 - (H / H_max) if H_max > 0 else 0.0

        # ── Duplication rate ──────────────────────────────────────────────────
        dup_count = sum(1 for p in posts if p.get("is_duplicate", False))
        dup_rate  = dup_count / n_posts if n_posts > 0 else 0.0

        # ── Burstiness ────────────────────────────────────────────────────────
        burstiness = _burstiness(posts)

        # ── Credibility ───────────────────────────────────────────────────────
        credibility = max(0.0, min(1.0,
            1.0 - alpha * dup_rate - beta * herfindahl - gamma * burstiness
        ))

        # ── Top 3 posts by upvote score (for detail page) ─────────────────────
        # Only include posts tagged to ≤5 tickers: these are genuinely discussing
        # this specific company. Posts tagged to hundreds of tickers are broad
        # market commentary and tell the user nothing about this stock.
        specific_posts = [p for p in posts if p.get("n_companies", 999) <= 5]
        top3 = sorted(specific_posts, key=lambda p: p.get("score", 0), reverse=True)[:3]
        top_posts_data = [
            {
                "title":       p.get("title", "")[:200],
                "text":        p.get("normalised_text", "")[:280],
                "author":      p.get("author", "unknown"),
                "score":       int(p.get("score", 0)),
                "subreddit":   p.get("subreddit", ""),
                "created_utc": float(p.get("created_utc", 0)),
            }
            for p in top3
        ]

        rows.append({
            "ticker":              ticker,
            "date":                date_str,
            "n_posts":             n_posts,
            "tone_finbert":        tone_finbert,
            "upvote_weighted_tone": upvote_weighted_tone,
            "sentiment_vol":       sentiment_vol,
            "tone_lm":             tone_lm,
            "rel_share":           rel_share,
            "breadth":             breadth,
            "herfindahl":          herfindahl,
            "agreement":           agreement,
            "dup_rate":            dup_rate,
            "burstiness":          burstiness,
            "credibility":         credibility,
            "top_posts":           json.dumps(top_posts_data),
        })

    df = pd.DataFrame(rows)

    # ── Attention shock ─────────────────────────────────────────────────────
    if history_df is not None and not history_df.empty:
        full_vol = (
            pd.concat([history_df[["ticker", "date", "n_posts"]], df[["ticker", "date", "n_posts"]]])
            .sort_values(["ticker", "date"])
            .drop_duplicates(subset=["ticker", "date"], keep="last")
        )
        shocks = {}
        for ticker, grp in full_vol.groupby("ticker"):
            grp = grp.set_index("date")["n_posts"]
            shock_series = compute_attention_shock(grp, window)
            if date_str in shock_series.index:
                val = shock_series[date_str]
                shocks[ticker] = float(val.iloc[-1] if isinstance(val, pd.Series) else val)
            else:
                shocks[ticker] = 0.0
        df["attn_shock"] = df["ticker"].map(shocks).fillna(0.0)
    else:
        df["attn_shock"] = 0.0   # No history available; defaults to zero

    # ── Tone delta: change vs prior day ──────────────────────────────────────
    if history_df is not None and not history_df.empty and "tone_finbert" in history_df.columns:
        prev_tone = (
            history_df.sort_values("date")
            .groupby("ticker")["tone_finbert"]
            .last()
        )
        df["tone_delta"] = df["tone_finbert"] - df["ticker"].map(prev_tone).fillna(0.0)
    else:
        df["tone_delta"] = 0.0

    return df.reset_index(drop=True)


# ── Save / load ───────────────────────────────────────────────────────────────

def save_features(df: pd.DataFrame, date_str: str):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DATA_DIR, f"{date_str}.csv")
    df.to_csv(path, index=False)
    print(f"[features] Saved {len(df)} ticker-day rows → {path}")
    return path


def load_all_features() -> pd.DataFrame:
    """Loads and concatenates all processed feature CSVs into one DataFrame."""
    frames = []
    if not os.path.exists(PROCESSED_DATA_DIR):
        return pd.DataFrame()
    for fname in sorted(os.listdir(PROCESSED_DATA_DIR)):
        if fname.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join(PROCESSED_DATA_DIR, fname)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json as _json
    from text_utils import load_ticker_universe, process_raw_posts

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    raw_path = os.path.join("data/raw", f"{args.date}.json")
    with open(raw_path) as f:
        raw_posts = _json.load(f)

    universe = load_ticker_universe()
    processed = process_raw_posts(raw_posts, universe)

    history = load_all_features()
    df = compute_features(processed, args.date, history_df=history)
    save_features(df, args.date)
    print(df)
