# run_pipeline_stocktwits.py
# Full pipeline using StockTwits RoBERTa for sentiment scoring.

import sys
import os
sys.path.insert(0, "src")

import json
import math
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime, timedelta

from text_utils import load_ticker_universe, process_raw_posts
from features import (
    _trimmed_mean, _herfindahl, _entropy, _burstiness,
    compute_attention_shock, score_post_lm,
    PROCESSED_DATA_DIR
)
from sentiment_stocktwits import score_post_stocktwits
from csv_loader import load_and_index, posts_for_date, _load_company_map
from config import (
    TRIM_RATE, BURST_WINDOW_MINUTES,
    CRED_ALPHA, CRED_BETA, CRED_GAMMA,
    ATTN_WINDOW, RAW_DATA_DIR, PARAMS_PATH,
    TRAIN_START, TRAIN_END, EVAL_START, EVAL_END,
)

OUTPUT_DIR = "data/processed_stocktwits"


def compute_features_stocktwits(processed_posts, date_str, history_df=None,
                                window=ATTN_WINDOW,
                                alpha=CRED_ALPHA, beta=CRED_BETA, gamma=CRED_GAMMA):
    """
    Identical to features.compute_features() but uses the StockTwits RoBERTa
    via sentiment_stocktwits.score_post_stocktwits() instead of the base FinBERT.
    """
    ticker_posts = defaultdict(list)
    for post in processed_posts:
        for ticker in post["tickers"]:
            ticker_posts[ticker].append(post)

    if not ticker_posts:
        return pd.DataFrame()

    total_posts_day = sum(len(v) for v in ticker_posts.values())
    rows = []

    for ticker, posts in ticker_posts.items():
        n_posts = len(posts)
        eng_posts = [p for p in posts if p.get("is_english", True)
                     and not p.get("is_duplicate", False)]

        sp_scores = []
        probs_list = []
        lm_scores = []
        upvote_weights = []
        upvote_sp = []

        for post in eng_posts:
            try:
                result = score_post_stocktwits(post["normalised_text"])
                sp = result["sp"]
                sp_scores.append(sp)
                probs_list.append({
                    "positive": result.get("positive", 0),
                    "negative": result.get("negative", 0),
                    "neutral": result.get("neutral", 0),
                })
                lm_scores.append(score_post_lm(post["normalised_text"]))
                w = math.sqrt(max(0, post.get("score", 0)) + 1)
                upvote_weights.append(w)
                upvote_sp.append(sp * w)
            except Exception:
                continue

        tone_finbert = _trimmed_mean(sp_scores)
        tone_lm = float(np.mean(lm_scores)) if lm_scores else 0.0
        upvote_weighted_tone = (sum(upvote_sp) / sum(upvote_weights)
                                if upvote_weights else 0.0)
        sentiment_vol = float(np.std(sp_scores)) if len(sp_scores) > 1 else 0.0

        rel_share = n_posts / total_posts_day if total_posts_day > 0 else 0.0
        unique_authors = len({p["author"] for p in posts})
        breadth = unique_authors / (n_posts + 1e-9)
        herfindahl = _herfindahl(posts)

        H = _entropy(probs_list)
        H_max = math.log(3)
        agreement = 1.0 - (H / H_max) if H_max > 0 else 0.0

        dup_count = sum(1 for p in posts if p.get("is_duplicate", False))
        dup_rate = dup_count / n_posts if n_posts > 0 else 0.0
        burstiness = _burstiness(posts)
        credibility = max(0.0, min(1.0,
            1.0 - alpha * dup_rate - beta * herfindahl - gamma * burstiness
        ))

        specific_posts = [p for p in posts if p.get("n_companies", 999) <= 5]
        top3 = sorted(specific_posts, key=lambda p: p.get("score", 0), reverse=True)[:3]
        top_posts_data = [
            {
                "title": p.get("title", "")[:200],
                "text": p.get("normalised_text", "")[:280],
                "author": p.get("author", "unknown"),
                "score": int(p.get("score", 0)),
                "subreddit": p.get("subreddit", ""),
                "created_utc": float(p.get("created_utc", 0)),
            }
            for p in top3
        ]

        rows.append({
            "ticker": ticker,
            "date": date_str,
            "n_posts": n_posts,
            "tone_finbert": tone_finbert,
            "upvote_weighted_tone": upvote_weighted_tone,
            "sentiment_vol": sentiment_vol,
            "tone_lm": tone_lm,
            "rel_share": rel_share,
            "breadth": breadth,
            "herfindahl": herfindahl,
            "agreement": agreement,
            "dup_rate": dup_rate,
            "burstiness": burstiness,
            "credibility": credibility,
            "top_posts": json.dumps(top_posts_data),
        })

    df = pd.DataFrame(rows)

    # Attention shock
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
        df["attn_shock"] = 0.0

    # Tone delta
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


def run_pipeline():
    print("[pipeline_stocktwits] Starting StockTwits RoBERTa pipeline")
    print("[pipeline_stocktwits] Loading CSV dataset...")

    # Load and index the CSV dataset (same as original pipeline)
    df, day_index, trading_days, post_company_counts = load_and_index(
        start_str=TRAIN_START,
        end_str=EVAL_END,
    )
    company_map = _load_company_map()
    universe = load_ticker_universe()

    # Get calibrated window
    window = ATTN_WINDOW
    try:
        with open(PARAMS_PATH) as f:
            params = json.load(f)
            window = params.get("window", ATTN_WINDOW)
    except FileNotFoundError:
        pass

    # Process all trading days that have data
    days_with_data = [d for d in trading_days if d in day_index]
    print(f"[pipeline_stocktwits] {len(days_with_data)} trading days with data")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history_df = pd.DataFrame()
    total_rows = 0
    processed_days = 0

    for i, date_str in enumerate(days_with_data):
        raw_posts = posts_for_date(date_str, df, day_index, company_map, post_company_counts)
        if not raw_posts:
            continue

        processed = process_raw_posts(raw_posts, universe)
        features = compute_features_stocktwits(
            processed, date_str, history_df=history_df, window=window
        )

        if features.empty:
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{date_str}.csv")
        features.to_csv(out_path, index=False)
        total_rows += len(features)
        processed_days += 1

        # Add to history for subsequent days
        history_df = pd.concat([history_df, features], ignore_index=True)

        if processed_days % 50 == 0:
            print(f"  [{processed_days}/{len(days_with_data)}] {date_str}: {len(features)} tickers")

    print(f"\n[pipeline_stocktwits] Done! {total_rows} total ticker-day rows across {processed_days} days")
    print(f"[pipeline_stocktwits] Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_pipeline()
