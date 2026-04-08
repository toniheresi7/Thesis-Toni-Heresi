# run_batch.py
# Processes the full CSV dataset for TRAIN_START→EVAL_END, trains the model,
# and writes data/model_report.json.
#
# Usage:
#   python run_batch.py                        # uses config.py date ranges
#   python run_batch.py --start 2022-01-01 --end 2023-12-31
#   python run_batch.py --skip-existing        # skip days already in data/processed/
#   python run_batch.py --stats-only           # print coverage stats, no processing

import sys
import os
sys.path.insert(0, "src")

import argparse
from datetime import date as date_type
import pandas as pd

from config import TRAIN_START, TRAIN_END, EVAL_START, EVAL_END, PROCESSED_DATA_DIR
from csv_loader import load_and_index, posts_for_date, get_trading_days, _load_company_map
from text_utils import load_ticker_universe, process_raw_posts
from features   import compute_features, save_features, load_all_features


# ── Batched FinBERT setup (MPS on Apple Silicon, CPU fallback) ────────────────
# We monkey-patch features.score_post_finbert to use a per-day cache that is
# pre-filled using batch inference. This reduces FinBERT time by ~60x vs
# one-at-a-time CPU (4 ms/post on MPS batch-64 vs 260 ms/post on CPU).

import torch
import features as _feat
from transformers import pipeline as _hf_pipeline

_DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
_BATCH_SIZE = 64 if _DEVICE == "mps" else 8
print(f"[batch] FinBERT device: {_DEVICE}  batch_size: {_BATCH_SIZE}")

_PIPE = _hf_pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    return_all_scores=True,
    truncation=True,
    max_length=512,
    device=_DEVICE,
    batch_size=_BATCH_SIZE,
)

_FINBERT_CACHE: dict = {}   # text[:512] → {positive, negative, neutral, sp}


def _score_post_finbert_cached(text: str) -> dict:
    """Drop-in replacement for features.score_post_finbert — hits the day cache."""
    key = text[:512]
    if key in _FINBERT_CACHE:
        return _FINBERT_CACHE[key]
    # Fallback for any cache miss (shouldn't happen in batch mode)
    result = _PIPE([key])[0]
    probs  = {r["label"].lower(): r["score"] for r in result}
    entry  = {**probs, "sp": probs.get("positive", 0.0) - probs.get("negative", 0.0)}
    _FINBERT_CACHE[key] = entry
    return entry


# Replace the per-post function with our cached version
_feat.score_post_finbert = _score_post_finbert_cached


def _prefill_cache(processed_posts: list):
    """
    Runs batch FinBERT on all unique English non-duplicate texts in the
    day's post list and fills _FINBERT_CACHE.
    """
    _FINBERT_CACHE.clear()

    texts = list({
        post.get("normalised_text", post.get("text", ""))[:512]
        for post in processed_posts
        if post.get("is_english", True) and not post.get("is_duplicate", False)
        and post.get("normalised_text", post.get("text", "")).strip()
    })

    if not texts:
        return

    for i in range(0, len(texts), _BATCH_SIZE):
        batch   = texts[i : i + _BATCH_SIZE]
        results = _PIPE(batch)
        for text, result in zip(batch, results):
            probs = {r["label"].lower(): r["score"] for r in result}
            _FINBERT_CACHE[text] = {
                **probs,
                "sp": probs.get("positive", 0.0) - probs.get("negative", 0.0),
            }


# ── Step 1: Process all trading days ─────────────────────────────────────────

def process_range(start_str: str, end_str: str,
                  df, day_index, trading_days,
                  company_map: dict,
                  skip_existing: bool = False,
                  post_company_counts: dict = None):

    universe   = load_ticker_universe()
    history_df = load_all_features()

    days_in_range = [d for d in trading_days if start_str <= d <= end_str]
    print(f"\n[batch] Processing {len(days_in_range)} trading days "
          f"({start_str} → {end_str})\n")

    processed_count = skipped_count = empty_count = 0

    for i, date_str in enumerate(days_in_range):
        out_path = os.path.join(PROCESSED_DATA_DIR, f"{date_str}.csv")
        if skip_existing and os.path.exists(out_path):
            skipped_count += 1
            continue

        posts = posts_for_date(date_str, df, day_index, company_map, post_company_counts)
        if not posts:
            empty_count += 1
            continue

        # 1. text_utils: normalise, match tickers, detect duplicates
        processed = process_raw_posts(posts, universe)

        # 2. pre-fill FinBERT cache with batch inference
        _prefill_cache(processed)

        # 3. compute features (score_post_finbert hits the cache, near-zero cost)
        features = compute_features(processed, date_str, history_df=history_df)

        if not features.empty:
            save_features(features, date_str)
            history_df = pd.concat([history_df, features], ignore_index=True) \
                         if not history_df.empty else features
            processed_count += 1
        else:
            empty_count += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(days_in_range):
            print(f"  [{i+1:4d}/{len(days_in_range)}] {date_str}  "
                  f"posts={len(posts):3d}  "
                  f"done={processed_count}  skip={skipped_count}  empty={empty_count}")

    print(f"\n[batch] Done: {processed_count} days saved, "
          f"{skipped_count} skipped, {empty_count} empty.\n")
    return processed_count


# ── Step 2: Train model ───────────────────────────────────────────────────────

def train_model():
    from model import train, save_model, FEATURE_COLS
    from market_data import download_prices, compute_returns, compute_momentum_features

    print("[batch] Loading all features for training...")
    features_df = load_all_features()
    if features_df.empty:
        print("[batch] No features found.")
        return False

    tickers = features_df["ticker"].unique().tolist()
    dates   = sorted(features_df["date"].unique())
    start   = date_type.fromisoformat(dates[0])
    end     = date_type.fromisoformat(dates[-1])

    print(f"[batch] {len(features_df):,} rows, {len(tickers)} tickers, {len(dates)} days.")
    print(f"[batch] Downloading prices {start} → {end} ...")
    prices_df = download_prices(tickers, start, end)

    print("[batch] Computing price momentum features ...")
    mom_df = compute_momentum_features(prices_df, dates)
    features_df = features_df.merge(mom_df, on=["date", "ticker"], how="left")
    for col in ["mom_1d", "mom_5d", "mom_20d"]:
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna(0.0)

    returns_df = compute_returns(prices_df, dates)

    print("[batch] Training LightGBM model ...")
    model, scaler, params, train_dates, eval_dates = train(features_df, returns_df)

    save_model(model, scaler, {
        "model_type":   "lightgbm",
        "params":       params,
        "train_dates":  train_dates,
        "eval_dates":   eval_dates,
        "feature_cols": FEATURE_COLS,
        "window":       5,
    })
    print(f"[batch] Model saved. params={params}, "
          f"train={len(train_dates)}d, eval={len(eval_dates)}d.")
    return True


# ── Step 3: Coverage stats ────────────────────────────────────────────────────

def print_stats(day_index: dict, trading_days: list):
    total_posts  = sum(len(v) for v in day_index.values())
    covered_days = sum(1 for d in trading_days if day_index.get(d))
    qualifying   = sum(1 for d in trading_days if len(day_index.get(d, [])) >= 3)

    print(f"\n{'='*60}")
    print(f"  DATASET COVERAGE SUMMARY")
    print(f"{'='*60}")
    print(f"  Trading days in range : {len(trading_days)}")
    print(f"  Days with posts       : {covered_days}  "
          f"({covered_days/max(len(trading_days),1)*100:.1f}%)")
    print(f"  Total posts indexed   : {total_posts:,}")
    if covered_days:
        print(f"  Avg posts / day       : {total_posts/covered_days:.0f}")
    print(f"  Days with ≥3 posts    : {qualifying}")
    print(f"{'='*60}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",         default=TRAIN_START)
    parser.add_argument("--end",           default=EVAL_END)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stats-only",    action="store_true")
    parser.add_argument("--no-train",      action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Social Signal Batch Pipeline")
    print(f"  Range : {args.start} → {args.end}")
    print(f"{'='*60}\n")

    df, day_index, trading_days, post_company_counts = load_and_index(
        start_str=args.start, end_str=args.end,
    )
    trading_days = [d for d in trading_days if args.start <= d <= args.end]

    print_stats(day_index, trading_days)

    if args.stats_only:
        return

    company_map = _load_company_map()
    n = process_range(
        args.start, args.end, df, day_index, trading_days,
        company_map, skip_existing=args.skip_existing,
        post_company_counts=post_company_counts,
    )

    if n == 0 and not args.skip_existing:
        print("[batch] Nothing processed. Exiting.")
        return

    if not args.no_train:
        ok = train_model()
        if not ok:
            return

    print("\n[batch] All steps complete.")


if __name__ == "__main__":
    main()
