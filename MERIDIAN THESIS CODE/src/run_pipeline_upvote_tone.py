# run_pipeline_upvote_tone.py
# Runs the full pipeline using upvote-weighted tone aggregation for tone_finbert.
# Uses the CSV data source to cover all trading days.
# Saves processed CSVs to data/processed_upvote_tone/

import sys
import os
sys.path.insert(0, "src")

import json
import pandas as pd
from text_utils import load_ticker_universe, process_raw_posts
from features_upvote_weighted import compute_features_upvote_tone
from csv_loader import load_and_index, posts_for_date, _load_company_map
from config import (
    ATTN_WINDOW, PARAMS_PATH,
    TRAIN_START, EVAL_END,
)

OUTPUT_DIR = "data/processed_upvote_tone"


def run_pipeline():
    print("[pipeline_upvote_tone] Starting upvote-weighted tone pipeline")
    print("[pipeline_upvote_tone] Loading CSV dataset...")

    df, day_index, trading_days, post_company_counts = load_and_index(
        start_str=TRAIN_START,
        end_str=EVAL_END,
    )
    company_map = _load_company_map()
    universe = load_ticker_universe()

    window = ATTN_WINDOW
    try:
        with open(PARAMS_PATH) as f:
            params = json.load(f)
            window = params.get("window", ATTN_WINDOW)
    except FileNotFoundError:
        pass

    days_with_data = [d for d in trading_days if d in day_index]
    print(f"[pipeline_upvote_tone] {len(days_with_data)} trading days with data")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history_df = pd.DataFrame()
    total_rows = 0
    processed_days = 0

    for i, date_str in enumerate(days_with_data):
        raw_posts = posts_for_date(date_str, df, day_index, company_map, post_company_counts)
        if not raw_posts:
            continue

        processed = process_raw_posts(raw_posts, universe)
        features = compute_features_upvote_tone(
            processed, date_str, history_df=history_df, window=window
        )

        if features.empty:
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{date_str}.csv")
        features.to_csv(out_path, index=False)
        total_rows += len(features)
        processed_days += 1

        history_df = pd.concat([history_df, features], ignore_index=True)

        if processed_days % 50 == 0:
            print(f"  [{processed_days}/{len(days_with_data)}] {date_str}: {len(features)} tickers")

    print(f"\n[pipeline_upvote_tone] Done! {total_rows} total ticker-day rows across {processed_days} days")
    print(f"[pipeline_upvote_tone] Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_pipeline()
