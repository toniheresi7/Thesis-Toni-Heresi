# run_pipeline.py
# Master script: runs one complete daily pipeline cycle.
#
# Usage:
#   python run_pipeline.py --date 2025-01-15 --mock         (mock Reddit data)
#   python run_pipeline.py --date 2025-01-15                (live Reddit API)
#
# What it does:
#   1. Collect Reddit posts (or generate mock data)
#   2. Process text: normalise, map tickers, detect duplicates
#   3. Compute ticker-day features
#   4. Load frozen model and rank tickers
#   5. Print the Top-10 shortlist to console

import sys
import os
sys.path.insert(0, "src")

import json
import argparse
from datetime import datetime, date

from collector   import collect, generate_mock_data, save_raw_posts
from text_utils  import load_ticker_universe, process_raw_posts
from features    import compute_features, save_features, load_all_features
from model       import load_model, predict_and_rank, build_shortlist
import config as _cfg


def run(trading_date: date, mock: bool = False):
    print(f"\n{'='*60}")
    print(f"  Social Signal Pipeline · {trading_date}")
    print(f"{'='*60}\n")

    # ── Step 1a: CSV / Reddit / mock collection ────────────────────────────────
    print("[1/4] Collecting posts...")
    if _cfg.DATA_SOURCE == "csv":
        from csv_loader import load_and_index, posts_for_date, _load_company_map
        from datetime import timedelta
        print(f"    Data source: CSV ({_cfg.CSV_PATH})")
        df, day_index, _, post_company_counts = load_and_index(
            start_str=(trading_date - timedelta(days=2)).isoformat(),
            end_str=trading_date.isoformat(),
        )
        cmap      = _load_company_map()
        all_posts = posts_for_date(trading_date.isoformat(), df, day_index, cmap, post_company_counts)
        save_raw_posts(all_posts, trading_date)
    elif mock:
        raw_path  = generate_mock_data(trading_date)
        print(f"    Mock Reddit data generated.")
        with open(raw_path) as f:
            all_posts = json.load(f)
    else:
        raw_path = collect(
            trading_date,
            client_id=os.environ["REDDIT_CLIENT_ID"],
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=os.environ["REDDIT_USER_AGENT"],
        )
        with open(raw_path) as f:
            all_posts = json.load(f)

    print(f"    Posts loaded: {len(all_posts)}.")
    print()

    # ── Step 2: Text processing ───────────────────────────────────────────────
    print("[2/4] Processing text (normalise, map tickers, detect duplicates)...")
    universe  = load_ticker_universe()
    processed = process_raw_posts(all_posts, universe)

    matched   = sum(1 for p in processed if p["tickers"])
    dup_count = sum(1 for p in processed if p.get("is_duplicate"))
    non_eng   = sum(1 for p in processed if not p.get("is_english", True))
    print(f"    Posts with matched tickers: {matched}")
    print(f"    Near-duplicates flagged:    {dup_count}")
    print(f"    Non-English posts excluded: {non_eng}\n")

    # ── Step 3: Feature computation ───────────────────────────────────────────
    print("[3/4] Computing ticker-day features...")
    history_df = load_all_features()

    # Load calibrated parameters if available
    window = 10   # default
    try:
        from model import load_model as _lm
        _, _, params = _lm()
        window = params.get("window", 10)
    except FileNotFoundError:
        pass

    date_str   = trading_date.isoformat()
    features   = compute_features(processed, date_str, history_df=history_df, window=window)

    if features.empty:
        print("    No tickers with sufficient data today. Pipeline complete.\n")
        return

    save_features(features, date_str)
    print(f"    {len(features)} ticker-day rows computed.\n")

    # ── Step 4: Rank and display ──────────────────────────────────────────────
    print("[4/4] Ranking tickers and building shortlist...")
    try:
        model, scaler, _ = load_model()
    except FileNotFoundError:
        print("    Model not found. Train it first: python src/model.py --train")
        print("    (Skipping ranking step.)\n")
        return

    top_k     = predict_and_rank(features, model, scaler)

    if top_k.empty:
        print("    No tickers met the minimum evidence rule today.")
        print("    Credibility of today's signal is structurally low.\n")
        return

    shortlist = build_shortlist(top_k, model, scaler)

    print(f"\n{'─'*60}")
    print(f"  TOP {len(shortlist)} STOCKS TO MONITOR · {trading_date}")
    print(f"{'─'*60}")
    for item in shortlist:
        cred = item['credibility_label'].upper()
        print(f"  {item['rank']:2}. {item['ticker']:6}  [{cred:6}]  {item['explanation']}")
    print(f"{'─'*60}")
    print("  ⚠️  For monitoring purposes only. Not financial advice.")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the daily signal pipeline.")
    parser.add_argument("--date", required=True, help="Trading day YYYY-MM-DD")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock Reddit data (when API not yet available)")
    args = parser.parse_args()

    trading_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    run(trading_date, mock=args.mock)
