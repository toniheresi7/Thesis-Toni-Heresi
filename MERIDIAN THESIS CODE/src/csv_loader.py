# csv_loader.py
# Reads data/reddit_sp500.csv and returns per-trading-day post lists
# in the same format as collector.py output.

import os
import json
import pandas as pd
import pytz
from datetime import datetime, timedelta

from config import (
    CSV_PATH, MARKET_CLOSE_HOUR_ET, MAX_POSTS_PER_DAY,
    COMPANY_TICKER_MAP_PATH,
)

EASTERN = pytz.timezone("America/New_York")


# ── Load company→ticker map ───────────────────────────────────────────────────

def _load_company_map(path: str = COMPANY_TICKER_MAP_PATH) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ── NYSE trading calendar ─────────────────────────────────────────────────────

def get_trading_days(start_str: str, end_str: str) -> list:
    """Returns sorted list of NYSE trading day strings 'YYYY-MM-DD' in [start, end]."""
    try:
        import pandas_market_calendars as mcal
        nyse     = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=start_str, end_date=end_str)
        return [str(d.date()) for d in schedule.index]
    except ImportError:
        # Fallback: all weekdays (approximate — no holiday removal)
        days, d = [], datetime.strptime(start_str, "%Y-%m-%d").date()
        end = datetime.strptime(end_str, "%Y-%m-%d").date()
        while d <= end:
            if d.weekday() < 5:
                days.append(d.isoformat())
            d += timedelta(days=1)
        return days


# ── Load and index the CSV ────────────────────────────────────────────────────

def load_and_index(csv_path: str = None,
                   start_str: str = None,
                   end_str: str   = None) -> tuple:
    """Loads CSV, assigns trading days, returns (df, day_index, trading_days, post_company_counts)."""
    path = csv_path or CSV_PATH
    print(f"[csv_loader] Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)

    # ── Parse timestamps ──────────────────────────────────────────────────────
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df = df.dropna(subset=["created_utc", "company"]).copy()
    df["company"] = df["company"].astype(str).str.strip()

    # Compute n_companies per post (before date filter)
    post_company_counts = (
        df.groupby("id")["company"].nunique().to_dict()
    )

    # ── Filter to date range (±1 day buffer for timezone shift) ──────────────
    if start_str and end_str:
        start_et = EASTERN.localize(datetime.strptime(start_str, "%Y-%m-%d").replace(hour=0))
        end_et   = EASTERN.localize(datetime.strptime(end_str,   "%Y-%m-%d").replace(hour=23, minute=59))
        buf      = timedelta(days=1)
        ts_start = (start_et - buf).timestamp()
        ts_end   = (end_et   + buf).timestamp()
        df = df[(df["created_utc"] >= ts_start) & (df["created_utc"] <= ts_end)].copy()

    print(f"[csv_loader] {len(df):,} rows after date filter.")

    # ── Combine title + text ──────────────────────────────────────────────────
    title = df["title"].fillna("").astype(str)
    body  = df["text"].fillna("").astype(str)
    df["full_text"] = (title + " " + body).str.strip()

    # ── Assign each post to a trading day (vectorised) ────────────────────────
    # Convert UTC timestamp → Eastern datetime
    dt_et    = pd.to_datetime(df["created_utc"], unit="s", utc=True).dt.tz_convert(EASTERN)
    et_date  = dt_et.dt.date.astype(str)
    et_hour  = dt_et.dt.hour

    # Posts at/after 16:00 ET belong to the NEXT trading day
    next_day = (pd.to_datetime(et_date) + pd.Timedelta(days=1)).dt.date.astype(str)
    df["signal_date_raw"] = et_date.where(et_hour < MARKET_CLOSE_HOUR_ET, next_day)

    # Build calendar set for O(1) lookup
    trading_days    = get_trading_days(start_str or "2020-01-01", end_str or "2025-12-31")
    trading_day_set = set(trading_days)

    # Forward-fill non-trading days to the next trading day
    cal_sorted = sorted(trading_day_set)
    # Build map: every calendar date → nearest next trading day
    all_cal = pd.date_range(
        start=(pd.Timestamp(cal_sorted[0])  - pd.Timedelta(days=5)).date(),
        end  =(pd.Timestamp(cal_sorted[-1]) + pd.Timedelta(days=5)).date(),
        freq="D"
    )
    fwd_map = {}
    for d in all_cal:
        ds = str(d.date())
        for offset in range(8):
            cand = str((d + pd.Timedelta(days=offset)).date())
            if cand in trading_day_set:
                fwd_map[ds] = cand
                break

    df["trading_day"] = df["signal_date_raw"].map(fwd_map)
    df = df.dropna(subset=["trading_day"])

    # ── Build day index {date_str: [row_indices]} ─────────────────────────────
    day_index = df.groupby("trading_day").apply(lambda g: g.index.tolist()).to_dict()
    covered   = [d for d in trading_days if d in day_index]
    print(f"[csv_loader] Indexed {len(df):,} posts → {len(covered)} trading days covered.")

    return df, day_index, trading_days, post_company_counts


# ── Per-day post retrieval ────────────────────────────────────────────────────

def posts_for_date(date_str: str,
                   df:        pd.DataFrame,
                   day_index: dict,
                   company_map: dict = None,
                   post_company_counts: dict = None) -> list:
    """Returns deduplicated post dicts for a single trading day."""
    if company_map is None:
        company_map = _load_company_map()

    row_indices = day_index.get(date_str, [])
    if not row_indices:
        return []

    day_df = df.loc[row_indices]

    # Deduplicate by post id: keep the highest-scoring row
    day_df = (
        day_df.sort_values("score", ascending=False)
              .drop_duplicates(subset="id")
    )

    posts = []
    for _, row in day_df.iterrows():
        post_id = str(row.get("id", f"csv_{row.name}"))
        company = str(row.get("company", "")).strip()
        ticker  = company_map.get(company, "").strip().upper()
        text    = str(row.get("full_text", "")).strip()

        # Prepend authoritative ticker so text_utils never misses it
        if ticker and f"${ticker}" not in text.upper():
            text = f"${ticker} {text}"

        n_companies = 1
        if post_company_counts is not None:
            n_companies = post_company_counts.get(post_id, 1)

        posts.append({
            "post_id":      post_id,
            "subreddit":    str(row.get("subreddit", "reddit")).lower(),
            "author":       str(row.get("author", "unknown")),
            "title":        str(row.get("title", "")).strip(),
            "text":         text,
            "created_utc":  float(row.get("created_utc", 0)),
            "score":        int(row.get("score", 0) or 0),
            "n_companies":  n_companies,
        })

    # Even-time sampling if over daily cap (mirrors collector.py)
    if len(posts) > MAX_POSTS_PER_DAY:
        posts_sorted = sorted(posts, key=lambda p: p["created_utc"])
        step  = len(posts_sorted) / MAX_POSTS_PER_DAY
        posts = [posts_sorted[int(i * step)] for i in range(MAX_POSTS_PER_DAY)]

    return posts


# ── Coverage summary (used by run_batch.py) ───────────────────────────────────

def coverage_summary(day_index: dict, trading_days: list) -> dict:
    """Returns per-trading-day post counts for the indexed period."""
    return {d: len(day_index.get(d, [])) for d in trading_days}
