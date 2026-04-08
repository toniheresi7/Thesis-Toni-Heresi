# market_data.py
# Price data via yfinance, return computation, and momentum features.

import os
import json
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from datetime import date, timedelta

from config import HORIZONS, TICKER_UNIVERSE_FILE

# ── NYSE trading calendar ─────────────────────────────────────────────────────
NYSE = mcal.get_calendar("NYSE")


def get_trading_day(reference_date: date, offset: int) -> date:
    """Returns the trading day `offset` trading days after reference_date."""
    start = reference_date + timedelta(days=1)
    end   = reference_date + timedelta(days=offset * 3)   # generous buffer
    schedule = NYSE.schedule(
        start_date=start.isoformat(),
        end_date=end.isoformat()
    )
    trading_days = mcal.date_range(schedule, frequency="1D")
    trading_days = [d.date() for d in trading_days]
    if len(trading_days) < offset:
        raise ValueError(f"Not enough trading days found for offset={offset}")
    return trading_days[offset - 1]


# ── Price download ────────────────────────────────────────────────────────────

def download_prices(tickers: list,
                    start_date: date,
                    end_date: date) -> pd.DataFrame:
    """Downloads daily adjusted close prices via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    if not tickers:
        return pd.DataFrame(columns=["date", "ticker", "adj_close"])

    raw = yf.download(
        tickers,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),   # yfinance end is exclusive
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "adj_close"])

    # Handle single vs multiple tickers (yfinance returns different structures)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    close.index = pd.to_datetime(close.index).date
    close.index.name = "date"

    df = close.reset_index().melt(id_vars="date", var_name="ticker", value_name="adj_close")
    df = df.dropna(subset=["adj_close"])
    return df


# ── Return computation ────────────────────────────────────────────────────────

def compute_returns(prices_df: pd.DataFrame,
                    signal_dates: list) -> pd.DataFrame:
    """Computes T+1 and T+5 returns for each (ticker, signal_date)."""
    # Use string dates throughout to avoid date/Timestamp hash mismatch
    price_pivot = prices_df.pivot(index="date", columns="ticker", values="adj_close")
    price_pivot.index = price_pivot.index.astype(str)

    sig_strs = [d if isinstance(d, str) else d.isoformat() for d in signal_dates]
    if not sig_strs:
        return pd.DataFrame(columns=["date", "ticker", "ret_t1", "ret_t5"])

    min_d = min(sig_strs)
    max_d = max(sig_strs)

    # Build NYSE calendar once for the full range
    schedule = NYSE.schedule(
        start_date=min_d,
        end_date=(date.fromisoformat(max_d) + timedelta(days=30)).isoformat(),
    )
    cal_strs = [str(d.date()) for d in mcal.date_range(schedule, frequency="1D")]
    cal_idx  = {d: i for i, d in enumerate(cal_strs)}   # str → int index

    # Map each signal date → T+1 and T+5 string dates
    date_map = {}
    for ds in sig_strs:
        i = cal_idx.get(ds)
        if i is None:
            continue
        t1 = cal_strs[i + 1] if i + 1 < len(cal_strs) else None
        t5 = cal_strs[i + 5] if i + 5 < len(cal_strs) else None
        date_map[ds] = (t1, t5)

    # Bulk return computation
    rows = []
    price_idx = set(price_pivot.index)
    nan_row   = pd.Series(np.nan, index=price_pivot.columns)

    for ds, (t1, t5) in date_map.items():
        if ds not in price_idx:
            continue
        p_t  = price_pivot.loc[ds]
        p_t1 = price_pivot.loc[t1] if t1 in price_idx else nan_row
        p_t5 = price_pivot.loc[t5] if t5 in price_idx else nan_row

        day_df = pd.DataFrame({
            "ret_t1": (p_t1 - p_t) / p_t.replace(0, np.nan),
            "ret_t5": (p_t5 - p_t) / p_t.replace(0, np.nan),
        })
        day_df.index.name = "ticker"
        day_df = day_df.reset_index()
        day_df["date"] = ds
        rows.append(day_df)

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "ret_t1", "ret_t5"])

    return pd.concat(rows, ignore_index=True)


# ── Price momentum features for model training ───────────────────────────────

def compute_momentum_features(prices_df: pd.DataFrame,
                               signal_dates: list) -> pd.DataFrame:
    """Computes backward-looking momentum (1d, 5d, 20d) for each (signal_date, ticker)."""
    price_pivot = (
        prices_df.pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
    )

    mom_1d  = price_pivot.pct_change(1)
    mom_5d  = price_pivot.pct_change(5)
    mom_20d = price_pivot.pct_change(20)

    sig_date_set = set()
    for d in signal_dates:
        sig_date_set.add(date.fromisoformat(d) if isinstance(d, str) else d)

    valid_dates = [d for d in price_pivot.index if d in sig_date_set]

    frames = []
    for d in valid_dates:
        sub = pd.DataFrame({
            "date":   d.isoformat() if hasattr(d, "isoformat") else str(d),
            "ticker": price_pivot.columns,
            "mom_1d":  mom_1d.loc[d].values  if d in mom_1d.index  else np.nan,
            "mom_5d":  mom_5d.loc[d].values  if d in mom_5d.index  else np.nan,
            "mom_20d": mom_20d.loc[d].values if d in mom_20d.index else np.nan,
        })
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "mom_1d", "mom_5d", "mom_20d"])

    result = pd.concat(frames, ignore_index=True)
    for col in ["mom_1d", "mom_5d", "mom_20d"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0).clip(-0.5, 0.5)
    return result


# ── Momentum baseline ─────────────────────────────────────────────────────────

def compute_momentum_baseline(prices_df: pd.DataFrame,
                               signal_dates: list,
                               k: int = 10,
                               lookback: int = 5) -> dict:
    """Momentum baseline: Top-K tickers ranked by past `lookback`-day return."""
    sig_strs = [d if isinstance(d, str) else d.isoformat() for d in signal_dates]
    if not sig_strs:
        return {}

    # Use string keys throughout to avoid date/Timestamp hash mismatch
    price_pivot = prices_df.pivot(index="date", columns="ticker", values="adj_close")
    price_pivot.index = price_pivot.index.astype(str)
    price_idx = set(price_pivot.index)

    min_d = min(sig_strs)
    max_d = max(sig_strs)

    # Build NYSE calendar once
    schedule = NYSE.schedule(
        start_date=(date.fromisoformat(min_d) - timedelta(days=lookback * 3)).isoformat(),
        end_date=max_d,
    )
    cal_strs = [str(d.date()) for d in mcal.date_range(schedule, frequency="1D")]
    cal_idx  = {d: i for i, d in enumerate(cal_strs)}

    baselines = {}
    for ds in sig_strs:
        i = cal_idx.get(ds)
        if i is None or i < lookback:
            continue
        past_str = cal_strs[i - lookback]

        if ds not in price_idx or past_str not in price_idx:
            continue

        p_now  = price_pivot.loc[ds]
        p_past = price_pivot.loc[past_str]
        momentum = ((p_now - p_past) / p_past.replace(0, np.nan)).dropna()
        top_k = momentum.nlargest(k).index.tolist()
        baselines[ds] = top_k

    return baselines


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start",   required=True, help="YYYY-MM-DD")
    parser.add_argument("--end",     required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    prices = download_prices(args.tickers, start, end)
    print(prices.head(20))
