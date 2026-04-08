# backend/main.py
# SignalScreen FastAPI backend — reads real pipeline output from data/ directory

import sys
import os
from datetime import datetime, timezone, date as _date_type
import json
import math

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Project root: set CWD so relative paths in pipeline modules resolve ────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from features import load_all_features
from model import (
    load_model, predict_and_rank, build_shortlist,
    FEATURE_COLS, get_credibility_label, generate_explanation,
)
from market_data import download_prices, compute_momentum_features
from config import CRED_ALPHA, CRED_BETA, CRED_GAMMA, RAW_DATA_DIR

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SignalScreen API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static maps ───────────────────────────────────────────────────────────────
_SECTOR = {
    "AAPL": "Technology",   "MSFT": "Technology",  "NVDA": "Technology",
    "AMD":  "Technology",   "GOOGL":"Communication Services",
    "META": "Communication Services", "NFLX": "Communication Services",
    "TSLA": "Consumer Discretionary", "AMZN": "Consumer Discretionary",
    "SPY":  "ETF",
}

_FEAT_LABELS = {
    "tone_finbert":         "FinBERT Sentiment",
    "upvote_weighted_tone": "Upvote-Weighted Tone",
    "tone_delta":           "Sentiment Change",
    "attn_shock":           "Attention Shock",
    "rel_share":            "Relative Share",
    "breadth":              "Author Breadth",
    "herfindahl":           "Author Concentration",
    "agreement":            "Tone Agreement",
    "mom_1d":               "1-Day Momentum",
    "mom_5d":               "5-Day Momentum",
    "mom_20d":              "20-Day Momentum",
}

# ── Pipeline state (populated at startup) ─────────────────────────────────────
_model       = None
_scaler      = None
_params      = None
_all_feats   = None    # full history DataFrame with predictions
_today_str   = None    # most recent date string
_shortlist   = []      # list of dicts from build_shortlist()
_by_ticker   = {}      # ticker → shortlist dict
_today_feats = {}      # ticker → full features dict (incl. dup_rate, herfindahl…)
_raw_by_ticker = {}    # ticker → list of raw post dicts for today (fallback pool)


def _load_raw_posts_by_ticker():
    global _raw_by_ticker
    if not _today_str:
        return
    raw_path = os.path.join(RAW_DATA_DIR, f"{_today_str}.json")
    if not os.path.exists(raw_path):
        return
    try:
        with open(raw_path) as f:
            posts = json.load(f)
        by_ticker: dict = {}
        for p in posts:
            text = p.get("text", "")
            seen = set()
            for token in text.split():
                if token.startswith("$") and len(token) > 1:
                    tkr = token[1:].upper().rstrip(".,;:!?")
                    if tkr and tkr not in seen:
                        by_ticker.setdefault(tkr, []).append(p)
                        seen.add(tkr)
        _raw_by_ticker = by_ticker
        print(f"[backend] Raw posts indexed: {len(by_ticker)} tickers.")
    except Exception as e:
        print(f"[backend] Warning: could not index raw posts: {e}")


def _load_state():
    global _model, _scaler, _params, _all_feats
    global _today_str, _shortlist, _by_ticker, _today_feats

    try:
        _model, _scaler, _params = load_model()
        print("[backend] Model loaded successfully.")
    except FileNotFoundError:
        print("[backend] Model not found — run: python src/model.py --train")

    _all_feats = load_all_features()
    if _all_feats is None or _all_feats.empty:
        print("[backend] No feature data found in data/processed/")
        return

    mom_cols = ["mom_1d", "mom_5d", "mom_20d"]
    if any(c not in _all_feats.columns for c in mom_cols):
        try:
            tickers    = _all_feats["ticker"].unique().tolist()
            all_dates  = sorted(_all_feats["date"].unique())
            start_date = _date_type.fromisoformat(all_dates[0])
            end_date   = _date_type.fromisoformat(all_dates[-1])
            print("[backend] Downloading prices for momentum features …", flush=True)
            prices_df = download_prices(tickers, start_date, end_date)
            mom_df    = compute_momentum_features(prices_df, all_dates)
            _all_feats = _all_feats.merge(mom_df, on=["date", "ticker"], how="left")
            for col in mom_cols:
                _all_feats[col] = _all_feats[col].fillna(0.0)
            print(f"[backend] Momentum features merged ({len(mom_df)} rows).")
        except Exception as e:
            print(f"[backend] Warning: momentum feature computation failed: {e}")
            for col in mom_cols:
                _all_feats[col] = 0.0

    if _model is not None and _scaler is not None:
        for col in FEATURE_COLS:
            if col not in _all_feats.columns:
                _all_feats[col] = 0.0
        X = _all_feats[FEATURE_COLS].fillna(0).values
        X_scaled = _scaler.transform(X)
        _all_feats = _all_feats.copy()
        _all_feats["y_hat"]     = _model.predict(X_scaled)
        _all_feats["adj_score"] = _all_feats["y_hat"] * _all_feats["credibility"]

    dates      = sorted(_all_feats["date"].unique())
    _today_str = dates[-1]
    print(f"[backend] Data loaded: {len(dates)} days, latest = {_today_str}")

    today_df = _all_feats[_all_feats["date"] == _today_str].copy()
    for _, row in today_df.iterrows():
        _today_feats[row["ticker"]] = row.to_dict()

    if _model is not None:
        top_k = predict_and_rank(today_df, _model, _scaler)
        if not top_k.empty:
            _shortlist = build_shortlist(top_k, _model, _scaler)
            _by_ticker = {s["ticker"]: s for s in _shortlist}
            print(f"[backend] Shortlist: {[s['ticker'] for s in _shortlist]}")
        else:
            print("[backend] No tickers passed the minimum evidence rule today.")

    _load_raw_posts_by_ticker()


_load_state()


# ── Helper: real credibility breakdown ───────────────────────────────────────

def _cred_breakdown(ticker: str) -> dict:
    row = _today_feats.get(ticker, {})
    dup_rate    = float(row.get("dup_rate",    0.0))
    herfindahl  = float(row.get("herfindahl",  0.0))
    burstiness  = float(row.get("burstiness",  0.0))
    credibility = float(row.get("credibility", 0.5))

    dup_penalty   = round(CRED_ALPHA * dup_rate,    4)
    herf_penalty  = round(CRED_BETA  * herfindahl,  4)
    burst_penalty = round(CRED_GAMMA * burstiness,  4)
    total_penalty = round(dup_penalty + herf_penalty + burst_penalty, 4)

    return {
        "dup_rate":      round(dup_rate,    3),
        "herfindahl":    round(herfindahl,  3),
        "burstiness":    round(burstiness,  3),
        "dup_penalty":   dup_penalty,
        "herf_penalty":  herf_penalty,
        "burst_penalty": burst_penalty,
        "total_penalty": total_penalty,
        "composite":     round(credibility, 4),
    }


# ── Helper: real feature contributions ───────────────────────────────────────

def _feature_contributions(ticker: str) -> list:
    row = _today_feats.get(ticker, {})
    if _model is None or _scaler is None or not row:
        return []

    x        = np.array([row.get(f, 0.0) for f in FEATURE_COLS]).reshape(1, -1)
    x_scaled = _scaler.transform(x)[0]

    importances = _model.feature_importances_.astype(float)
    total_imp   = importances.sum() or 1.0
    contribs    = (importances / total_imp) * np.sign(x_scaled)

    return [
        {
            "feature":      f,
            "label":        _FEAT_LABELS.get(f, f),
            "value":        round(float(row.get(f, 0.0)), 4),
            "contribution": round(float(contribs[i]) * 1000, 4),
        }
        for i, f in enumerate(FEATURE_COLS)
    ]


# ── Helper: real ticker history from processed CSVs ───────────────────────────

def _ticker_history(ticker: str) -> list:
    if _all_feats is None or _all_feats.empty:
        return []

    sub = (
        _all_feats[_all_feats["ticker"] == ticker]
        .sort_values("date")
        .tail(30)
    )

    result = []
    for _, row in sub.iterrows():
        result.append({
            "date":         row["date"],
            "adj_score":    round(float(row.get("adj_score", 0.0)), 4),
            "tone_finbert": round(float(row.get("tone_finbert", 0.0)), 4),
            "n_posts":      int(row.get("n_posts", 0)),
            "attn_shock":   round(float(row.get("attn_shock", 0.0)), 4),
            "credibility":  round(float(row.get("credibility", 0.0)), 4),
        })
    return result


# ── Helper: time ago string ───────────────────────────────────────────────────

def _time_ago(created_utc: float) -> str:
    if not created_utc:
        return "recently"
    now  = datetime.now(timezone.utc).timestamp()
    diff = now - float(created_utc)
    if diff < 3600:
        return f"{int(diff / 60)}m ago"
    elif diff < 86400:
        return f"{int(diff / 3600)}h ago"
    else:
        return f"{int(diff / 86400)}d ago"


# ── Helper: simple sentiment classification ───────────────────────────────────

def _classify_sentiment(text: str, default_tone: float) -> str:
    t = text.lower()
    bull = sum(1 for w in (
        "bullish", "buying", "bought", "long", "calls", "breakout",
        "strong", "crushed", "beat", "raised", "loading", "conviction",
    ) if w in t)
    bear = sum(1 for w in (
        "bearish", "selling", "sold", "short", "puts", "disaster",
        "avoid", "miss", "cut", "dump", "broken", "going to zero",
    ) if w in t)
    if bull > bear:
        return "bullish"
    if bear > bull:
        return "bearish"
    if default_tone > 0.05:
        return "bullish"
    if default_tone < -0.05:
        return "bearish"
    return "neutral"


_SUB_DISPLAY = {
    "wallstreetbets":   "r/wallstreetbets",
    "stocks":           "r/stocks",
    "investing":        "r/investing",
    "StockMarket":      "r/StockMarket",
    "SecurityAnalysis": "r/SecurityAnalysis",
}


def _enriched_top_posts(posts: list, tone: float, is_specific: bool = True) -> list:
    result = []
    for p in posts:
        txt    = p.get("text", "")
        author = p.get("author", "unknown")
        sub    = p.get("subreddit", "")
        result.append({
            "title":       p.get("title", ""),
            "text":        txt,
            "author":      author if author.startswith("u/") else f"u/{author}",
            "subreddit":   _SUB_DISPLAY.get(sub, f"r/{sub}"),
            "score":       int(p.get("score", 0)),
            "sentiment":   _classify_sentiment(txt, tone),
            "time_ago":    _time_ago(p.get("created_utc", 0)),
            "is_specific": is_specific,
        })
    return result


def _get_top_posts(ticker: str, specific_posts: list, tone: float) -> list:
    if specific_posts:
        return _enriched_top_posts(specific_posts, tone, is_specific=True)
    fallback = sorted(
        _raw_by_ticker.get(ticker, []),
        key=lambda p: p.get("score", 0),
        reverse=True,
    )[:3]
    return _enriched_top_posts(fallback, tone, is_specific=False)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "model":   "SignalScreen v2.0",
        "date":    _today_str or _date_type.today().isoformat(),
        "tickers": len(_shortlist),
        "has_model": _model is not None,
        "days_of_history": len(_all_feats["date"].unique()) if (_all_feats is not None and not _all_feats.empty) else 0,
    }


@app.get("/shortlist")
def get_shortlist():
    if not _shortlist:
        raise HTTPException(
            status_code=503,
            detail="Shortlist not available. Run the pipeline first: "
                   "python run_pipeline.py --date YYYY-MM-DD --mock"
        )
    return {
        "date":      _today_str,
        "tickers":   len(_shortlist),
        "shortlist": _shortlist,
    }


@app.get("/stock/{ticker}")
def get_stock(ticker: str):
    t = ticker.upper()
    if t not in _by_ticker:
        raise HTTPException(status_code=404, detail=f"Ticker {t} not in today's shortlist")

    row  = _by_ticker[t]
    tone = float(row.get("tone_finbert", 0.0))

    return {
        "ticker":                 t,
        "rank":                   row["rank"],
        "sector":                 _SECTOR.get(t, "Equity"),
        "adj_score":              row["adj_score"],
        "credibility_score":      row["credibility_score"],
        "credibility_label":      row["credibility_label"],
        "explanation":            row["explanation"],
        "n_posts":                row["n_posts"],
        "tone_finbert":           tone,
        "attn_shock":             float(row.get("attn_shock", 0.0)),
        "agreement":              float(row.get("agreement", 0.0)),
        "feature_contributions":  _feature_contributions(t),
        "credibility_breakdown":  _cred_breakdown(t),
        "top_posts":              _get_top_posts(t, row.get("top_posts", []), tone),
        "history":                _ticker_history(t),
    }
