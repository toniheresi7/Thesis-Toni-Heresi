# model.py
# LightGBM model: train, predict, rank, and explain.

import os
import json
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

from config import (
    K, TRAIN_SPLIT, MIN_POSTS,
    MODEL_PATH, SCALER_PATH, PARAMS_PATH
)

# Features used as predictors
FEATURE_COLS = [
    "tone_finbert",         # FinBERT trimmed-mean sentiment
    "upvote_weighted_tone", # sentiment weighted by post upvotes
    "tone_delta",           # change in sentiment vs prior day
    "attn_shock",           # rolling z-score of post volume
    "rel_share",            # ticker's share of total daily posts
    "breadth",              # unique authors / total posts
    "herfindahl",           # author concentration index
    "agreement",            # 1 - normalised entropy of sentiment
    "mom_1d",               # prior 1-day price return
    "mom_5d",               # prior 5-day price return
    "mom_20d",              # prior 20-day price return
]

# LightGBM hyperparameter candidates
LGBM_CANDIDATES = [
    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 15, "min_child_samples": 20},
    {"n_estimators": 200, "learning_rate": 0.02, "max_depth": 4, "num_leaves": 31, "min_child_samples": 20},
    {"n_estimators": 300, "learning_rate": 0.01, "max_depth": 5, "num_leaves": 31, "min_child_samples": 30},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5, "num_leaves": 63, "min_child_samples": 30},
]

# Target variable
TARGET_COL = "ret_t1"


# ── Training ──────────────────────────────────────────────────────────────────

def train(features_df: pd.DataFrame,
          returns_df: pd.DataFrame) -> tuple:
    """Trains LightGBM on the training split; returns (model, scaler, params, train_dates, eval_dates)."""
    # Merge features with returns
    df = features_df.merge(
        returns_df[["date", "ticker", TARGET_COL]],
        on=["date", "ticker"],
        how="inner"
    ).dropna(subset=FEATURE_COLS + [TARGET_COL])

    if df.empty:
        raise ValueError("No data available after merging features and returns.")

    dates = sorted(df["date"].unique())
    n_train = int(len(dates) * TRAIN_SPLIT)
    train_dates = dates[:n_train]
    eval_dates  = dates[n_train:]

    train_df = df[df["date"].isin(train_dates)]

    # Validation block: last 20% of training window for lambda selection
    n_val    = max(1, int(len(train_dates) * 0.2))
    val_dates  = train_dates[-n_val:]
    fit_dates  = train_dates[:-n_val]

    fit_df = train_df[train_df["date"].isin(fit_dates)]
    val_df = train_df[train_df["date"].isin(val_dates)]

    X_fit = fit_df[FEATURE_COLS].fillna(0).values
    y_fit = fit_df[TARGET_COL].values
    X_val = val_df[FEATURE_COLS].fillna(0).values
    y_val = val_df[TARGET_COL].values

    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit)
    X_val_s = scaler.transform(X_val)

    best_params = LGBM_CANDIDATES[0]
    best_score  = -np.inf

    for params in LGBM_CANDIDATES:
        m = LGBMRegressor(**params, random_state=42, verbose=-1)
        m.fit(X_fit_s, y_fit)
        score = m.score(X_val_s, y_val)
        if score > best_score:
            best_score  = score
            best_params = params

    print(f"[model] Selected params={best_params} (val R²={best_score:.4f})")

    # Refit on full training window with best params
    X_train = train_df[FEATURE_COLS].fillna(0).values
    y_train = train_df[TARGET_COL].values
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
    model.fit(X_train_s, y_train)

    return model, scaler, best_params, train_dates, eval_dates


def save_model(model, scaler, params: dict):
    """Saves the frozen model, scaler, and calibrated parameters to disk."""
    os.makedirs("data", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)
    print(f"[model] Saved model → {MODEL_PATH}")
    print(f"[model] Saved scaler → {SCALER_PATH}")
    print(f"[model] Saved params → {PARAMS_PATH}")


def load_model():
    """Loads the frozen model, scaler, and calibrated parameters."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(PARAMS_PATH) as f:
        params = json.load(f)
    return model, scaler, params


# ── Prediction and ranking ────────────────────────────────────────────────────

def predict_and_rank(day_features: pd.DataFrame,
                     model,
                     scaler) -> pd.DataFrame:
    """Predicts, applies credibility adjustment, ranks, and returns Top-K tickers."""
    # Minimum evidence rule
    eligible = day_features[day_features["n_posts"] >= MIN_POSTS].copy()

    if eligible.empty:
        print("[model] No tickers meet the minimum evidence rule today.")
        return pd.DataFrame()

    for col in FEATURE_COLS:
        if col not in eligible.columns:
            eligible[col] = 0.0
    X = eligible[FEATURE_COLS].fillna(0).values
    X_scaled = scaler.transform(X)

    eligible["y_hat"]     = model.predict(X_scaled)
    eligible["adj_score"] = eligible["y_hat"] * eligible["credibility"]

    ranked = eligible.sort_values("adj_score", ascending=False)

    top_k = ranked.head(K).copy()

    return top_k.reset_index(drop=True)


# ── Explanations ──────────────────────────────────────────────────────────────

# Maps feature names to human-readable signal descriptions
_SIGNAL_DESCRIPTIONS = {
    "tone_finbert": {
        "pos": "strong positive FinBERT sentiment",
        "neg": "strongly negative FinBERT sentiment",
        "neu": "neutral FinBERT sentiment"
    },
    "upvote_weighted_tone": {
        "pos": "highly upvoted posts skewing bullish",
        "neg": "highly upvoted posts skewing bearish",
        "neu": ""
    },
    "tone_delta": {
        "pos": "sentiment improving vs prior day",
        "neg": "sentiment deteriorating vs prior day",
        "neu": ""
    },
    "attn_shock": {
        "pos": "unusual spike in discussion volume",
        "neg": "unusually low discussion activity",
        "neu": "normal discussion activity"
    },
    "rel_share": {
        "pos": "high share of today's total Reddit activity",
        "neg": "",
        "neu": ""
    },
    "breadth": {
        "pos": "broad participation from many distinct voices",
        "neg": "concentrated discussion from few authors",
        "neu": ""
    },
    "agreement": {
        "pos": "high agreement in tone across posts",
        "neg": "mixed and contradictory sentiment",
        "neu": ""
    },
    "herfindahl": {
        "pos": "",
        "neg": "highly concentrated author activity (possible coordination)",
        "neu": ""
    },
    "mom_1d": {
        "pos": "positive prior-day price momentum",
        "neg": "negative prior-day price momentum",
        "neu": ""
    },
    "mom_5d": {
        "pos": "positive 5-day price momentum",
        "neg": "negative 5-day price momentum",
        "neu": ""
    },
    "mom_20d": {
        "pos": "positive 20-day price trend",
        "neg": "negative 20-day price trend",
        "neu": ""
    },
}

_CREDIBILITY_LABELS = {
    "high":   (0.70, 1.01),
    "medium": (0.40, 0.70),
    "low":    (-0.01, 0.40),
}


def get_credibility_label(cred_score: float) -> str:
    for label, (lo, hi) in _CREDIBILITY_LABELS.items():
        if lo <= cred_score < hi:
            return label
    return "low"


def generate_explanation(row: pd.Series, model, scaler) -> str:
    """Generates a one-sentence explanation from top contributing features."""
    x = np.array([row.get(f, 0.0) for f in FEATURE_COLS]).reshape(1, -1)
    x_scaled = scaler.transform(x)[0]

    # Rank by importance × |scaled value|
    importances = model.feature_importances_.astype(float)
    directions  = np.sign(x_scaled)
    contribution = importances * np.abs(x_scaled)   # shape: (n_features,)

    top_indices = np.argsort(contribution)[::-1]

    parts = []
    for idx in top_indices:
        if len(parts) == 3:
            break
        feat = FEATURE_COLS[idx]
        dir_ = directions[idx]
        desc = _SIGNAL_DESCRIPTIONS.get(feat, {})

        if dir_ > 0:
            label = desc.get("pos", "")
        elif dir_ < 0:
            label = desc.get("neg", "")
        else:
            continue  # zero scaled value — no direction to report

        if label:
            parts.append(label)

    if not parts:
        return "Ranked based on composite social signal activity."

    sentence = parts[0].capitalize()
    if len(parts) == 2:
        sentence += f", {parts[1]}."
    elif len(parts) == 3:
        sentence += f", {parts[1]}, and {parts[2]}."
    else:
        sentence += "."

    return sentence


def _parse_top_posts(row: pd.Series) -> list:
    """Safely deserialise the top_posts JSON stored in a feature row."""
    raw = row.get("top_posts", "[]")
    if not isinstance(raw, str) or not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []


def build_shortlist(top_k: pd.DataFrame, model, scaler) -> list:
    """Converts Top-K DataFrame to a list of dashboard-ready dicts."""
    shortlist = []
    for rank, (_, row) in enumerate(top_k.iterrows(), start=1):
        shortlist.append({
            "rank":              rank,
            "ticker":            row["ticker"],
            "adj_score":         round(float(row["adj_score"]), 4),
            "credibility_score": round(float(row["credibility"]), 4),
            "credibility_label": get_credibility_label(row["credibility"]),
            "explanation":       generate_explanation(row, model, scaler),
            "n_posts":           int(row["n_posts"]),
            "tone_finbert":      round(float(row.get("tone_finbert", 0)), 4),
            "attn_shock":        round(float(row.get("attn_shock", 0)), 4),
            "agreement":         round(float(row.get("agreement", 0)), 4),
            "top_posts":          _parse_top_posts(row),
        })
    return shortlist


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from features import load_all_features
    from market_data import download_prices, compute_returns
    from datetime import date

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train and save model")
    parser.add_argument("--rank",  help="Rank tickers for a given date YYYY-MM-DD")
    args = parser.parse_args()

    if args.train:
        features_df = load_all_features()
        if features_df.empty:
            print("[model] No features found. Run features.py first.")
        else:
            # Get all unique tickers and dates to download returns
            tickers = features_df["ticker"].unique().tolist()
            dates   = sorted(features_df["date"].unique())
            start   = date.fromisoformat(dates[0])
            end     = date.fromisoformat(dates[-1])

            print(f"[model] Downloading prices for {len(tickers)} tickers...")
            prices_df  = download_prices(tickers, start, end)
            returns_df = compute_returns(prices_df, dates)

            # Merge price momentum features (not stored in processed CSVs)
            from market_data import compute_momentum_features
            mom_df = compute_momentum_features(prices_df, dates)
            features_df = features_df.merge(mom_df, on=["date", "ticker"], how="left")
            for _col in ["mom_1d", "mom_5d", "mom_20d"]:
                features_df[_col] = features_df[_col].fillna(0.0)

            model, scaler, best_params, train_dates, eval_dates = train(features_df, returns_df)
            save_model(model, scaler, {
                "model_type":   "lightgbm",
                "params":       best_params,
                "train_dates":  train_dates,
                "eval_dates":   eval_dates,
                "feature_cols": FEATURE_COLS,
            })

    if args.rank:
        from features import load_all_features
        all_features = load_all_features()
        day_features = all_features[all_features["date"] == args.rank]

        if day_features.empty:
            print(f"[model] No features found for {args.rank}")
        else:
            model, scaler, params = load_model()
            top_k = predict_and_rank(day_features, model, scaler)
            shortlist = build_shortlist(top_k, model, scaler)
            for item in shortlist:
                print(f"{item['rank']:2}. {item['ticker']:6} | "
                      f"cred={item['credibility_label']:6} | {item['explanation']}")
