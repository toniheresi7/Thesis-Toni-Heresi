# evaluate_retrained.py
# Retrains a fresh LightGBM model per experiment on the 65/35 split, then evaluates.
# Usage: PYTHONPATH=src python evaluation/evaluate_retrained.py --data-dir data/processed_stocktwits --output results/stocktwits_retrained_evaluation.txt

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import date as _date, timedelta
from scipy import stats
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

from config import K, HORIZONS, TRAIN_SPLIT, MIN_POSTS
from model import FEATURE_COLS, TARGET_COL, LGBM_CANDIDATES
from market_data import (
    download_prices, compute_returns,
    compute_momentum_features, compute_momentum_baseline,
)


# ── Feature loader ────────────────────────────────────────────────────────────

def load_features_from_dir(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    frames = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join(data_dir, fname)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Train model from scratch on this experiment's features ────────────────────

def train_model(features_df, returns_df):
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
    eval_dates = dates[n_train:]

    print(f"[retrained] Total dates: {len(dates)}")
    print(f"[retrained] Train dates: {len(train_dates)} ({train_dates[0]} to {train_dates[-1]})")
    print(f"[retrained] Eval dates:  {len(eval_dates)} ({eval_dates[0]} to {eval_dates[-1]})")

    train_df = df[df["date"].isin(train_dates)]

    n_val = max(1, int(len(train_dates) * 0.2))
    val_dates = train_dates[-n_val:]
    fit_dates = train_dates[:-n_val]

    fit_df = train_df[train_df["date"].isin(fit_dates)]
    val_df = train_df[train_df["date"].isin(val_dates)]

    X_fit = fit_df[FEATURE_COLS].fillna(0).values
    y_fit = fit_df[TARGET_COL].values
    X_val = val_df[FEATURE_COLS].fillna(0).values
    y_val = val_df[TARGET_COL].values

    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit)
    X_val_s = scaler.transform(X_val)

    # Hyperparameter selection
    best_params = LGBM_CANDIDATES[0]
    best_score = -np.inf

    for params in LGBM_CANDIDATES:
        m = LGBMRegressor(**params, random_state=42, verbose=-1)
        m.fit(X_fit_s, y_fit)
        score = m.score(X_val_s, y_val)
        if score > best_score:
            best_score = score
            best_params = params

    print(f"[retrained] Selected params={best_params} (val R²={best_score:.4f})")

    X_train = train_df[FEATURE_COLS].fillna(0).values
    y_train = train_df[TARGET_COL].values
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
    model.fit(X_train_s, y_train)

    return model, scaler, best_params, train_dates, eval_dates


# ── Predict and rank ──────────────────────────────────────────────────────────

def predict_and_rank(day_features, model, scaler):
    eligible = day_features[day_features["n_posts"] >= MIN_POSTS].copy()
    if eligible.empty:
        return pd.DataFrame()
    for col in FEATURE_COLS:
        if col not in eligible.columns:
            eligible[col] = 0.0
    X = eligible[FEATURE_COLS].fillna(0).values
    X_scaled = scaler.transform(X)
    eligible["y_hat"] = model.predict(X_scaled)
    eligible["adj_score"] = eligible["y_hat"] * eligible["credibility"]
    ranked = eligible.sort_values("adj_score", ascending=False)
    return ranked.head(K).copy().reset_index(drop=True)


# ── Shortlist returns ─────────────────────────────────────────────────────────

def shortlist_returns(shortlists, returns_df):
    rows = []
    ret_pivot_t1 = returns_df.pivot(index="date", columns="ticker", values="ret_t1")
    ret_pivot_t5 = returns_df.pivot(index="date", columns="ticker", values="ret_t5")

    for d, tickers in shortlists.items():
        for horizon, pivot in [(1, ret_pivot_t1), (5, ret_pivot_t5)]:
            if d not in pivot.index:
                continue
            rets = pivot.loc[d, [t for t in tickers if t in pivot.columns]].dropna()
            if rets.empty:
                continue
            rows.append({
                "date": d,
                "horizon": horizon,
                "mean_ret": float(rets.mean()),
                "mean_abs_ret": float(rets.abs().mean()),
                "n_tickers": len(rets),
            })
    return pd.DataFrame(rows)


# ── Turnover ──────────────────────────────────────────────────────────────────

def compute_turnover(shortlists):
    dates = sorted(shortlists.keys())
    rows = []
    for i in range(1, len(dates)):
        prev = set(shortlists[dates[i - 1]])
        curr = set(shortlists[dates[i]])
        changed = len(prev.symmetric_difference(curr))
        rows.append({"date": dates[i], "turnover": changed / K})
    return pd.DataFrame(rows)


# ── Information Coefficient ───────────────────────────────────────────────────

def compute_ic(features_df, returns_df, eval_dates, model, scaler):
    ret_pivot = returns_df.pivot(index="date", columns="ticker", values="ret_t1")
    rows = []

    for d in eval_dates:
        day_f = features_df[features_df["date"] == d].copy()
        if day_f.empty or d not in ret_pivot.index:
            continue
        eligible = day_f[day_f["n_posts"] >= 3]
        if len(eligible) < 5:
            continue
        X = eligible[FEATURE_COLS].fillna(0).values
        X_scaled = scaler.transform(X)
        scores = model.predict(X_scaled) * eligible["credibility"].values
        tickers = eligible["ticker"].values
        rets = ret_pivot.loc[d, [t for t in tickers if t in ret_pivot.columns]]
        aligned = [
            (scores[i], rets[t])
            for i, t in enumerate(tickers)
            if t in rets.index and not np.isnan(rets[t])
        ]
        if len(aligned) < 5:
            continue
        pred_scores, realised = zip(*aligned)
        rho, _ = stats.spearmanr(pred_scores, realised)
        rows.append({"date": d, "ic": float(rho), "n_tickers": len(aligned)})

    df = pd.DataFrame(rows)
    if not df.empty:
        df.attrs["mean_ic"] = float(df["ic"].mean())
        df.attrs["ic_std"] = float(df["ic"].std())
        df.attrs["ir"] = float(df["ic"].mean() / df["ic"].std()) if df["ic"].std() > 0 else 0.0
        df.attrs["pct_positive"] = float((df["ic"] > 0).mean())
    return df


# ── Hit rate ──────────────────────────────────────────────────────────────────

def compute_hit_rate(shortlists, returns_df):
    ret_pivot_t1 = returns_df.pivot(index="date", columns="ticker", values="ret_t1")
    ret_pivot_t5 = returns_df.pivot(index="date", columns="ticker", values="ret_t5")
    hits = {1: 0, 5: 0}
    totals = {1: 0, 5: 0}
    for d, tickers in shortlists.items():
        for horizon, pivot in [(1, ret_pivot_t1), (5, ret_pivot_t5)]:
            if d not in pivot.index:
                continue
            rets = pivot.loc[d, [t for t in tickers if t in pivot.columns]].dropna()
            hits[horizon] += int((rets > 0).sum())
            totals[horizon] += len(rets)
    return {
        1: hits[1] / totals[1] if totals[1] > 0 else float("nan"),
        5: hits[5] / totals[5] if totals[5] > 0 else float("nan"),
        "n_obs_t1": totals[1],
        "n_obs_t5": totals[5],
    }


# ── Attention-only baseline ──────────────────────────────────────────────────

def attention_only_baseline(features_df, eval_dates):
    baselines = {}
    for d in eval_dates:
        day_f = features_df[features_df["date"] == d]
        eligible = day_f[day_f["n_posts"] >= 3].copy()
        if eligible.empty:
            continue
        ranked = eligible.sort_values("attn_shock", ascending=False)
        baselines[d] = ranked.head(K)["ticker"].tolist()
    return baselines


# ── Ablation (retrains per removed feature) ──────────────────────────────────

def ablation(features_df, returns_df, train_dates, eval_dates, lgbm_params):
    """Ablation that retrains a fresh model for each removed feature."""
    # Full model for baseline comparison
    df = features_df.merge(
        returns_df[["date", "ticker", TARGET_COL]],
        on=["date", "ticker"], how="inner"
    ).dropna(subset=FEATURE_COLS + [TARGET_COL])

    train_df = df[df["date"].isin(train_dates)]

    # Train full model
    X_full = train_df[FEATURE_COLS].fillna(0).values
    y_full = train_df[TARGET_COL].values
    full_scaler = StandardScaler()
    X_full_s = full_scaler.fit_transform(X_full)
    full_model = LGBMRegressor(**lgbm_params, random_state=42, verbose=-1)
    full_model.fit(X_full_s, y_full)

    # Full model shortlists
    full_shortlists = {}
    for d in eval_dates:
        day_f = features_df[features_df["date"] == d]
        if day_f.empty:
            continue
        top_k = predict_and_rank(day_f, full_model, full_scaler)
        if not top_k.empty:
            full_shortlists[d] = top_k["ticker"].tolist()

    full_rets = shortlist_returns(full_shortlists, returns_df)
    full_mean = full_rets.groupby("horizon")["mean_ret"].mean().to_dict()

    rows = []
    for removed in FEATURE_COLS:
        ablated_cols = [f for f in FEATURE_COLS if f != removed]

        abl_train = df[df["date"].isin(train_dates)]
        X_train = abl_train[ablated_cols].fillna(0).values
        y_train = abl_train[TARGET_COL].values

        abl_scaler = StandardScaler()
        X_s = abl_scaler.fit_transform(X_train)
        abl_model = LGBMRegressor(**lgbm_params, random_state=42, verbose=-1)
        abl_model.fit(X_s, y_train)

        abl_shortlists = {}
        for d in eval_dates:
            day_f = features_df[features_df["date"] == d].copy()
            if day_f.empty:
                continue
            X = day_f[ablated_cols].fillna(0).values
            Xs = abl_scaler.transform(X)
            day_f = day_f.copy()
            day_f["y_hat"] = abl_model.predict(Xs)
            day_f["adj_score"] = day_f["y_hat"] * day_f["credibility"]
            eligible = day_f[day_f["n_posts"] >= 3].sort_values("adj_score", ascending=False)
            if not eligible.empty:
                abl_shortlists[d] = eligible.head(K)["ticker"].tolist()

        abl_rets = shortlist_returns(abl_shortlists, returns_df)
        abl_mean = abl_rets.groupby("horizon")["mean_ret"].mean().to_dict()

        for h in HORIZONS:
            rows.append({
                "removed_feature": removed,
                "horizon": h,
                "mean_ret_full": full_mean.get(h, np.nan),
                "mean_ret_ablated": abl_mean.get(h, np.nan),
                "delta": abl_mean.get(h, np.nan) - full_mean.get(h, np.nan),
            })

    return pd.DataFrame(rows)


# ── Full evaluation ──────────────────────────────────────────────────────────

def run_evaluation(features_df, returns_df, prices_df, eval_dates, model, scaler,
                   train_dates, lgbm_params):
    # Platform shortlists
    platform_shortlists = {}
    for d in eval_dates:
        day_f = features_df[features_df["date"] == d]
        if day_f.empty:
            continue
        top_k = predict_and_rank(day_f, model, scaler)
        if not top_k.empty:
            platform_shortlists[d] = top_k["ticker"].tolist()

    # Baselines
    momentum_shortlists = compute_momentum_baseline(prices_df, eval_dates, k=K)
    attention_shortlists = attention_only_baseline(features_df, eval_dates)

    # Returns
    platform_rets = shortlist_returns(platform_shortlists, returns_df)
    momentum_rets = shortlist_returns(momentum_shortlists, returns_df)
    attention_rets = shortlist_returns(attention_shortlists, returns_df)

    # Turnover
    turnover_df = compute_turnover(platform_shortlists)

    # IC
    ic_df = compute_ic(features_df, returns_df, eval_dates, model, scaler)

    # Hit rate
    hit_rates = compute_hit_rate(platform_shortlists, returns_df)

    # Ablation
    print("[retrained] Running ablation analysis (this may take a few minutes)...")
    ablation_df = ablation(features_df, returns_df, train_dates, eval_dates, lgbm_params)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (RETRAINED MODEL)")
    print("=" * 60)

    for horizon in HORIZONS:
        p = platform_rets[platform_rets["horizon"] == horizon]["mean_ret"].mean()
        m = momentum_rets[momentum_rets["horizon"] == horizon]["mean_ret"].mean()
        a = attention_rets[attention_rets["horizon"] == horizon]["mean_ret"].mean()
        print(f"\nT+{horizon} Mean Return:")
        print(f"  Platform:       {p:.4f}")
        print(f"  Momentum:       {m:.4f}")
        print(f"  Attention-only: {a:.4f}")

    print(f"\nMean daily Top-K turnover: {turnover_df['turnover'].mean():.2%}")

    if not ic_df.empty:
        print(f"\nInformation Coefficient (Spearman, T+1):")
        print(f"  Mean IC:          {ic_df.attrs['mean_ic']:+.4f}")
        print(f"  IC Std:           {ic_df.attrs['ic_std']:.4f}")
        print(f"  IR (IC/Std):      {ic_df.attrs['ir']:+.4f}")
        print(f"  Days IC > 0:      {ic_df.attrs['pct_positive']:.1%}  ({int(ic_df['ic'].gt(0).sum())}/{len(ic_df)} days)")

    print(f"\nHit Rate (shortlist stocks with positive return):")
    print(f"  T+1: {hit_rates[1]:.1%}  (n={hit_rates['n_obs_t1']})")
    print(f"  T+5: {hit_rates[5]:.1%}  (n={hit_rates['n_obs_t5']})")

    print(f"\nModel: LightGBM (retrained on this experiment's features)")
    print(f"  Params: {lgbm_params}")
    print(f"  Train: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
    print(f"  Eval:  {len(eval_dates)} days ({eval_dates[0]} to {eval_dates[-1]})")

    print("\nAblation results (delta vs full model at T+1):")
    abl_t1 = ablation_df[ablation_df["horizon"] == 1]
    for _, row in abl_t1.sort_values("delta").iterrows():
        print(f"  Remove {row['removed_feature']:25}: delta={row['delta']:+.4f}")

    return {
        "platform_returns": platform_rets,
        "momentum_returns": momentum_rets,
        "attention_returns": attention_rets,
        "turnover": turnover_df,
        "ic": ic_df,
        "hit_rates": hit_rates,
        "ablation": ablation_df,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate with a retrained LightGBM model per experiment."
    )
    parser.add_argument("--data-dir", required=True,
                        help="Path to folder containing processed feature CSVs")
    parser.add_argument("--output", required=True,
                        help="Path to save evaluation results text file")
    args = parser.parse_args()

    print(f"[retrained] Loading features from: {args.data_dir}")
    features_df = load_features_from_dir(args.data_dir)

    if features_df.empty:
        print("[retrained] No features found.")
        sys.exit(1)

    tickers = features_df["ticker"].unique().tolist()
    dates = sorted(features_df["date"].unique())
    start = _date.fromisoformat(dates[0])
    end = _date.fromisoformat(dates[-1])

    print(f"[retrained] Downloading prices for {len(tickers)} tickers...")
    prices_df = download_prices(tickers, start, end)
    returns_df = compute_returns(prices_df, dates)

    # Add momentum features
    mom_df = compute_momentum_features(prices_df, dates)
    features_df = features_df.merge(mom_df, on=["date", "ticker"], how="left")
    for _col in ["mom_1d", "mom_5d", "mom_20d"]:
        features_df[_col] = features_df[_col].fillna(0.0)

    # Train fresh model on this experiment's features
    print("[retrained] Training fresh LightGBM model...")
    model, scaler, best_params, train_dates, eval_dates = train_model(features_df, returns_df)

    # Run evaluation and tee output to file
    import io
    from contextlib import redirect_stdout

    output_buf = io.StringIO()

    class TeeWriter:
        def __init__(self, *writers):
            self.writers = writers
        def write(self, s):
            for w in self.writers:
                w.write(s)
        def flush(self):
            for w in self.writers:
                w.flush()

    tee = TeeWriter(sys.stdout, output_buf)

    with redirect_stdout(tee):
        results = run_evaluation(
            features_df, returns_df, prices_df,
            eval_dates, model, scaler, train_dates, best_params
        )

    # Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(output_buf.getvalue())
    print(f"\n[retrained] Results saved to: {args.output}")
