# config.py
# All pipeline parameters in one place.

# ── Output ────────────────────────────────────────────────────────────────────
K = 10                          # Shortlist size

# ── Tone ──────────────────────────────────────────────────────────────────────
TRIM_RATE = 0.10                # Discard top/bottom 10% of post scores
FINBERT_MODEL = "ProsusAI/finbert"

# ── Attention ─────────────────────────────────────────────────────────────────
ATTN_WINDOW = 10                # Rolling z-score window (trading days)

# ── Credibility ───────────────────────────────────────────────────────────────
CRED_ALPHA = 0.33               # Weight on duplication rate
CRED_BETA  = 0.33               # Weight on author concentration (Herfindahl)
CRED_GAMMA = 0.34               # Weight on burstiness
BURST_WINDOW_MINUTES = 30       # Time window for burstiness calculation
DEDUP_THRESHOLD = 0.8           # Jaccard similarity threshold for near-duplicates

# ── Minimum evidence rule ────────────────────────────────────────────────────
MIN_POSTS = 3                   # Ticker-days with fewer posts are excluded

# ── Model ─────────────────────────────────────────────────────────────────────
TRAIN_SPLIT = 0.65              # ~65% training, ~35% evaluation

# ── Evaluation ────────────────────────────────────────────────────────────────
HORIZONS = [1, 5]               # T+1 and T+5 trading day horizons

# ── Data collection ───────────────────────────────────────────────────────────
SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "SecurityAnalysis",
    "StockMarket",
]
MARKET_CLOSE_HOUR_ET = 16       # Signal cutoff: 16:00 ET (NYSE close)
MAX_POSTS_PER_DAY = 500         # Cap on posts per day

# ── Ticker universe ───────────────────────────────────────────────────────────
TICKER_UNIVERSE_FILE = "data/sp500_tickers.csv"

# ── Data source toggle ────────────────────────────────────────────────────────
DATA_SOURCE = "csv"
CSV_PATH    = "data/reddit_sp500.csv"

# Date ranges for batch training
TRAIN_START = "2020-01-01"
TRAIN_END   = "2022-12-31"
EVAL_START  = "2023-01-01"
EVAL_END    = "2023-12-31"

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DATA_DIR            = "data/raw"
PROCESSED_DATA_DIR      = "data/processed"
MODEL_PATH              = "data/model.pkl"
SCALER_PATH             = "data/scaler.pkl"
PARAMS_PATH             = "data/calibrated_params.json"
COMPANY_TICKER_MAP_PATH = "data/company_to_ticker.json"
