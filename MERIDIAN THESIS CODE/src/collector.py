# collector.py
# Reddit data collection; mock data generation for testing.

import json
import os
import random
import string
from datetime import datetime, timedelta, timezone
import pytz

# PRAW is imported lazily so the file doesn't crash if it isn't installed yet
try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False

from config import (
    SUBREDDITS, MARKET_CLOSE_HOUR_ET, MAX_POSTS_PER_DAY, RAW_DATA_DIR,
)

EASTERN = pytz.timezone("America/New_York")


# ── Signal window ─────────────────────────────────────────────────────────────

def get_signal_window(date: datetime.date):
    """Returns (start_utc, end_utc) for the 16:00 ET-to-16:00 ET signal window."""
    prev_day = date - timedelta(days=1)
    start_et = EASTERN.localize(
        datetime(prev_day.year, prev_day.month, prev_day.day,
                 MARKET_CLOSE_HOUR_ET, 0, 0)
    )
    end_et = EASTERN.localize(
        datetime(date.year, date.month, date.day,
                 MARKET_CLOSE_HOUR_ET, 0, 0)
    )
    return start_et.astimezone(pytz.utc), end_et.astimezone(pytz.utc)


# ── Live collection ───────────────────────────────────────────────────────────

def collect(date: datetime.date,
            client_id: str,
            client_secret: str,
            user_agent: str) -> str:
    """Fetches Reddit posts for the signal window; saves to data/raw/."""
    if not PRAW_AVAILABLE:
        raise ImportError("praw is not installed. Run: pip install praw")

    start_utc, end_utc = get_signal_window(date)
    start_ts = start_utc.timestamp()
    end_ts   = end_utc.timestamp()

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    posts = []
    for subreddit_name in SUBREDDITS:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.new(limit=1000):
            created = submission.created_utc
            if created < start_ts:
                break                       # Reddit returns newest-first; stop early
            if created > end_ts:
                continue
            posts.append({
                "post_id":   submission.id,
                "subreddit": subreddit_name,
                "author":    str(submission.author),
                "text":      submission.title + " " + (submission.selftext or ""),
                "created_utc": created,
                "score":     submission.score,
            })

    # Even-time sampling if over cap
    if len(posts) > MAX_POSTS_PER_DAY:
        posts = _even_sample(posts, MAX_POSTS_PER_DAY)

    return _save(posts, date)


# ── Public helper — save a merged post list ───────────────────────────────────

def save_raw_posts(posts: list, date: datetime.date) -> str:
    """Saves a merged post list to data/raw/YYYY-MM-DD.json."""
    return _save(posts, date)


# ── Mock data (use when API is not yet approved) ──────────────────────────────

# A small set of realistic ticker mentions for mock posts
_MOCK_TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "SPY"]

_BULLISH_TEMPLATES = [
    "{ticker} is looking very strong today, massive volume spike",
    "Bought more {ticker} on the dip — long-term conviction, not selling",
    "{ticker} earnings absolutely crushed it, huge beat on revenue",
    "Bullish on {ticker} going into next week, chart looks perfect",
    "Why is nobody talking about {ticker}? This is a massive opportunity",
    "{ticker} just broke out of resistance — next stop much higher",
    "All in on {ticker}, the fundamentals are undeniable right now",
    "{ticker} insiders have been buying aggressively — follow the smart money",
    "Short squeeze incoming on {ticker}? Short interest at multi-year highs",
    "{ticker} is my top pick for Q4, growth story intact",
    "Loading up {ticker} calls before earnings, setup looks pristine",
    "{ticker} guidance was raised — analysts are way too conservative",
]

_BEARISH_TEMPLATES = [
    "Sold all my {ticker} — earnings were a disaster, avoid",
    "{ticker} getting destroyed today, the trend is clearly broken",
    "Short {ticker} here, valuation is completely disconnected from reality",
    "{ticker} is going to zero — debt load is unsustainable",
    "Dumped my entire {ticker} position, management has lost the plot",
    "{ticker} revenue miss was ugly — guidance cut is the real concern",
    "Put options on {ticker} are printing, chart screams distribution",
    "{ticker} insiders have been selling every single rally — red flag",
    "Avoid {ticker} until they report again, too much uncertainty",
    "{ticker} — the hype cycle is over, reality is setting in fast",
    "Trimmed {ticker} significantly, risk/reward is terrible at these levels",
    "{ticker} margins compressed again, competition is eating their lunch",
]

_NEUTRAL_TEMPLATES = [
    "{ticker} — neutral on this one, waiting for more data",
    "Anyone else watching {ticker}? Not sure which way it breaks",
    "{ticker} holding support but hasn't confirmed direction yet",
    "Mixed signals on {ticker} — earnings fine but guidance was vague",
    "{ticker} in a tight range, probably consolidating before next move",
    "What's the consensus on {ticker}? I see valid cases both ways",
    "{ticker} moving with the broader market, no real edge here today",
    "Watching {ticker} at this level — needs volume confirmation",
    "{ticker} is interesting but I'm sitting on my hands for now",
    "Not touching {ticker} until after the Fed meeting, too much noise",
]


def generate_mock_data(date: datetime.date, n_posts: int = None) -> str:
    """Generates a synthetic Reddit dataset with deterministic per-date seed."""
    start_utc, end_utc = get_signal_window(date)
    start_ts = start_utc.timestamp()
    end_ts   = end_utc.timestamp()
    window   = end_ts - start_ts

    # Unique seed per calendar day — different data every day
    rng = random.Random(date.toordinal())

    # ── Daily volume: varies between 60 and 280 posts ─────────────────────────
    if n_posts is None:
        n_posts = rng.randint(60, 280)

    # ── Pick 1–2 "hot" tickers that dominate discussion today ────────────────
    n_hot = rng.randint(1, 2)
    hot_tickers = rng.sample(_MOCK_TICKERS, k=n_hot)

    # ── Per-ticker sentiment bias for this day ────────────────────────────────
    ticker_bias = {}
    for ticker in _MOCK_TICKERS:
        if ticker in hot_tickers:
            # Hot tickers get a strong directional lean (mostly bull/bear events)
            ticker_bias[ticker] = rng.choices(
                ["bullish", "bearish", "neutral"], weights=[50, 40, 10]
            )[0]
        else:
            ticker_bias[ticker] = rng.choices(
                ["bullish", "bearish", "neutral"], weights=[35, 30, 35]
            )[0]

    # ── Weighted mention pool: hot tickers get 5x pull ───────────────────────
    weights = [5.0 if t in hot_tickers else 1.0 for t in _MOCK_TICKERS]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    # ── Burst window for hot tickers: cluster in a random 2-hour block ───────
    burst_center = rng.uniform(0.15, 0.75)   # fraction through the 24-hour window
    burst_width  = 2 * 3600 / window         # 2-hour window as fraction

    posts = []
    for i in range(n_posts):
        ticker = rng.choices(_MOCK_TICKERS, weights=weights, k=1)[0]
        bias   = ticker_bias[ticker]

        if bias == "bullish":
            template = rng.choice(_BULLISH_TEMPLATES)
        elif bias == "bearish":
            template = rng.choice(_BEARISH_TEMPLATES)
        else:
            template = rng.choice(_NEUTRAL_TEMPLATES)

        author = "user_" + "".join(rng.choices(string.ascii_lowercase, k=6))

        # Hot tickers burst into a 2-hour window; others spread uniformly
        if ticker in hot_tickers:
            frac = rng.gauss(burst_center, burst_width / 4)
            frac = max(0.0, min(0.99, frac))
            t = start_ts + frac * window
        else:
            t = start_ts + rng.uniform(0, window)

        posts.append({
            "post_id":    f"mock_{i:05d}",
            "subreddit":  rng.choice(SUBREDDITS),
            "author":     author,
            "text":       template.format(ticker=f"${ticker}"),
            "created_utc": t,
            "score":      rng.randint(10, 5000) if ticker in hot_tickers
                          else rng.randint(1, 300),
        })

    # ── Sprinkle near-duplicate posts for credibility pipeline ───────────────
    n_dups = rng.randint(3, 12)
    sources = rng.sample(posts, min(n_dups, len(posts)))
    for j, src in enumerate(sources):
        dup = dict(src)
        dup["post_id"] = f"mock_dup_{j:04d}"
        # Slightly mutate the text (swap one word so Jaccard stays > 0.8)
        words = dup["text"].split()
        if len(words) > 4:
            idx = rng.randint(1, len(words) - 2)
            words[idx] = rng.choice(["really", "actually", "honestly", "definitely"])
        dup["text"] = " ".join(words)
        dup["author"] = "user_" + "".join(rng.choices(string.ascii_lowercase, k=6))
        # Post within 5 minutes of original (burstiness)
        dup["created_utc"] = src["created_utc"] + rng.uniform(30, 300)
        posts.append(dup)

    rng.shuffle(posts)
    return _save(posts, date)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _even_sample(posts: list, n: int) -> list:
    """Sample n posts evenly across the time window."""
    posts_sorted = sorted(posts, key=lambda p: p["created_utc"])
    step = len(posts_sorted) / n
    return [posts_sorted[int(i * step)] for i in range(n)]


def _save(posts: list, date: datetime.date) -> str:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    path = os.path.join(RAW_DATA_DIR, f"{date}.json")
    with open(path, "w") as f:
        json.dump(posts, f, indent=2)
    print(f"[collector] Saved {len(posts)} posts → {path}")
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Trading day YYYY-MM-DD")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock data instead of live API")
    args = parser.parse_args()

    date = datetime.strptime(args.date, "%Y-%m-%d").date()

    if args.mock:
        path = generate_mock_data(date)
    else:
        # Credentials should be set as environment variables, never hardcoded
        path = collect(
            date,
            client_id=os.environ["REDDIT_CLIENT_ID"],
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=os.environ["REDDIT_USER_AGENT"],
        )
    print(f"[collector] Done: {path}")
