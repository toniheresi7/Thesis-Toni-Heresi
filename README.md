# Social Signal Screening Platform
### Antoni Heresi — IE University Capstone

A daily Top-10 stock monitoring shortlist derived from Reddit social signals,
ranked by a LightGBM model using sentiment, attention, breadth, and price momentum features.

---

## Project Structure

```
social_signal_platform/
│
├── src/
│   ├── config.py                     # All parameters (K, windows, credibility weights, paths)
│   ├── csv_loader.py                 # Loads and indexes the Reddit CSV dataset
│   ├── collector.py                  # Live Reddit collection via PRAW (or mock data)
│   ├── text_utils.py                 # Text normalisation, ticker mapping, deduplication
│   ├── features.py                   # Feature computation (tone, attention, breadth, credibility)
│   ├── market_data.py                # yfinance: prices, returns, momentum
│   ├── model.py                      # LightGBM: train, predict, rank, explain
│   │
│   ├── sentiment_stocktwits.py       # Experiment 2: StockTwits RoBERTa scorer
│   ├── sentiment_finetuned.py        # Experiment 3: Fine-tuned FinBERT scorer
│   ├── features_upvote_weighted.py   # Experiment 4: Upvote-weighted tone aggregation
│   │
│   ├── generate_pseudo_labels.py     # Generates pseudo-labels for FinBERT fine-tuning
│   ├── finetune_finbert.py           # Fine-tunes FinBERT on pseudo-labels
│   │
│   ├── run_pipeline_stocktwits.py    # Batch pipeline for Experiment 2
│   ├── run_pipeline_finetuned.py     # Batch pipeline for Experiment 3
│   └── run_pipeline_upvote_tone.py   # Batch pipeline for Experiment 4
│
├── evaluation/
│   └── evaluate_retrained.py         # Retrains LightGBM per experiment and evaluates
│
├── backend/
│   └── main.py                       # FastAPI backend serving the dashboard
│
├── frontend/                         # React/Vite dashboard
│
├── jobs/                             # Slurm job scripts (HPC cluster)
├── results/                          # Final evaluation outputs (txt)
├── data/                             # Model artifacts, lexicons, ticker universe
├── run_pipeline.py                   # Single-day pipeline (CSV or live Reddit)
├── run_batch.py                      # Full dataset batch processing + model training
├── run_batch_cluster.py              # Cluster variant of run_batch.py (CUDA)
└── requirements.txt
```

---

## Execution Order

### Step 0 — Install dependencies
```bash
pip install -r requirements.txt
```

---

### Step 1 — Process the dataset and train the baseline model

The pipeline reads from `data/reddit_sp500.csv` (emilpartow/reddit-finance-posts-sp500).

```bash
python run_batch.py
```

This processes all 1,006 trading days (2020–2023), computes features, and trains the
LightGBM model. Outputs go to `data/processed/`, `data/model.pkl`, `data/scaler.pkl`,
and `data/calibrated_params.json`.

To skip days already processed:
```bash
python run_batch.py --skip-existing
```

---

### Step 2 — Evaluate the baseline model

```bash
PYTHONPATH=src python evaluation/evaluate_retrained.py \
    --data-dir data/processed \
    --output results/baseline_retrained_evaluation.txt
```

---

### Step 3 — Run experiment pipelines (sentiment variants)

**Experiment 2 — StockTwits RoBERTa:**
```bash
PYTHONPATH=src python src/run_pipeline_stocktwits.py
```

**Experiment 3 — Fine-tuned FinBERT:**
```bash
# First generate pseudo-labels, then fine-tune, then run the pipeline
PYTHONPATH=src python src/generate_pseudo_labels.py
PYTHONPATH=src python src/finetune_finbert.py
PYTHONPATH=src python src/run_pipeline_finetuned.py
```

**Experiment 4 — Upvote-weighted tone:**
```bash
PYTHONPATH=src python src/run_pipeline_upvote_tone.py
```

---

### Step 4 — Evaluate each experiment

```bash
PYTHONPATH=src python evaluation/evaluate_retrained.py \
    --data-dir data/processed_stocktwits \
    --output results/stocktwits_retrained_evaluation.txt

PYTHONPATH=src python evaluation/evaluate_retrained.py \
    --data-dir data/processed_finetuned \
    --output results/finetuned_retrained_evaluation.txt

PYTHONPATH=src python evaluation/evaluate_retrained.py \
    --data-dir data/processed_upvote_tone \
    --output results/upvote_tone_retrained_evaluation.txt
```

Pre-computed results are already in `results/`.

---

### Step 5 — Run the dashboard (optional)

Backend:
```bash
PYTHONPATH=src python -m uvicorn backend.main:app --port 8000
```

Frontend (requires Node.js):
```bash
cd frontend && npm install && npm run dev
```

---

## Pipeline Flow

```
csv_loader → text_utils → features → model → evaluate_retrained
                                   ↑
                            market_data (prices, momentum)
```

---

## Key Parameters (src/config.py)

| Parameter       | Value | Description                              |
|----------------|-------|------------------------------------------|
| K              | 10    | Shortlist size                           |
| TRIM_RATE      | 0.10  | Trimmed mean: discard top/bottom 10%     |
| ATTN_WINDOW    | 10    | Rolling z-score window (trading days)    |
| TRAIN_SPLIT    | 0.65  | 65% training / 35% evaluation            |
| MIN_POSTS      | 3     | Minimum posts for a ticker to qualify    |
| CRED_ALPHA/BETA/GAMMA | 0.33/0.33/0.34 | Credibility penalty weights |

---

## Notes

- The dataset (`data/reddit_sp500.csv`) is not included due to size (2.3 GB).
  Download from: [emilpartow/reddit-finance-posts-sp500](https://huggingface.co/datasets/emilpartow/reddit-finance-posts-sp500)
- Experiments 2 and 3 require a GPU for practical runtime. Job scripts for an HPC
  cluster (Slurm) are in `jobs/`.
- Pre-trained model artifacts (`model.pkl`, `scaler.pkl`) are included so Steps 2–5
  can be run without retraining.
