# generate_pseudo_labels.py
# Generates pseudo-labels from Reddit posts using StockTwits RoBERTa.

import os
import json
import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/pseudo_labelled"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pseudo_labels.csv")
MODEL_NAME = "zhayunduo/roberta-base-stocktwits-finetuned"
CONFIDENCE_THRESHOLD = 0.85
BATCH_SIZE = 64


def load_all_posts():
    """Load all posts from all JSON files in data/raw/."""
    posts = []
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(RAW_DIR, fname)
        with open(path) as f:
            day_posts = json.load(f)
        for p in day_posts:
            text = p.get("text", "").strip()
            if len(text) > 10:  # skip very short posts
                posts.append({
                    "post_id": p.get("post_id", ""),
                    "text": text,
                })
    return posts


def generate_pseudo_labels():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pseudo_labels] Using device: {device}")

    print(f"[pseudo_labels] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    # Get label mapping from model config
    id2label = model.config.id2label
    print(f"[pseudo_labels] Model labels: {id2label}")

    # Find indices for positive and negative labels
    pos_idx = None
    neg_idx = None
    for idx, label in id2label.items():
        ll = label.lower()
        if "positive" in ll or "bullish" in ll:
            pos_idx = int(idx)
        if "negative" in ll or "bearish" in ll:
            neg_idx = int(idx)

    if pos_idx is None or neg_idx is None:
        raise ValueError(f"Could not find positive/negative in model labels: {id2label}")
    print(f"[pseudo_labels] Positive idx={pos_idx}, Negative idx={neg_idx}")

    posts = load_all_posts()
    print(f"[pseudo_labels] Total posts loaded: {len(posts)}")

    results = []
    for i in range(0, len(posts), BATCH_SIZE):
        batch = posts[i:i + BATCH_SIZE]
        texts = [p["text"][:512] for p in batch]

        inputs = tokenizer(texts, padding=True, truncation=True,
                           max_length=128, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = softmax(logits, dim=-1)

        for j, p in enumerate(batch):
            max_prob, pred_idx = probs[j].max(dim=-1)
            max_prob = max_prob.item()
            pred_idx = pred_idx.item()

            if max_prob < CONFIDENCE_THRESHOLD:
                continue

            if pred_idx == pos_idx:
                label = "positive"
            elif pred_idx == neg_idx:
                label = "negative"
            else:
                continue  # skip neutral/other predictions

            results.append({
                "post_id": p["post_id"],
                "text": p["text"],
                "label": label,
                "confidence": round(max_prob, 4),
            })

        if (i // BATCH_SIZE) % 50 == 0:
            print(f"[pseudo_labels] Processed {min(i + BATCH_SIZE, len(posts))}/{len(posts)} posts, "
                  f"{len(results)} labelled so far")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["post_id", "text", "label", "confidence"])
        writer.writeheader()
        writer.writerows(results)

    # Stats
    pos_count = sum(1 for r in results if r["label"] == "positive")
    neg_count = sum(1 for r in results if r["label"] == "negative")
    print(f"\n[pseudo_labels] Done!")
    print(f"  Total posts processed: {len(posts)}")
    print(f"  Posts labelled (conf > {CONFIDENCE_THRESHOLD}): {len(results)}")
    print(f"  Positive: {pos_count}")
    print(f"  Negative: {neg_count}")
    print(f"  Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_pseudo_labels()
