# finetune_finbert.py
# Fine-tunes ProsusAI/finbert on pseudo-labelled Reddit posts.

import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

PSEUDO_LABELS_PATH = "data/pseudo_labelled/pseudo_labels.csv"
OUTPUT_DIR = "models/finbert_finetuned"
BASE_MODEL = "ProsusAI/finbert"

# Hyperparameters
EPOCHS = 3
LR = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 128
TRAIN_RATIO = 0.8

# FinBERT label mapping: positive=0, negative=1, neutral=2
LABEL_MAP = {"positive": 0, "negative": 1}


class PseudoLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_pseudo_labels():
    texts, labels = [], []
    with open(PSEUDO_LABELS_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = LABEL_MAP.get(row["label"])
            if label is not None:
                texts.append(row["text"][:512])
                labels.append(label)
    return texts, labels


def finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[finetune] Using device: {device}")

    print(f"[finetune] Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)
    model.to(device)

    texts, labels = load_pseudo_labels()
    print(f"[finetune] Loaded {len(texts)} pseudo-labelled samples")

    dataset = PseudoLabelDataset(texts, labels, tokenizer, MAX_LENGTH)

    train_size = int(len(dataset) * TRAIN_RATIO)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print(f"[finetune] Train: {train_size}, Val: {val_size}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                 num_training_steps=total_steps)

    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"].to(device)).sum().item()
                total += len(batch["labels"])

        val_acc = correct / total if total > 0 else 0
        print(f"[finetune] Epoch {epoch + 1}/{EPOCHS}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[finetune] Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    finetune()
