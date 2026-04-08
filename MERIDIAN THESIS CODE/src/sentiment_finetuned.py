# sentiment_finetuned.py
# Scores posts using the fine-tuned FinBERT model.

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

MODEL_DIR = "models/finbert_finetuned"

_model = None
_tokenizer = None
_device = None


def _load_model():
    global _model, _tokenizer, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.to(_device)
        _model.eval()
    return _model, _tokenizer, _device


def score_post_finetuned(text: str) -> dict:
    """Scores a post with fine-tuned FinBERT; returns probabilities and sp."""
    model, tokenizer, device = _load_model()

    inputs = tokenizer(
        text[:512],
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = softmax(logits, dim=-1)[0]

    # FinBERT labels: 0=positive, 1=negative, 2=neutral
    pos = probs[0].item()
    neg = probs[1].item()
    neu = probs[2].item()
    sp = pos - neg

    return {
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "sp": sp,
    }
