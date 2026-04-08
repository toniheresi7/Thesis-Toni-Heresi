# sentiment_stocktwits.py
# Scores posts using the StockTwits-finetuned RoBERTa model.

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

MODEL_NAME = "zhayunduo/roberta-base-stocktwits-finetuned"

_model = None
_tokenizer = None
_device = None
_pos_idx = None
_neg_idx = None


def _load_model():
    global _model, _tokenizer, _device, _pos_idx, _neg_idx
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()

        # Model labels: {0: 'Negative', 1: 'Positive'}
        id2label = _model.config.id2label
        for idx, label in id2label.items():
            ll = label.lower()
            if "positive" in ll or "bullish" in ll:
                _pos_idx = int(idx)
            if "negative" in ll or "bearish" in ll:
                _neg_idx = int(idx)

        if _pos_idx is None or _neg_idx is None:
            raise ValueError(f"Could not find positive/negative in model labels: {id2label}")

    return _model, _tokenizer, _device


def score_post_stocktwits(text: str) -> dict:
    """Scores a post with StockTwits RoBERTa; returns probabilities and sp."""
    model, tokenizer, device = _load_model()

    inputs = tokenizer(
        text[:512],
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = softmax(logits, dim=-1)[0]

    pos = probs[_pos_idx].item()
    neg = probs[_neg_idx].item()
    # Binary model — no neutral class, set to 0
    neu = 0.0
    sp = pos - neg

    return {
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "sp": sp,
    }
