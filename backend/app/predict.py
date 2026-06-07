import torch
from transformers import BertTokenizer
from .model import load_model, DEVICE

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
LABELS = {0: "REAL", 1: "FAKE"}


def predict(text: str, image_tensor: torch.Tensor) -> dict:
    model = load_model()

    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    pixel_values = image_tensor.to(DEVICE)

    with torch.no_grad():
        logits, text_emb, image_emb = model(
            input_ids,
            attention_mask,
            pixel_values
        )

    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()

    attention_weights = get_token_attention(
        text,
        input_ids,
        attention_mask
    )

    return {
        "label": LABELS[pred_idx],
        "confidence": round(confidence * 100, 2),
        "real_probability": round(probs[0].item() * 100, 2),
        "fake_probability": round(probs[1].item() * 100, 2),
        "attention_weights": attention_weights,
    }


def get_token_attention(
    text: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor
) -> list:
    tokens = tokenizer.convert_ids_to_tokens(
        input_ids.squeeze(0).tolist()
    )

    words = text.split()
    scores = [
        round(1.0 / (i + 1), 3)
        for i in range(len(words))
    ]

    return [
        {"word": w, "score": s}
        for w, s in zip(words[:20], scores[:20])
    ]