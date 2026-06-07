import os
import torch
import torch.nn as nn
import timm
from transformers import BertModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


class MultimodalFakeNewsClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )

        self.fusion = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        self.text_proj = nn.Linear(768, 256)
        self.image_proj = nn.Linear(768, 256)

        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_emb = bert_out.pooler_output
        image_emb = self.vit(pixel_values)

        fused = torch.cat([text_emb, image_emb], dim=1)
        logits = self.fusion(fused)

        return logits, text_emb, image_emb


def load_model(weights_path: str = None):
    global _model

    if _model is not None:
        return _model

    if weights_path is None:
        weights_path = os.path.join(
            os.path.dirname(__file__),
            "../models/fusion_model.pt"
        )

    if not os.path.exists(weights_path):
        print("Downloading model...")

        from huggingface_hub import hf_hub_download

        os.makedirs(os.path.dirname(weights_path), exist_ok=True)

        weights_path = hf_hub_download(
            repo_id="SonalPriyam/fake-news-detector-model",
            filename="fusion_model.pt",
            repo_type="model",
            local_dir=os.path.dirname(weights_path)
        )

        print("Downloaded!")

    print(f"Loading model from {weights_path}...")

    _model = MultimodalFakeNewsClassifier(num_classes=2)

    _model.load_state_dict(
        torch.load(weights_path, map_location=DEVICE),
        strict=False
    )

    _model.to(DEVICE)
    _model.eval()

    print(f"Model loaded on {DEVICE}")

    return _model