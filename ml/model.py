import torch
import torch.nn as nn
from transformers import BertModel
import timm


class MultimodalFakeNewsClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        for name, param in self.bert.named_parameters():
            if any(f"encoder.layer.{i}" in name for i in [8, 9, 10, 11]) or "pooler" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )

        for name, param in self.vit.named_parameters():
            if "blocks.10" in name or "blocks.11" in name or "norm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

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

        text_proj = self.text_proj(text_emb).unsqueeze(1)
        image_proj = self.image_proj(image_emb).unsqueeze(1)

        attended, _ = self.attention(
            text_proj,
            image_proj,
            image_proj
        )

        attended = attended.squeeze(1)

        fused = torch.cat([text_emb, image_emb], dim=1)

        logits = self.fusion(fused)

        return logits, text_emb, image_emb

    def get_text_embedding(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        return out.pooler_output

    def get_image_embedding(self, pixel_values):
        with torch.no_grad():
            return self.vit(pixel_values)


if __name__ == "__main__":
    model = MultimodalFakeNewsClassifier()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")