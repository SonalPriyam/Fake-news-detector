import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchvision import transforms
import requests
from io import BytesIO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class FakedditDataset(Dataset):
    def __init__(self, csv_path=None, img_dir=None, max_len=128):
        if csv_path is None:
            csv_path = os.path.join(BASE_DIR, "data", "multimodal_train.tsv")

        if img_dir is None:
            img_dir = os.path.join(BASE_DIR, "data", "images")

        self.df = pd.read_csv(csv_path, sep="\t")
        self.img_dir = img_dir
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def load_image(self, img_id):
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
        else:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        return self.image_transform(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row["clean_title"])

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        image = self.load_image(row["id"])

        label = int(row["2_way_label"])

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "pixel_values": image,
            "label": torch.tensor(label, dtype=torch.long)
        }