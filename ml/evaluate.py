import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

def full_evaluation(model, dataset, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch["pixel_values"].to(device)
            labels         = batch["label"].to(device)

            if hasattr(model, "vit"):
                output = model(input_ids, attention_mask, pixel_values)
                logits = output[0] if isinstance(output, tuple) else output
            else:
                logits = model(input_ids, attention_mask)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return all_preds, all_labels, all_probs