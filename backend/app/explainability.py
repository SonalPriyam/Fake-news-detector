import torch
import numpy as np
from transformers import BertTokenizer
from PIL import Image
import base64
from io import BytesIO

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except Exception:
    GRADCAM_AVAILABLE = False

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def get_gradcam_heatmap(model, image_tensor: torch.Tensor) -> str:
    if not GRADCAM_AVAILABLE:
        return ""
    try:
        target_layer = model.vit.blocks[-1].norm1
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=image_tensor, targets=None)
        grayscale_cam = grayscale_cam[0]

        img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1).astype(np.float32)

        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        pil_img = Image.fromarray(cam_image)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return ""


def get_text_attention(
    text: str,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor
) -> list:
    try:
        with torch.no_grad():
            outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )

        last_layer_attn = outputs.attentions[-1]
        cls_attention = last_layer_attn[0, :, 0, :]
        avg_attention = cls_attention.mean(dim=0)

        tokens = tokenizer.convert_ids_to_tokens(
            input_ids.squeeze(0).tolist()
        )

        word_scores = []
        current_word = ""
        current_score = 0.0

        for token, score in zip(tokens, avg_attention.tolist()):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            if token.startswith("##"):
                current_word += token[2:]
                current_score = max(current_score, score)
            else:
                if current_word:
                    word_scores.append({
                        "word": current_word,
                        "score": round(current_score, 4)
                    })

                current_word = token
                current_score = score

        if current_word:
            word_scores.append({
                "word": current_word,
                "score": round(current_score, 4)
            })

        if word_scores:
            max_score = max(w["score"] for w in word_scores)
            min_score = min(w["score"] for w in word_scores)
            score_range = max_score - min_score or 1.0

            for w in word_scores:
                w["score"] = round(
                    (w["score"] - min_score) / score_range,
                    4
                )

        return word_scores[:30]

    except Exception as e:
        print(f"Attention extraction error: {e}")
        return []