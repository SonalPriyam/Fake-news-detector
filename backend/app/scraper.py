import requests
from newspaper import Article
from PIL import Image
from io import BytesIO
from torchvision import transforms
import torch

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def scrape_article(url: str) -> dict:
    try:
        article = Article(url)
        article.download()
        article.parse()

        title = article.title or ""
        text = article.text or ""
        full_text = f"{title}. {text[:512]}"

        image_tensor = fetch_image(article.top_image)

        return {
            "title": title,
            "text": full_text,
            "image_url": article.top_image,
            "image_tensor": image_tensor,
            "success": True
        }

    except Exception as e:
        return {
            "title": "",
            "text": "",
            "image_url": "",
            "image_tensor": blank_image_tensor(),
            "success": False,
            "error": str(e)
        }


def fetch_image(image_url: str) -> torch.Tensor:
    try:
        if not image_url:
            return blank_image_tensor()

        response = requests.get(image_url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        return IMAGE_TRANSFORM(img).unsqueeze(0)

    except Exception:
        return blank_image_tensor()


def blank_image_tensor() -> torch.Tensor:
    return torch.zeros(1, 3, 224, 224)