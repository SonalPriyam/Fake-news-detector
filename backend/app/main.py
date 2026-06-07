from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import os
from transformers import BertTokenizer

from .model import load_model, DEVICE
from .scraper import scrape_article
from .predict import predict
from .explainability import get_gradcam_heatmap, get_text_attention

app = FastAPI(
    title="Fake News Detector API",
    description="Multimodal BERT + ViT fake news detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class URLRequest(BaseModel):
    url: str


class TextRequest(BaseModel):
    text: str
    image_url: str = ""


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    real_probability: float
    fake_probability: float
    title: str
    text_preview: str
    image_url: str
    attention_weights: list
    gradcam_image: str


@app.on_event("startup")
async def startup_event():
    try:
        print("Loading model...")
        load_model()
        print("Model ready!")
    except Exception as e:
        print(f"Model loading error: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict", response_model=PredictionResponse)
async def predict_url(request: URLRequest):
    article = scrape_article(request.url)

    if not article["success"]:
        raise HTTPException(
            status_code=400,
            detail=f"Could not scrape URL: {article.get('error', 'Unknown error')}"
        )

    text = article["text"]
    image_tensor = article["image_tensor"].to(DEVICE)

    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    model = load_model()

    result = predict(text, image_tensor)

    gradcam_image = get_gradcam_heatmap(model, image_tensor)
    attention_weights = get_text_attention(
        text,
        model,
        input_ids,
        attention_mask
    )

    return PredictionResponse(
        label=result["label"],
        confidence=result["confidence"],
        real_probability=result["real_probability"],
        fake_probability=result["fake_probability"],
        title=article["title"],
        text_preview=text[:300],
        image_url=article["image_url"],
        attention_weights=attention_weights,
        gradcam_image=gradcam_image
    )


@app.post("/predict-text")
async def predict_text(request: TextRequest):
    blank_image = torch.zeros(1, 3, 224, 224).to(DEVICE)
    result = predict(request.text, blank_image)
    return result


frontend_path = os.path.join(
    os.path.dirname(__file__),
    "../../frontend/dist"
)

if os.path.exists(frontend_path):
    app.mount(
        "/assets",
        StaticFiles(directory=f"{frontend_path}/assets"),
        name="assets"
    )

    @app.get("/", include_in_schema=False)
    def serve_react():
        return FileResponse(f"{frontend_path}/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,
        reload=True
    )