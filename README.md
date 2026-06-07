# 🔍 Fake News Detector

### 🚀 **TRY THE LIVE APPLICATION**

**Live Demo:**
https://sonalpriyam-fake-news-detector.hf.space/



Paste any news article URL and receive:

* REAL / FAKE prediction
* Confidence score
* Class probability breakdown
* Text attention visualization
* Grad-CAM image explanation

---

## Overview

Fake news articles often use misleading headlines along with unrelated or manipulated images to influence readers. Traditional fake news detection systems primarily focus on textual information and ignore visual context.

This project presents a **multimodal fake news detection system** that jointly analyzes article text and associated images. The model combines a fine-tuned **BERT** text encoder and a **Vision Transformer (ViT)** image encoder to classify news articles as **REAL** or **FAKE**.

The system automatically scrapes article content from a URL, processes both modalities, performs inference, and provides explainable predictions through attention visualization and Grad-CAM heatmaps.

---

## Key Features

* Multimodal fake news detection using text and image information
* BERT-based textual feature extraction
* Vision Transformer (ViT) image feature extraction
* Cross-modal feature fusion architecture
* Automatic news article scraping from URLs
* Explainable AI with Grad-CAM visualization
* Word-level attention highlighting
* REST API built using FastAPI
* Interactive React frontend
* Dockerized deployment on Hugging Face Spaces

---

## Model Architecture

### Text Encoder

* BERT Base Uncased
* Fine-tuned on fake news classification data

### Image Encoder

* Vision Transformer (ViT Base Patch16 224)
* Extracts visual representations from article images

### Fusion Layer

* Concatenates text and image embeddings
* Fully connected neural network for classification

### Prediction Pipeline

```text
News Article URL
        │
        ▼
Article Scraper
        │
 ┌──────┴──────┐
 ▼             ▼
BERT          ViT
(Text)      (Image)
 └──────┬──────┘
        ▼
 Feature Fusion
        ▼
 Classification
        ▼
REAL / FAKE
        ▼
Explainability
(Grad-CAM + Attention)
```

---

## Dataset

**Dataset:** FAKEDDIT

FAKEDDIT is one of the largest publicly available multimodal fake news datasets containing Reddit posts with associated images.

Dataset Statistics:

* Total dataset size: 564,000+ samples
* Training subset used: 10,000 samples
* Binary classification setup
* Labels:

  * REAL
  * FAKE

---

## Results



The multimodal architecture improved performance over the text-only baseline by incorporating visual information alongside textual context.

---

## Explainability

The model not only predicts but also explains its decisions.

### Grad-CAM

Generates visual heatmaps showing which image regions contributed most to the prediction.

### Text Attention

Highlights important words that influenced the model's decision, helping users understand the reasoning behind the classification.

---

## Technology Stack

### Machine Learning

* PyTorch
* Hugging Face Transformers
* timm
* Grad-CAM

### Backend

* FastAPI
* Uvicorn
* Pydantic
* newspaper3k

### Frontend

* React
* Vite

### Deployment

* Docker
* Hugging Face Spaces

### Training Environment

* Google Colab
* NVIDIA T4 GPU

---

## Project Structure

```text
fake-news-detector/

├── ml/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── bert_baseline.py
│
├── backend/
│   └── app/
│       ├── main.py
│       ├── model.py
│       ├── predict.py
│       ├── scraper.py
│       └── explainability.py
│
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── UrlInput.jsx
│       ├── ResultCard.jsx
│       ├── TextHighlight.jsx
│       └── GradCamOverlay.jsx
│
├── models/
│
├── requirements.txt
│
└── Dockerfile
```

---

## Local Setup

### Clone Repository

```bash
git clone https://github.com/<SonalPriyam>/fake-news-detector.git

cd fake-news-detector
```

---

### Backend Setup

```bash
cd backend

pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

Backend Server:

```text
http://localhost:8000
```

API Documentation:

```text
http://localhost:8000/docs
```

---

### Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

Frontend Application:

```text
http://localhost:5173
```


 





