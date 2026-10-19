import torch 
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from transformers import (AutoTokenizer, AutoModelForSequenceClassification)

MODEL_PATH = "model/fake_news_model"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ArticleRequest(BaseModel):
    text:str

@app.get("/")
def home():
    return{"message":"Fake News Detection API is running"}


@app.post("/api/predict")
def predict(article:ArticleRequest):
    return{
        "prediction": "fake",
        "confidence": 0.85,
        "text_received": article.text
    }