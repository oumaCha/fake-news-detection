import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


MODEL_PATH = "model/fake_news_model"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)


text = """
Scientists announced a new discovery after publishing
the results of a multi-year research study.
"""


inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=256
)


with torch.no_grad():
    outputs = model(**inputs)


logits = outputs.logits


probabilities = torch.softmax(logits, dim=1)


predicted_class = torch.argmax(probabilities, dim=1).item()


confidence = probabilities[0][predicted_class].item()


label = model.config.id2label[predicted_class]


print("Article:")
print(text)

print("\nPrediction:")
print(label)

print("\nConfidence:")
print(f"{confidence * 100:.2f}%")