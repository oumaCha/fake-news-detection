from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels = 2
)

print("Model loaded successfully!")

text = "Scientists announced a new discovery today"
tokens = tokenizer(
    text,
    truncation = True,
    padding = True,
    return_tensors = "pt"
)

print("\nOriginal text:")
print(text)

print("\nTokens:")
print(tokens)