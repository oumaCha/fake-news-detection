import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


MODEL_NAME = "distilbert-base-uncased"


# --------------------------------------------------
# 1. Load the CSV datasets
# --------------------------------------------------

train_df = pd.read_csv("data/train.csv")
validation_df = pd.read_csv("data/validation.csv")


# --------------------------------------------------
# 2. Convert Pandas DataFrames to Hugging Face datasets
# --------------------------------------------------

train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)


# --------------------------------------------------
# 3. Load DistilBERT tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# --------------------------------------------------
# 4. Define tokenization
# --------------------------------------------------

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


# --------------------------------------------------
# 5. Tokenize datasets
# --------------------------------------------------

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True
)

tokenized_validation = validation_dataset.map(
    tokenize_function,
    batched=True
)


# --------------------------------------------------
# 6. Load DistilBERT classification model
# --------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "REAL",
        1: "FAKE"
    },
    label2id={
        "REAL": 0,
        "FAKE": 1
    }
)


# --------------------------------------------------
# 7. Define evaluation metrics
# --------------------------------------------------

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# --------------------------------------------------
# 8. Configure the training
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir="model/checkpoints",

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=2,

    weight_decay=0.01,

    load_best_model_at_end=True,

    metric_for_best_model="f1",
    greater_is_better=True,

    save_total_limit=2,

    logging_steps=100,

    report_to="none"
)


# --------------------------------------------------
# 9. Create Trainer
# --------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,

    compute_metrics=compute_metrics
)


# --------------------------------------------------
# 10. Train the model
# --------------------------------------------------

print("Starting training...")

trainer.train()


# --------------------------------------------------
# 11. Save the final model
# --------------------------------------------------

trainer.save_model("model/fake_news_model")
tokenizer.save_pretrained("model/fake_news_model")


print("Training finished!")
print("Model saved to model/fake_news_model")