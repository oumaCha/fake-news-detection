import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


train_df = pd.read_csv("data/train.csv")
validation_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")


train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)
test_dataset = Dataset.from_pandas(test_df)


def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_validation = validation_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)


print("Train dataset:")
print(tokenized_train)

print("\nValidation dataset:")
print(tokenized_validation)

print("\nTest dataset:")
print(tokenized_test)

print("\nExample tokenized row:")
print(tokenized_train[0])