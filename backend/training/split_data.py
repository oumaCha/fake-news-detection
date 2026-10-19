import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/fake_news_clean.csv")

train_df, temp_df = train_test_split (
    df,
    test_size = 0.20,
    random_state= 42,
    stratify=df["label"]
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

train_df.to_csv("data/train.csv", index=False)
validation_df.to_csv("data/validation.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Training set:")
print(train_df.shape)

print("\nValidation set:")
print(validation_df.shape)

print("\nTest set:")
print(test_df.shape)

print("\nTraining labels:")
print(train_df["label"].value_counts())

print("\nValidation labels:")
print(validation_df["label"].value_counts())

print("\nTest labels:")
print(test_df["label"].value_counts())