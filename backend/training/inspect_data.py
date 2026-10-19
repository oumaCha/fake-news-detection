import pandas as pd

fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")


print("Fake dataset shape:")
print(fake_df.shape)

print("\nTrue dataset shape:")
print(true_df.shape)

fake_df["label"] = 1
true_df["label"] = 0


combined_df = pd.concat([fake_df, true_df], ignore_index=True)

combined_df = combined_df[["title", "text", "label"]]

combined_df = combined_df.dropna()
combined_df = combined_df.drop_duplicates()

combined_df = combined_df.sample(frac= 1, random_state= 42).reset_index(drop= True)

combined_df.to_csv("data/fake_news_clean.csv", index=False)

print("\nFinal dataset shape:")
print(combined_df.shape)
print("\nFirsst 5 ROWS:")
print(combined_df.head())

print("\nLabel distribution:")
print(combined_df["label"].value_counts())

