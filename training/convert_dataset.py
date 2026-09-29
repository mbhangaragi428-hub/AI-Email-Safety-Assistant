import pandas as pd

# Read the original SMS Spam Collection dataset
data = pd.read_csv(
    "dataset/spamdata/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "text"]
)

# Keep only the required columns
data = data[["text", "label"]]

# Save in our project's CSV format
data.to_csv(
    "dataset/emails.csv",
    index=False
)

print("Dataset converted successfully!")
print("Total messages:", len(data))

print("\nDataset preview:")
print(data.head())

print("\nLabel counts:")
print(data["label"].value_counts())