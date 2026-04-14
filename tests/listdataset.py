import pandas as pd

FILE_NAME = "datasets/gesture_dataset.csv"

df = pd.read_csv(FILE_NAME)

labels = df['label'].unique().tolist()

print(labels)

for label in labels:

    filter = df[df["label"] == label]

    print(f"There are {len(filter)} samples with label {label}")