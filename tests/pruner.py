import pandas as pd

FILE_NAME = "datasets/gesture_dataset.csv"
OUTPUT_NAME = "datasets/gesture_dataset_pruned.csv"

# Load the dataset
df = pd.read_csv(FILE_NAME)
print(f"Original Dataset Size: {len(df)} samples")

# 1. Completely remove 'Ruler_Gesture'
df = df[df['label'] != 'Ruler_Gesture']

# 2. Drop 50% of the 'Thumb_Up' samples randomly
# Get indices of all Thumb_Up rows
thumb_up_indices = df[df['label'] == 'Thumb_Up'].index
# Randomly select half of them to drop
indices_to_drop = thumb_up_indices.to_series().sample(frac=0.5, random_state=42)
df = df.drop(indices_to_drop)

# Save the pruned dataset
df.to_csv(OUTPUT_NAME, index=False)

print(f"\nPruned Dataset Size: {len(df)} samples")
print("\nRemaining gestures:")
print(df['label'].value_counts())