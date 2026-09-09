import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to your label.npy file
label_path = r"label.npy"

# Load labels
labels = np.load(label_path, allow_pickle=True)

print("Shape:", labels.shape)
print("Data type:", labels.dtype)
print("\nFirst 20 labels:")
print(labels[:20])

# Flatten in case labels are stored as column vector
labels_flat = labels.ravel()

# Count unique values
unique_labels, counts = np.unique(labels_flat, return_counts=True)

print("\nUnique labels and counts:")
for label, count in zip(unique_labels, counts):
    print(f"{label}: {count}")

# Convert to DataFrame for cleaner display
df_counts = pd.DataFrame({
    "Label": unique_labels,
    "Count": counts
})

df_counts["Percentage"] = 100 * df_counts["Count"] / df_counts["Count"].sum()

print("\nLabel distribution:")
print(df_counts)

# Plot distribution
plt.figure(figsize=(6, 4))
plt.bar(df_counts["Label"].astype(str), df_counts["Count"])
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Distribution of Labels in label.npy")
plt.tight_layout()
plt.show()