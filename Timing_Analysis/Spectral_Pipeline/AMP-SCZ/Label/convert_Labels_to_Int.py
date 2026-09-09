import numpy as np
from pathlib import Path

# ============================================================
# Paths
# ============================================================
label_path = Path("label.npy")
backup_path = Path("label_original_string.npy")
output_path = Path("label.npy")   # overwrite label.npy after backup

# ============================================================
# Load original label.npy
# Expected format:
# column 0 = label string, e.g. CHR / HC
# column 1 = subject ID
# ============================================================
labels = np.load(label_path, allow_pickle=True)

print("Original shape:", labels.shape)
print("First 5 rows before conversion:")
print(labels[:5])

# Backup original file
np.save(backup_path, labels)
print(f"\nBackup saved to: {backup_path}")

# ============================================================
# Convert CHR -> 1, HC -> 0
# ============================================================
label_map = {
    "HC": 0,
    "CHR": 1
}

new_labels = labels.copy()

for i in range(len(new_labels)):
    old_label = str(new_labels[i, 0])

    if old_label not in label_map:
        raise ValueError(f"Unknown label found at row {i}: {old_label}")

    new_labels[i, 0] = label_map[old_label]

# Optional: make sure labels are still object array because subject IDs are strings
new_labels = new_labels.astype(object)

# Save converted label.npy
np.save(output_path, new_labels)

print("\nFirst 5 rows after conversion:")
print(new_labels[:5])

print("\nUnique labels after conversion:")
print(np.unique(new_labels[:, 0]))

print(f"\nConverted label.npy saved to: {output_path}")