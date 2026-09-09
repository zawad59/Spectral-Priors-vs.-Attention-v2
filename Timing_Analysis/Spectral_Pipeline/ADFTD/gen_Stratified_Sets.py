import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# ============================================================
# Configuration
# ============================================================

INPUT_CSV = "adftd_fft_dwt_from_derivatives_09-08.csv"

OUTPUT_DIR = "stratified_holdout_splits"

SUBJECT_COL = "subj_ID"
LABEL_COL = "Group"

N_SPLITS = 5
TEST_SIZE = 0.20      # 20% holdout test
VAL_SIZE_IN_TRAIN = 0.15 # 15% of training set used as validation

RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("Original data shape:", df.shape)
print("\nClass distribution:")
print(df[LABEL_COL].value_counts().sort_index())

# ============================================================
# Build subject-level table
# ============================================================

subject_df = (
    df[[SUBJECT_COL, LABEL_COL]]
    .drop_duplicates()
    .reset_index(drop=True)
)

# Safety check: each subject should have only one label
label_counts_per_subject = df.groupby(SUBJECT_COL)[LABEL_COL].nunique()

if label_counts_per_subject.max() > 1:
    raise ValueError(
        "At least one subject has multiple labels. "
        "Please check your data before subject-level splitting."
    )

print("\nNumber of unique subjects:", subject_df[SUBJECT_COL].nunique())
print("\nSubject-level class distribution:")
print(subject_df[LABEL_COL].value_counts().sort_index())

X_subjects = subject_df[SUBJECT_COL].values
y_subjects = subject_df[LABEL_COL].values

# ============================================================
# Outer stratified holdout splits
# ============================================================

outer_splitter = StratifiedShuffleSplit(
    n_splits=N_SPLITS,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED
)

all_split_records = []

for split_idx, (trainval_idx, test_idx) in enumerate(
    outer_splitter.split(X_subjects, y_subjects),
    start=1
):
    print("\n" + "=" * 60)
    print(f"Creating split {split_idx}")
    print("=" * 60)

    split_dir = os.path.join(OUTPUT_DIR, f"split_{split_idx}")
    os.makedirs(split_dir, exist_ok=True)

    trainval_subject_df = subject_df.iloc[trainval_idx].reset_index(drop=True)
    test_subject_df = subject_df.iloc[test_idx].reset_index(drop=True)

    # ------------------------------------------------------------
    # Inner validation split from the training subjects only
    # ------------------------------------------------------------

    inner_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_SIZE_IN_TRAIN,
        random_state=RANDOM_SEED + split_idx
    )

    train_idx_inner, val_idx_inner = next(
        inner_splitter.split(
            trainval_subject_df[SUBJECT_COL].values,
            trainval_subject_df[LABEL_COL].values
        )
    )

    train_subject_df = trainval_subject_df.iloc[train_idx_inner].reset_index(drop=True)
    val_subject_df = trainval_subject_df.iloc[val_idx_inner].reset_index(drop=True)

    # ------------------------------------------------------------
    # Get subject IDs
    # ------------------------------------------------------------

    train_subjects = set(train_subject_df[SUBJECT_COL])
    val_subjects = set(val_subject_df[SUBJECT_COL])
    test_subjects = set(test_subject_df[SUBJECT_COL])

    # ------------------------------------------------------------
    # Safety leakage checks
    # ------------------------------------------------------------

    assert train_subjects.isdisjoint(val_subjects), "Train/val subject leakage!"
    assert train_subjects.isdisjoint(test_subjects), "Train/test subject leakage!"
    assert val_subjects.isdisjoint(test_subjects), "Val/test subject leakage!"

    # ------------------------------------------------------------
    # Create row-level CSVs from subject IDs
    # ------------------------------------------------------------

    train_df = df[df[SUBJECT_COL].isin(train_subjects)].reset_index(drop=True)
    val_df = df[df[SUBJECT_COL].isin(val_subjects)].reset_index(drop=True)
    test_df = df[df[SUBJECT_COL].isin(test_subjects)].reset_index(drop=True)

    # ------------------------------------------------------------
    # Save split data
    # ------------------------------------------------------------

    train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)

    # ------------------------------------------------------------
    # Save subject IDs separately
    # ------------------------------------------------------------

    train_subject_df.to_csv(
        os.path.join(split_dir, "train_subject_ids.csv"),
        index=False
    )

    val_subject_df.to_csv(
        os.path.join(split_dir, "val_subject_ids.csv"),
        index=False
    )

    test_subject_df.to_csv(
        os.path.join(split_dir, "test_subject_ids.csv"),
        index=False
    )

    # ------------------------------------------------------------
    # Save combined split subject summary
    # ------------------------------------------------------------

    train_summary = train_subject_df.copy()
    train_summary["split"] = split_idx
    train_summary["set"] = "train"

    val_summary = val_subject_df.copy()
    val_summary["split"] = split_idx
    val_summary["set"] = "val"

    test_summary = test_subject_df.copy()
    test_summary["split"] = split_idx
    test_summary["set"] = "test"

    split_summary = pd.concat(
        [train_summary, val_summary, test_summary],
        axis=0
    ).reset_index(drop=True)

    split_summary.to_csv(
        os.path.join(split_dir, "subject_split_summary.csv"),
        index=False
    )

    all_split_records.append(split_summary)

    # ------------------------------------------------------------
    # Print diagnostics
    # ------------------------------------------------------------

    print("Train shape:", train_df.shape)
    print("Val shape:  ", val_df.shape)
    print("Test shape: ", test_df.shape)

    print("\nTrain class counts:")
    print(train_df[LABEL_COL].value_counts().sort_index())

    print("\nVal class counts:")
    print(val_df[LABEL_COL].value_counts().sort_index())

    print("\nTest class counts:")
    print(test_df[LABEL_COL].value_counts().sort_index())

# ============================================================
# Save all splits together
# ============================================================

all_splits_df = pd.concat(all_split_records, axis=0).reset_index(drop=True)

all_splits_df.to_csv(
    os.path.join(OUTPUT_DIR, "all_split_subjects_summary.csv"),
    index=False
)

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
print(f"All splits saved inside: {OUTPUT_DIR}")
print("Combined subject summary saved as:")
print(os.path.join(OUTPUT_DIR, "all_split_subjects_summary.csv"))