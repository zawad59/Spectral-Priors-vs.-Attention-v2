import os
import numpy as np
import mne
from pathlib import Path

# CONFIG 
ROOT_DIR   = r"Eyes_Closed_visit1_only"
OUTPUT_DIR = r"Feature"
TARGET_FS  = 256      # Hz -- resample target
WINDOW_SEC = 1        # seconds per window


mne.set_log_level("WARNING")   # suppress verbose MNE output

os.makedirs(OUTPUT_DIR, exist_ok=True)

root = Path(ROOT_DIR)
subject_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

print(f"Found {len(subject_dirs)} subjects.\n")

for subj_dir in subject_dirs:
    subj_id = subj_dir.name                          # e.g. "BI07844"
    out_path = Path(OUTPUT_DIR) / f"Feature_{subj_id}.npy"

    if out_path.exists():
        print(f"[SKIP]  {subj_id}  --> already exists")
        continue

    #  Find the .vhdr file (BrainVision header) 
    # Support both flat layout and visit_1/ sub-folder layout
    vhdr_files = list(subj_dir.rglob("*.vhdr"))

    if not vhdr_files:
        print(f"[WARN]  {subj_id}  – no .vhdr found, skipping")
        continue

    # If multiple visits exist, sort and take the first (visit_1)
    vhdr_files.sort()
    vhdr_path = vhdr_files[0]

    try:
        #  Load 
        raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True)

        #  Remove VIS channel if present
        if "VIS" in raw.ch_names:
            raw.drop_channels(["VIS"])
            print(f"[INFO]  {subj_id}  --> dropped VIS channel")

        #  Resample to TARGET_FS 
        if raw.info["sfreq"] != TARGET_FS:
            raw.resample(TARGET_FS)

        #  Extract data: shape  (n_channels, n_times) 
        data = raw.get_data()                         # (C, T)
        n_channels, n_times = data.shape

        #  Slice into non-overlapping 1-second windows 
        samples_per_window = TARGET_FS * WINDOW_SEC   # 256 samples
        n_windows = n_times // samples_per_window     # drop any remainder

        data_cropped = data[:, : n_windows * samples_per_window]   # (C, n_win*256)

        # reshape --> (n_windows, timestamps, channels)
        # (C, n_win*256) -> (C, n_win, 256) -> (n_win, 256, C)
        windows = (data_cropped
                   .reshape(n_channels, n_windows, samples_per_window)
                   .transpose(1, 2, 0))               # (n_windows, 256, C)

        np.save(str(out_path), windows)

        print(f"[OK]    {subj_id}  --> shape {windows.shape}  --> {out_path.name}")

    except Exception as e:
        print(f"[ERROR] {subj_id}  --> {e}")

print("\nDone.")