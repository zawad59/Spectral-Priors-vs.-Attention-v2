#!/usr/bin/env python3
# SCZ_EpochCount_only.py
#

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import mne
import numpy as np
import pandas as pd


# ============================================================
# Defaults
# ============================================================

ROOT_DIR = r"Eyes_Closed_visit1_only"
OUTPUT_CSV = r"SCZ_n_epochs.csv"

TARGET_FS = 256.0
WIN_SEC = 4.0
STRIDE_SEC = 2.0

EEG_EXTS = [".vhdr", ".edf", ".set", ".fif"]


# ============================================================
# File discovery
# ============================================================

def natural_key(path: Path):
    """Natural sort helper so BIO2 appears before BIO10."""
    text = str(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def find_eeg_files(sub_dir: Path) -> List[Path]:
    """Find supported EEG files under one subject directory."""
    files: List[Path] = []
    for ext in EEG_EXTS:
        files.extend(sub_dir.rglob(f"*{ext}"))
    return sorted(set(files), key=natural_key)


def discover_subject_dirs(root: Path) -> List[Path]:
    """
    AMP-SCZ snapshot layout is subject/visit_1/*.vhdr.

    This function also supports a root that directly contains sub-* folders
    or nested subject folders.
    """
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    # Most direct interpretation: each immediate folder is a subject folder.
    immediate_subjects = []
    for p in root.iterdir():
        if p.is_dir() and find_eeg_files(p):
            immediate_subjects.append(p)

    if immediate_subjects:
        return sorted(immediate_subjects, key=natural_key)

    # Fallback: find all parent folders that contain EEG files.
    eeg_files = find_eeg_files(root)
    subject_dirs = sorted({guess_subject_dir_from_file(fp, root) for fp in eeg_files}, key=natural_key)
    return subject_dirs


def guess_subject_dir_from_file(fp: Path, root: Path) -> Path:
    """
    Given root/BIO7844/visit_1/file.vhdr, return root/BIO7844.
    If no visit_* parent exists, return the nearest parent under root.
    """
    fp = Path(fp)
    root = Path(root)

    for parent in fp.parents:
        if parent.name.lower().startswith("visit_"):
            return parent.parent

    rel_parts = fp.relative_to(root).parts
    if len(rel_parts) >= 2:
        return root / rel_parts[0]
    return fp.parent


# ============================================================
# Raw loading and channel selection (resampling handled separately below)
# ============================================================

def load_raw(fp: Path, preload: bool = True) -> mne.io.BaseRaw:
    """Load EEG in the same supported format style as the ADFTD/TDBrain scripts."""
    fp = Path(fp)

    if fp.suffix.lower() == ".vhdr":
        return mne.io.read_raw_brainvision(fp, preload=preload, verbose="ERROR")
    if fp.suffix.lower() == ".edf":
        return mne.io.read_raw_edf(fp, preload=preload, verbose="ERROR")
    if fp.suffix.lower() == ".set":
        return mne.io.read_raw_eeglab(fp, preload=preload, verbose="ERROR")
    if fp.suffix.lower() == ".fif":
        return mne.io.read_raw_fif(fp, preload=preload, verbose="ERROR")

    raise ValueError(f"Unsupported EEG file: {fp}")


def pick_eeg(raw: mne.io.BaseRaw, average_reference: bool = True) -> mne.io.BaseRaw:
    """
    Keep EEG channels only and set average reference, as in the ADFTD/TDBrain
    scripts. Done at the recording's native sample rate, BEFORE resampling
    (referencing/channel selection don't depend on sample rate). Kept here
    (even though epoch counting doesn't need referencing) so this script's
    pipeline exactly mirrors the full feature-extraction script -- any file
    that would fail/behave differently there fails/behaves the same way here.
    """
    picks = mne.pick_types(
        raw.info,
        eeg=True,
        meg=False,
        eog=False,
        stim=False,
        ecg=False,
        emg=False,
        misc=False,
        exclude="bads",
    )

    if len(picks) == 0:
        raise RuntimeError("No EEG channels found after MNE channel picking.")

    raw = raw.copy().pick(picks)

    if average_reference:
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    return raw


# ============================================================
# Epoching (sliding window, matching TDBrain_fft_dwt.py's load_csv)
# ============================================================

def segment_sliding(data_CxT: np.ndarray, fs: float, win_sec: float, stride_sec: float) -> List[np.ndarray]:
    win = int(win_sec * fs)
    hop = int(stride_sec * fs)
    T = data_CxT.shape[1]
    epochs: List[np.ndarray] = []
    for start in range(0, max(T - win + 1, 1), max(hop, 1)):
        seg = data_CxT[:, start:start + win]
        if seg.shape[1] >= max(int(fs), 32):
            epochs.append(seg)
    return epochs if epochs else [data_CxT]


# ============================================================
# Epoch counting (no feature computation)
# ============================================================

def n_epochs_for_file(fp: Path, target_fs: float, win_sec: float, stride_sec: float) -> int:
    """
    Loads one raw recording, downsamples to target_fs via MNE's built-in
    raw.resample() (proper anti-aliased resampling, not linear interpolation),
    segments into sliding-window epochs, and returns just the epoch count.
    """
    raw = load_raw(fp, preload=True)

    # Drop AMP-SCZ auxiliary/non-scalp channel if present (must happen before
    # pick_eeg(), since MNE may otherwise classify VIS as EEG).
    if "VIS" in raw.ch_names:
        print(f"[INFO] Dropping VIS channel from: {fp}")
        raw.drop_channels(["VIS"])

    raw = pick_eeg(raw, average_reference=True)

    native_fs = float(raw.info["sfreq"])
    if abs(native_fs - target_fs) > 1e-6:
        raw.resample(target_fs, verbose="ERROR")

    data = raw.get_data()  # (C, T), already at target_fs
    epochs = segment_sliding(data, fs=target_fs, win_sec=win_sec, stride_sec=stride_sec)

    return len(epochs)


def n_epochs_for_subject(
    files: List[Path], subj_id: str, target_fs: float, win_sec: float, stride_sec: float
) -> tuple[int, int]:
    """
    Sums n_epochs across multiple files/runs for the same subject.
    Returns (total_n_epochs, n_files_used).
    """
    total_n_epochs = 0
    n_files_used = 0

    for fp in files:
        try:
            n = n_epochs_for_file(fp, target_fs=target_fs, win_sec=win_sec, stride_sec=stride_sec)
            total_n_epochs += n
            n_files_used += 1
            print(f"[EPOCHS] subj {subj_id}  file={fp.name}  n_epochs={n}")
        except Exception as e:
            print(f"[WARN] Skipping file due to error: {fp}")
            print(f"       {e}")

    return total_n_epochs, n_files_used


# ============================================================
# Main loop
# ============================================================

def process_all_subjects(
    root_dir: str | Path, output_csv: str | Path,
    target_fs: float, win_sec: float, stride_sec: float,
) -> pd.DataFrame:
    root = Path(root_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    subject_dirs = discover_subject_dirs(root)

    print(f"[INFO] Root: {root}")
    print(f"[INFO] Found {len(subject_dirs)} subject folders with EEG files.")
    print(f"[INFO] target_fs={target_fs}  win_sec={win_sec}  stride_sec={stride_sec}  resampling=linear-interpolation")

    rows = []

    for subject_dir in subject_dirs:
        subj_id = subject_dir.name
        eeg_files = find_eeg_files(subject_dir)

        # Prefer .vhdr for BrainVision so the linked .eeg/.vmrk are read correctly.
        vhdr_files = [fp for fp in eeg_files if fp.suffix.lower() == ".vhdr"]
        files_to_process = vhdr_files if vhdr_files else eeg_files

        if not files_to_process:
            print(f"[INFO] No EEG files found for {subj_id}. Skipping.")
            continue

        n_epochs, n_files_used = n_epochs_for_subject(
            files_to_process, subj_id, target_fs=target_fs, win_sec=win_sec, stride_sec=stride_sec
        )

        if n_files_used > 0:
            # subj_ID printed right next to its n_epochs
            print(f"[OK] {subj_id}: n_epochs={n_epochs}  (files_used={n_files_used})")
            rows.append({"subj_ID": subj_id, "n_epochs": n_epochs, "n_files_used": n_files_used})
        else:
            print(f"[WARN] No valid epochs computed for {subj_id}.")

    if not rows:
        raise RuntimeError("No epochs computed for any subject.")

    df = pd.DataFrame(rows).sort_values("subj_ID").reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    # ---- summary stats across all subjects ----
    counts = df["n_epochs"].to_numpy(dtype=float)
    print("\n[SUMMARY] n_epochs per subject:")
    print(f"  n_subjects = {len(counts)}")
    print(f"  mean       = {counts.mean():.2f}")
    print(f"  median     = {np.median(counts):.2f}")
    print(f"  min        = {counts.min():.0f}")
    print(f"  max        = {counts.max():.0f}")
    print(f"  std        = {counts.std(ddof=1):.2f}" if len(counts) > 1 else "  std        = n/a (only 1 subject)")

    print("\n[DONE]")
    print(f"[OK] Wrote: {output_csv}")
    print(f"[OK] Shape: {df.shape}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Count epochs per subject from AMP-SCZ raw EEG files, using "
            "linear-interpolation resampling and sliding-window epoching "
            "(no feature computation). Output is one CSV row per subject: "
            "subj_ID, n_epochs, n_files_used."
        )
    )

    parser.add_argument(
        "--root",
        type=str,
        default=ROOT_DIR,
        help="Root directory containing AMP-SCZ subject folders, e.g. Eyes_Closed_visit1_only.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=OUTPUT_CSV,
        help="Output CSV path.",
    )

    parser.add_argument("--fs", type=float, default=TARGET_FS, help="Target sampling rate (Hz) after resampling.")
    parser.add_argument("--win-sec", type=float, default=WIN_SEC, help="Window length (s) for epoching.")
    parser.add_argument("--stride-sec", type=float, default=STRIDE_SEC, help="Stride (s) between windows.")

    args = parser.parse_args()

    process_all_subjects(
        args.root, args.out,
        target_fs=args.fs, win_sec=args.win_sec, stride_sec=args.stride_sec,
    )


if __name__ == "__main__":
    main()
