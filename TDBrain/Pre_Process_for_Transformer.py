#!/usr/bin/env python3
# tdbrain_parkinson_csv_to_feature_npy.py


from __future__ import annotations
import os
import re
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import interpolate


# ---------------------- verbatim port of the notebook's functions ----------------------
def resampling(array: np.ndarray, freq: float = 500, kind: str = "linear") -> np.ndarray:
    """Exact port of the notebook's resampling() -- per-channel linear interpolation."""
    t = np.linspace(1, len(array), len(array))
    f = interpolate.interp1d(t, array, kind=kind)
    t_new = np.linspace(1, len(array), int(len(array) / freq * 256))
    new_array = f(t_new)
    return new_array


def segment(df: pd.DataFrame, window_size: int = 256) -> List[pd.DataFrame]:
    """Exact port of the notebook's segment() -- non-overlapping windows, starting from the middle."""
    res = []
    start = int(df.shape[0] / 2)
    left_index = start - int(start / window_size) * window_size
    right_index = start + int((df.shape[0] - start) / window_size) * window_size
    for i in range(left_index, right_index, window_size):
        res.append(df.iloc[i: i + window_size, :])
    return res


def eeg_data(eeg_path: str, src_fs: float, window_size: int = 256) -> List[pd.DataFrame]:
    """Exact port of the notebook's eeg_data() -- resample every channel, then segment."""
    df = pd.read_csv(eeg_path)
    trial = []
    for i in range(df.shape[1]):
        data = resampling(df.iloc[:, i].values, freq=src_fs, kind="linear")
        data = pd.Series(data)
        trial.append(data)
    df_new = pd.concat(trial, axis=1, ignore_index=True)
    res_df = segment(df_new, window_size=window_size)
    return res_df


# ---------------------- filename <-> subj_ID matching ----------------------
def _extract_subj_id(fname: str) -> Optional[str]:
    """Pull the numeric subj_ID out of a filename like sub-87974617_ses-1_task-restEC_eeg.csv"""
    m = re.match(r"sub-(\d+)", fname)
    return m.group(1) if m else None


def _norm_id(x) -> str:
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    return m.group(1).lstrip("0") or "0" if m else s


def main():
    ap = argparse.ArgumentParser(
        description="Convert combined EC CSV files into per-subject feature_{subj_ID}.npy files, "
                    "using the Medformer notebook's exact resample+segment logic."
    )
    ap.add_argument("--csv-dir", type=str, required=True,
                     help="Folder containing sub-XXXXXXXX_..._task-restEC_eeg.csv files")
    ap.add_argument("--labels", type=str, default=None,
                     help="Optional: path to label.npy OR a folder containing label.npy "
                          "(shape (N,2): [label, subj_ID]). Only used to print a cross-check "
                          "of which subj_IDs matched a CSV -- feature generation itself doesn't "
                          "need it, since subj_ID comes straight from each CSV's filename.")
    ap.add_argument("--out", type=str, default="./Feature", help="Output directory for feature_{subj_ID}.npy")
    ap.add_argument("--src-fs", type=float, default=500.0, help="Native sampling rate of the CSVs (Hz)")
    ap.add_argument("--window-size", type=int, default=256, help="Window length in samples after resampling")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # index csv files by subj_ID
    csv_files = sorted(csv_dir.glob("sub-*.csv"))
    if not csv_files:
        print(f"[ERROR] No sub-*.csv files found under {csv_dir}")
        return

    csv_by_id = {}
    for fp in csv_files:
        sid = _extract_subj_id(fp.name)
        if sid is None:
            print(f"[WARN] Could not parse subj_ID from {fp.name}, skipping.")
            continue
        if sid in csv_by_id:
            print(f"[WARN] Multiple CSVs found for subj_ID={sid} "
                  f"({csv_by_id[sid].name} and {fp.name}); keeping the first.")
            continue
        csv_by_id[sid] = fp

    print(f"[INFO] Found {len(csv_by_id)} subject CSV files under {csv_dir}")

    # optional: cross-check against label.npy (does not gate generation)
    if args.labels:
        label_base = Path(args.labels)
        label_fp = (label_base / "label.npy") if label_base.is_dir() else label_base
        if not label_fp.exists():
            raise FileNotFoundError(f"Label file not found: {label_fp}")

        labels_arr = np.asarray(np.load(label_fp, allow_pickle=True), dtype=object)
        if labels_arr.ndim != 2 or labels_arr.shape[1] != 2:
            raise ValueError(f"Expected label.npy shape (N,2) [label, subj_ID], got {labels_arr.shape}")

        label_ids = [_norm_id(labels_arr[i, 1]) for i in range(labels_arr.shape[0])]
        missing = [sid for sid in label_ids if sid not in csv_by_id]
        print(f"\n[VERIFY] {len(label_ids)} subjects in label.npy, "
              f"{len(label_ids) - len(missing)} matched to a CSV file, {len(missing)} missing.")
        if missing:
            print(f"  [WARN] No CSV found for subj_ID(s): {missing}")
        print()

    n_ok, n_skipped = 0, 0
    for sid, fp in csv_by_id.items():
        try:
            windows = eeg_data(str(fp), src_fs=args.src_fs, window_size=args.window_size)
            array_sub = np.array([w.values for w in windows])
        except Exception as e:
            print(f"[WARN] Skipping subj_ID={sid} ({fp.name}): {e}")
            n_skipped += 1
            continue

        out_fp = out_dir / f"feature_{sid}.npy"
        np.save(out_fp, array_sub)
        print(f"[OK] subj_ID={sid} ({fp.name})  ->  {out_fp.name}  shape={array_sub.shape}")
        n_ok += 1

    print(f"\n[DONE] Wrote {n_ok} feature files to {out_dir} ({n_skipped} subjects skipped).")


if __name__ == "__main__":
    main()
