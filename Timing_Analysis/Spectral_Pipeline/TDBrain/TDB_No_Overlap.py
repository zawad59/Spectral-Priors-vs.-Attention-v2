#!/usr/bin/env python3
# tdbrain_features_from_csv.py
#
# Same FFT/DWT/spectral-summary feature extraction used elsewhere in this
# project, adapted for TDBrain's per-subject CSV exports instead of
# ADFTD's raw EEGLAB .set files or APAVA's FieldTrip .mat files.
#
# INPUT LAYOUT -- flat directory, one CSV per subject/session/task:
#   <root>/sub-87974617_ses-1_task-restEC_eeg.csv
#   <root>/sub-87974621_ses-1_task-restEC_eeg.csv
#   ...
# (no per-subject subfolders, unlike ADFTD's sub-XXX/eeg/*.set layout)
#
# UNVERIFIED ASSUMPTIONS -- I have NOT seen an actual TDBrain CSV from
# this export, only the file listing. Everything below is inferred from
# the filenames and general knowledge of the public TDBRAIN dataset
# format, NOT confirmed against your data. VERIFY before trusting output:
#
#   1. CSV LAYOUT: assumed wide format -- rows = time samples, columns =
#      channels, with a header row of channel names. If your CSVs are
#      transposed (channels as rows) or have a leading time/index column
#      under an unexpected name, this will silently produce garbage --
#      the [DEBUG] column-list printout at the start of each run exists
#      specifically so you can catch that before trusting anything.
#
#   2. SAMPLING RATE: CSVs have no way to self-report this. --fs defaults
#      to 500.0 (the published TDBRAIN dataset's native rate) but this is
#      NOT confirmed for this specific export -- if your CSVs were
#      resampled or this isn't actually TDBRAIN-standard data, override
#      --fs explicitly.
#
#   3. NON-EEG CHANNELS: the public TDBRAIN release includes a few
#      auxiliary channels (commonly VPVA/VNVB/HPOL/Erbs/OrbOcc/Mass) that
#      are usually dropped before feature extraction, similar to the
#      AMP-SCZ pipeline's VIS-channel drop. This script does NOT drop
#      anything by default -- I'm not certain these are present, named
#      the same way, or even included in this particular export. Use
#      --drop-channels (comma-separated column names) once you've checked
#      the printed column list.
#
#   4. SUBJECT-ID / LABEL ALIGNMENT: mirrors the same row-order convention
#      as the ADFTD/APAVA scripts (numeric sub-ID order <-> label.npy row
#      order) as a default, since that's the established pattern in this
#      project. BUT the folder name "TDB_3class_Formal_Status" suggests
#      TDBrain might use a 3-CLASS label scheme, and possibly organizes
#      labels by folder/status rather than a flat label.npy at all --
#      neither of which I can confirm from a file listing. This is flagged
#      prominently, and the [VERIFY] preview is printed before writing
#      output either way -- but if labels here don't actually work by
#      label.npy positional order, this part needs to change before
#      trusting the CSV.
#
#   5. ONE CONTIGUOUS RECORDING PER FILE: unlike ADFTD (which splits at
#      EEGLAB 'boundary' annotations), a CSV has no such marker, so each
#      file is treated as ONE continuous segment. If a subject's export
#      actually concatenates multiple discontinuous recordings, windows
#      near a splice point will incorrectly span the discontinuity -- I
#      have no way to detect that from a CSV.

from __future__ import annotations
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch
import pywt

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
SUMMARY_FMIN, SUMMARY_FMAX = 0.5, 45.0

FEATURE_ORDER = (
    [f"FFT_{k}" for k in BANDS] +
    [f"DWT_{k}" for k in BANDS] +
    ["SPEC_mean_freq", "SPEC_median_freq", "SPEC_entropy"]
)

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

# Columns to drop if present, case-insensitively, before treating the rest
# as channels -- an obvious time/index column, not an EEG channel. Extend
# via --drop-channels for anything dataset-specific (auxiliary channels
# etc. -- see assumption #3 above).
_OBVIOUS_NON_CHANNEL_COLS = {"time", "timestamp", "index", "unnamed: 0", "sample"}


def _norm_sid(x) -> str:
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    if m:
        return m.group(1).lstrip("0") or "0"
    return s


# ---------------------- feature extraction (unchanged math -- identical to ADFTD/APAVA) ----------------------
def _welch_band_power(x: np.ndarray, fs: float, band: Tuple[float, float]) -> float:
    nperseg = max(int(min(4 * fs, len(x))), int(fs))
    noverlap = min(nperseg // 2, max(nperseg // 4, 0))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann")
    idx = (f >= band[0]) & (f <= band[1])
    return float(_trapz(Pxx[idx], f[idx]))


def relative_bandpowers_fft(epoch_CxT: np.ndarray, fs: float) -> Dict[str, float]:
    C = epoch_CxT.shape[0]
    abs_by_band = {k: np.zeros(C) for k in BANDS}
    for ci, ch in enumerate(epoch_CxT):
        for name, band in BANDS.items():
            abs_by_band[name][ci] = _welch_band_power(ch, fs, band)

    totals = np.zeros(C)
    for name in BANDS:
        totals += abs_by_band[name]

    out: Dict[str, float] = {}
    denom = np.maximum(totals, 1e-12)
    for name in BANDS:
        rel_ch = abs_by_band[name] / denom
        out[f"FFT_{name}"] = float(np.nanmean(rel_ch))
    return out


def spectral_summary_features(epoch_CxT: np.ndarray, fs: float) -> Dict[str, float]:
    T = epoch_CxT.shape[1]
    nperseg = max(int(min(4 * fs, T)), int(fs))
    noverlap = min(nperseg // 2, max(nperseg // 4, 0))
    means, medians, ents = [], [], []

    for ch in epoch_CxT:
        f, Pxx = welch(ch, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann")
        m = (f >= SUMMARY_FMIN) & (f <= SUMMARY_FMAX)
        f, P = f[m], Pxx[m].clip(min=0)
        if f.size == 0 or np.all(P == 0):
            means.append(np.nan); medians.append(np.nan); ents.append(np.nan)
            continue
        p = P / np.sum(P)
        cdf = np.cumsum(p)
        means.append(float(np.sum(f * p)))
        medians.append(float(np.interp(0.5, cdf, f)))
        nz = p[p > 0]
        ents.append(float(-np.sum(nz * np.log(nz))))

    return {
        "SPEC_mean_freq": float(np.nanmean(means)),
        "SPEC_median_freq": float(np.nanmean(medians)),
        "SPEC_entropy": float(np.nanmean(ents)),
    }


def dwt_band_energies_overlap(
    epoch_CxT: np.ndarray, fs: float, wavelet: str = "db4",
    max_level: int = 8, include_a_in_delta: bool = True
) -> Dict[str, float]:
    w = pywt.Wavelet(wavelet)
    out = {f"DWT_{k}": [] for k in BANDS}

    for ch in epoch_CxT:
        Lmax = pywt.dwt_max_level(len(ch), w.dec_len)
        L = max(1, min(max_level, Lmax))
        coeffs = pywt.wavedec(ch, wavelet=w, level=L, mode="periodization")
        cA, *cDs = coeffs
        E_A = float(np.dot(cA, cA))
        E_levels = [float(np.dot(c, c)) for c in cDs]

        band_energy = {k: 0.0 for k in BANDS}
        for idx, E in enumerate(E_levels):
            j = L - idx
            f_lo = fs / (2 ** (j + 1))
            f_hi = fs / (2 ** j)
            span = f_hi - f_lo
            if span <= 0:
                continue
            for name, (b_lo, b_hi) in BANDS.items():
                ov = max(0.0, min(f_hi, b_hi) - max(f_lo, b_lo))
                if ov > 0:
                    band_energy[name] += E * (ov / span)

        if include_a_in_delta:
            a_hi = fs / (2 ** (L + 1))
            span = a_hi
            if span > 0:
                b_lo, b_hi = BANDS["delta"]
                ov = max(0.0, min(a_hi, b_hi) - max(0.0, b_lo))
                if ov > 0:
                    band_energy["delta"] += E_A * (ov / span)

        denom = sum(band_energy.values()) + 1e-12
        for name in BANDS:
            out[f"DWT_{name}"].append(band_energy[name] / denom)

    return {k: float(np.nanmean(v)) for k, v in out.items()}


def detrend_dc(epoch_CxT: np.ndarray) -> np.ndarray:
    return epoch_CxT - np.mean(epoch_CxT, axis=1, keepdims=True)


def features_for_epoch(epoch_CxT: np.ndarray, fs: float, wavelet: str) -> Dict[str, float]:
    X = detrend_dc(epoch_CxT)
    out: Dict[str, float] = {}
    out.update(relative_bandpowers_fft(X, fs))
    out.update(dwt_band_energies_overlap(X, fs, wavelet=wavelet))
    out.update(spectral_summary_features(X, fs))
    return out


# ---------------------- loading sub-XXX_ses-N_task-TASK_eeg.csv files ----------------------
_FILENAME_RE = re.compile(r"^sub-(\d+)_ses-(\d+)_task-([A-Za-z0-9]+)_eeg\.csv$", re.IGNORECASE)


def find_subject_files(root: Path, task: str) -> List[Path]:
    """
    Flat directory of sub-XXX_ses-N_task-TASK_eeg.csv files -- NOT
    per-subject subfolders like ADFTD. Filters to filenames whose task
    group matches --task (case-insensitive), sorted by the numeric subject
    ID in the filename. If a subject has more than one matching file
    (e.g. multiple sessions), keeps the first (sorted by filename) and
    warns -- check the [WARN] output if that matters for your data.
    """
    candidates = sorted(root.glob("sub-*_eeg.csv"))
    by_subject: Dict[str, Path] = {}
    skipped_no_match = 0

    for fp in candidates:
        m = _FILENAME_RE.match(fp.name)
        if not m:
            skipped_no_match += 1
            continue
        sub_id, ses, file_task = m.groups()
        if task.lower() not in file_task.lower():
            continue
        if sub_id in by_subject:
            print(f"  [WARN] sub-{sub_id}: multiple matching files found, "
                  f"keeping {by_subject[sub_id].name}, skipping {fp.name}")
            continue
        by_subject[sub_id] = fp

    if skipped_no_match:
        print(f"  [WARN] {skipped_no_match} file(s) under {root} didn't match the expected "
              f"sub-<ID>_ses-<N>_task-<TASK>_eeg.csv naming pattern and were skipped.")

    def _num(sub_id: str) -> int:
        return int(sub_id)

    return [by_subject[sid] for sid in sorted(by_subject, key=_num)]


def load_windows_CxT(
    fp: Path, win_sec: float, fs: float, drop_channels: List[str]
) -> Tuple[List[np.ndarray], float, List[str]]:
    """
    Load one TDBrain CSV and split it into non-overlapping (C, T) windows
    of length win_sec. ASSUMES rows=samples, columns=channels, header row
    present (assumption #1 in the module docstring -- unverified). No
    boundary-annotation concept exists for a CSV, so the whole file is
    treated as ONE contiguous segment (assumption #5). Trailing samples
    that don't fill a whole window are dropped, same convention as
    ADFTD/APAVA. Returns (list_of_windows, fs, channel_names_used).
    """
    df = pd.read_csv(fp)

    drop_set = {c.strip().lower() for c in drop_channels}
    keep_cols = [
        c for c in df.columns
        if c.strip().lower() not in _OBVIOUS_NON_CHANNEL_COLS
        and c.strip().lower() not in drop_set
    ]
    dropped = [c for c in df.columns if c not in keep_cols]
    if dropped:
        print(f"  [INFO] {fp.name}: dropped non-channel column(s): {dropped}")

    data = df[keep_cols].to_numpy(dtype=float).T  # (C, T): rows->samples transposed to channels x samples

    win_len = int(round(win_sec * fs))
    T = data.shape[1]
    n_windows = T // win_len

    windows: List[np.ndarray] = []
    for i in range(n_windows):
        windows.append(data[:, i * win_len:(i + 1) * win_len])

    return windows, fs, keep_cols


def features_for_subject(
    fp: Path, win_sec: float, fs: float, wavelet: str, drop_channels: List[str]
) -> Tuple[Dict[str, float], float, int, List[str]]:
    windows, fs, channels = load_windows_CxT(fp, win_sec, fs, drop_channels)
    rows = [features_for_epoch(w, fs, wavelet) for w in windows]
    feats = pd.DataFrame(rows).mean(axis=0, numeric_only=True).to_dict()
    return feats, fs, len(windows), channels


def main():
    ap = ArgumentParser(
        description="Run FFT/DWT/spectral-summary feature extraction directly on TDBrain "
                    "sub-<ID>_ses-<N>_task-<TASK>_eeg.csv files, using 1s non-overlapping "
                    "windows. See module docstring for unverified format assumptions."
    )
    ap.add_argument("--root", type=str, required=True,
                     help="Path to the directory containing sub-*_eeg.csv files")
    ap.add_argument("--out", type=str, default="tdbrain_fft_dwt.csv", help="Output CSV path")
    ap.add_argument(
        "--labels", type=str, required=True,
        help="Path to label.npy OR a folder containing label.npy (shape (N,2): [label, subj_ID]). "
             "See assumption #4 -- this may not be how TDBrain labels actually work, VERIFY."
    )
    ap.add_argument("--task", type=str, default="restEC",
                     help="Substring matched against each file's task field.")
    ap.add_argument("--win-sec", type=float, default=1.0,
                     help="Non-overlapping window length in seconds.")
    ap.add_argument("--fs", type=float, default=256.0,
                     help="Sampling rate in Hz -- CANNOT be read from the CSV itself. "
                          "Default 256.0 is the published TDBRAIN dataset's native rate, "
                          "UNCONFIRMED for this export -- verify before trusting output.")
    ap.add_argument("--wavelet", type=str, default="db4", help="Wavelet (e.g. db4, sym5).")
    ap.add_argument("--drop-channels", type=str, default="",
                     help="Comma-separated column names to exclude as non-EEG channels "
                          "(e.g. VPVA,VNVB,HPOL,Erbs,OrbOcc,Mass), on top of the always-"
                          "dropped obvious time/index columns. Check the printed column "
                          "list from a first run before setting this.")
    args = ap.parse_args()

    drop_channels = [c.strip() for c in args.drop_channels.split(",") if c.strip()]

    root = Path(args.root)
    sub_files = find_subject_files(root, task=args.task)

    print(f"[DEBUG] root = {root.resolve()}")
    if not sub_files:
        print(f"[ERROR] No sub-*_eeg.csv files matching task='{args.task}' found under {root}")
        return
    print(f"[INFO] Found {len(sub_files)} subject file(s) under {root} matching task='{args.task}'")

    # Load just the header of the first file to show columns up front --
    # this is the main defense against assumption #1 (CSV layout) being
    # wrong; check this output before trusting a full run.
    first_cols = list(pd.read_csv(sub_files[0], nrows=0).columns)
    print(f"[DEBUG] {sub_files[0].name}  columns ({len(first_cols)}): {first_cols}")

    label_base = Path(args.labels)
    label_fp = (label_base / "label.npy") if label_base.is_dir() else label_base
    if not label_fp.exists():
        raise FileNotFoundError(f"Label file not found: {label_fp}")

    labels_arr = np.asarray(np.load(label_fp, allow_pickle=True), dtype=object)
    if labels_arr.ndim != 2 or labels_arr.shape[1] != 2:
        raise ValueError(f"Expected label.npy shape (N,2) [label, subj_ID], got {labels_arr.shape}")

    if len(sub_files) != labels_arr.shape[0]:
        print(
            f"[WARN] {len(sub_files)} subject files but {labels_arr.shape[0]} label rows -- "
            f"these should match 1:1 by numeric sub-ID order if the row-order assumption holds "
            f"(assumption #4 -- unverified for TDBrain, check this)."
        )

    n_pairs = min(len(sub_files), labels_arr.shape[0])

    print("\n[VERIFY] First 5 and last 5 (subject file -> assumed label.npy row) -- "
          "check this against what you know about the file<->subj_ID mapping before trusting "
          "the output. See assumption #4: TDBrain may not actually use this row-order scheme.")
    preview_idxs = sorted(set(list(range(min(5, n_pairs))) + list(range(max(0, n_pairs - 5), n_pairs))))
    for i in preview_idxs:
        lab, sid = labels_arr[i, 0], labels_arr[i, 1]
        fp = sub_files[i]
        print(f"  {fp.name}  ->  label.npy row {i}: label={lab}, subj_ID={sid}")
    print()

    rows = []
    for i in range(n_pairs):
        fp = sub_files[i]
        lab, sid = labels_arr[i, 0], labels_arr[i, 1]

        try:
            feats, fs_used, n_windows, channels = features_for_subject(
                fp, win_sec=args.win_sec, fs=args.fs, wavelet=args.wavelet,
                drop_channels=drop_channels
            )
        except Exception as e:
            print(f"[WARN] Skipping {fp.name}: {e}")
            continue

        row = {k: feats.get(k, np.nan) for k in FEATURE_ORDER}
        row["subj_ID"] = _norm_sid(sid)

        lab_s = str(lab).strip()
        row["Group"] = int(lab_s) if lab_s.lstrip("-").isdigit() else lab

        rows.append(row)
        print(f"[OK] {fp.name}  fs={fs_used:.1f}Hz  n_channels={len(channels)}  "
              f"n_windows={n_windows}  ->  subj_ID={row['subj_ID']}")

    if not rows:
        print("[ERROR] No features computed.")
        return

    cols = ["subj_ID"] + FEATURE_ORDER + ["Group"]
    df = pd.DataFrame(rows)[cols]
    df = df.sort_values("subj_ID").reset_index(drop=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[OK] Wrote {args.out} with shape {df.shape}")


if __name__ == "__main__":
    main()