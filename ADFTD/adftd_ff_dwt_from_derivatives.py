#!/usr/bin/env python3
# adftd_features_from_derivatives.py
#Calculation of Spectral Features

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
import pywt

mne.set_log_level("WARNING")

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


def _norm_sid(x) -> str:
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    if m:
        return m.group(1).lstrip("0") or "0"
    return s


# ---------------------- feature extraction (unchanged math) ----------------------
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


# ---------------------- loading raw .set files from derivatives/ ----------------------
def find_subject_dirs(root: Path) -> List[Path]:
    """sub-XXX directories directly under root, sorted by numeric subject index."""
    subs = [p for p in root.glob("sub-*") if p.is_dir()]

    def _num(p: Path) -> int:
        m = re.search(r"(\d+)", p.name)
        return int(m.group(1)) if m else -1

    return sorted(subs, key=_num)


def find_set_file(sub_dir: Path, task: str) -> Path | None:
    """Locate the .set file for a subject, preferring one matching --task."""
    eeg_dir = sub_dir / "eeg"
    search_dir = eeg_dir if eeg_dir.is_dir() else sub_dir
    set_files = sorted(search_dir.glob("*.set"))

    if not set_files:
        return None
    if len(set_files) == 1:
        return set_files[0]

    matches = [f for f in set_files if task.lower() in f.name.lower()]
    if matches:
        return matches[0]

    print(f"  [WARN] {sub_dir.name}: {len(set_files)} .set files found, none matched "
          f"task='{task}'; skipping. Files seen: {[f.name for f in set_files]}")
    return None


def _contiguous_segment_bounds(raw: "mne.io.Raw") -> List[Tuple[int, int]]:
    """
    Return [(start_sample, end_sample), ...] for each contiguous stretch of
    data, splitting at any 'boundary' annotation (EEGLAB's marker for
    discontinuities from concatenated/edited recordings). Windowing within
    a segment guarantees no window straddles a discontinuity.
    """
    sfreq = raw.info["sfreq"]
    n_samples = len(raw.times)

    boundary_onsets = [
        ann["onset"] for ann in raw.annotations
        if "boundary" in ann["description"].lower()
    ]
    # onsets are in seconds relative to raw start; convert to sample indices
    cut_samples = sorted(
        int(round((onset - raw.first_time) * sfreq)) for onset in boundary_onsets
    )
    cut_samples = [c for c in cut_samples if 0 < c < n_samples]

    bounds = []
    prev = 0
    for c in cut_samples:
        bounds.append((prev, c))
        prev = c
    bounds.append((prev, n_samples))
    return bounds


def load_windows_CxT(fp: Path, win_sec: float, fs_override: float | None) -> Tuple[List[np.ndarray], float]:
    """
    Load a continuous EEGLAB .set recording and split it into non-overlapping
    (C, T) windows of length win_sec. Splits at 'boundary' annotations first
    so no window spans a data discontinuity (leftover samples at the end of
    each segment, smaller than one window, are dropped). Returns
    (list_of_windows, fs_used).
    """
    raw = mne.io.read_raw_eeglab(str(fp), preload=True)
    native_fs = float(raw.info["sfreq"])

    fs = fs_override if fs_override else native_fs
    if fs_override and abs(fs_override - native_fs) > 1e-6:
        raw.resample(fs_override)
        fs = fs_override

    segment_bounds = _contiguous_segment_bounds(raw)
    data = raw.get_data()  # (C, n_samples), volts
    win_len = int(round(win_sec * fs))

    windows: List[np.ndarray] = []
    for seg_start, seg_end in segment_bounds:
        seg = data[:, seg_start:seg_end]
        n_windows = seg.shape[1] // win_len
        for i in range(n_windows):
            windows.append(seg[:, i * win_len:(i + 1) * win_len])

    if len(segment_bounds) > 1:
        print(f"  [INFO] {fp.name}: split into {len(segment_bounds)} contiguous segments "
              f"at boundary annotations before windowing "
              f"(segment lengths in samples: {[b - a for a, b in segment_bounds]})")

    return windows, fs


def features_for_subject(fp: Path, win_sec: float, fs_override: float | None, wavelet: str) -> Tuple[Dict[str, float], float, int]:
    windows, fs = load_windows_CxT(fp, win_sec, fs_override)
    rows = [features_for_epoch(w, fs, wavelet) for w in windows]
    feats = pd.DataFrame(rows).mean(axis=0, numeric_only=True).to_dict()
    return feats, fs, len(windows)


def main():
    ap = argparse.ArgumentParser(
        description="Run FFT/DWT/spectral-summary feature extraction directly on raw "
                    "derivatives/sub-XXX/eeg/*.set files (BIDS-style), instead of "
                    "precomputed feature_XX.npy epoch arrays."
    )
    ap.add_argument("--root", type=str, required=True,
                     help="Path to the derivatives directory containing sub-XXX/eeg/*.set files")
    ap.add_argument("--out", type=str, default="adftd_fft_dwt_from_derivatives.csv", help="Output CSV path")
    ap.add_argument(
        "--labels", type=str, required=True,
        help="Path to label.npy OR a folder containing label.npy (shape (N,2): [label, subj_ID])."
    )
    ap.add_argument("--task", type=str, default="eyesclosed",
                     help="Substring used to pick the right .set file when a subject has more than one.")
    ap.add_argument("--win-sec", type=float, default=1.0, help="Window length in seconds for feature extraction.")
    ap.add_argument("--fs", type=float, default=256.0,
                     help="Force this sampling rate (resamples each file). Default: use each file's native rate.")
    ap.add_argument("--wavelet", type=str, default="db4", help="Wavelet (e.g. db4, sym5).")
    args = ap.parse_args()

    root = Path(args.root)
    sub_dirs = find_subject_dirs(root)

    print(f"[DEBUG] root = {root.resolve()}")
    if not sub_dirs:
        print(f"[ERROR] No sub-* directories found under {root}")
        return
    print(f"[INFO] Found {len(sub_dirs)} subject directories under {root}")

    # resolve subject -> .set file, skipping subjects with no usable file
    sub_files: List[Tuple[Path, Path]] = []
    for sub_dir in sub_dirs:
        fp = find_set_file(sub_dir, args.task)
        if fp is None:
            print(f"  [WARN] {sub_dir.name}: no matching .set file found, skipping.")
            continue
        sub_files.append((sub_dir, fp))

    if not sub_files:
        print("[ERROR] No usable .set files found.")
        return

    print(f"[DEBUG] {sub_files[0][1].name}  (first file to be loaded to confirm channels/fs)")

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
            f"these should match 1:1 by sub-XXX numeric order if the row-order assumption holds."
        )

    n_pairs = min(len(sub_files), labels_arr.shape[0])

    print("\n[VERIFY] First 5 and last 5 (sub-dir -> assumed label.npy row) -- "
          "check this against what you know about the folder<->subj_ID mapping before trusting the output:")
    preview_idxs = sorted(set(list(range(min(5, n_pairs))) + list(range(max(0, n_pairs - 5), n_pairs))))
    for i in preview_idxs:
        lab, sid = labels_arr[i, 0], labels_arr[i, 1]
        sub_dir, fp = sub_files[i]
        print(f"  {sub_dir.name} ({fp.name})  ->  label.npy row {i}: label={lab}, subj_ID={sid}")
    print()

    rows = []
    for i in range(n_pairs):
        sub_dir, fp = sub_files[i]
        lab, sid = labels_arr[i, 0], labels_arr[i, 1]

        try:
            feats, fs_used, n_windows = features_for_subject(
                fp, win_sec=args.win_sec, fs_override=args.fs, wavelet=args.wavelet
            )
        except Exception as e:
            print(f"[WARN] Skipping {sub_dir.name} ({fp.name}): {e}")
            continue

        row = {k: feats.get(k, np.nan) for k in FEATURE_ORDER}
        row["subj_ID"] = _norm_sid(sid)

        lab_s = str(lab).strip()
        row["Group"] = int(lab_s) if lab_s.lstrip("-").isdigit() else lab

        rows.append(row)
        print(f"[OK] {sub_dir.name} ({fp.name})  fs={fs_used:.1f}Hz  n_windows={n_windows}  ->  subj_ID={row['subj_ID']}")

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
