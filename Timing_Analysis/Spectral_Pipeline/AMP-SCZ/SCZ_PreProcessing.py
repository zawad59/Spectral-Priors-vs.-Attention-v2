#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
import pywt
from scipy.signal import welch


# ============================================================
# Defaults
# ============================================================

ROOT_DIR = r"Eyes_Closed_visit1_only"
OUTPUT_CSV = r"SCZ_FFT_DWT.csv"

TARGET_FS = 256.0
WIN_SEC = 1.0
WAVELET = "db4"

BANDS: Dict[str, Tuple[float, float]] = {
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

EEG_EXTS = [".vhdr", ".edf", ".set", ".fif"]

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ============================================================
# File discovery (AMP-SCZ specific, unchanged)
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
# Raw loading (AMP-SCZ specific, unchanged)
# ============================================================

def load_raw(fp: Path, preload: bool = True) -> mne.io.BaseRaw:
    """Load EEG in the same supported format style as the ADFTD script."""
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


def ensure_fs(raw: mne.io.BaseRaw, target_fs: float = TARGET_FS) -> mne.io.BaseRaw:
    """Resample to target frequency, matching ADFTD target_fs=256 Hz."""
    if abs(float(raw.info["sfreq"]) - float(target_fs)) > 1e-6:
        raw = raw.copy().resample(target_fs, verbose="ERROR")
    return raw


def pick_eeg(raw: mne.io.BaseRaw, average_reference: bool = True) -> mne.io.BaseRaw:
    """
    Keep EEG channels only and set average reference, as in the ADFTD script.
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
# FFT / DWT / spectral features -- ported VERBATIM from
# tdbrain_features_from_csv.py (per-window/epoch functions, not the old
# whole-recording versions). Operates on one (C, T) window at a time.
# ============================================================

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


# ============================================================
# Windowing + per-file / per-subject aggregation
# ============================================================

def windows_from_raw(raw: mne.io.BaseRaw, win_sec: float) -> Tuple[List[np.ndarray], float]:
    """
    Split one already-preprocessed (resampled, EEG-picked, average-
    referenced) continuous recording into non-overlapping (C, T) windows
    of length win_sec, same convention as tdbrain_features_from_csv.py's
    load_windows_CxT: trailing samples that don't fill a whole window are
    dropped. No boundary-annotation splitting here (AMP-SCZ recordings in
    this snapshot are single continuous files) -- if that's ever not true
    for some subject, windows could span a discontinuity; not currently
    guarded against, same as the TDBrain CSV version.
    """
    data = raw.get_data()  # (C, n_samples), volts
    fs = float(raw.info["sfreq"])
    win_len = int(round(win_sec * fs))

    T = data.shape[1]
    n_windows = T // win_len

    windows: List[np.ndarray] = []
    for i in range(n_windows):
        windows.append(data[:, i * win_len:(i + 1) * win_len])

    return windows, fs


def features_for_file(
    fp: Path, target_fs: float = TARGET_FS, win_sec: float = WIN_SEC, wavelet: str = WAVELET
) -> Dict[str, float]:
    """
    Compute one feature vector from one raw EEG recording: split into
    non-overlapping win_sec windows, run features_for_epoch on each, then
    average across windows -- exactly the tdbrain_features_from_csv.py
    pattern, applied here to an MNE-loaded/resampled/picked recording
    instead of a CSV.
    """
    raw = load_raw(fp, preload=True)

    # ------------------------------------------------------------
    # Drop AMP-SCZ auxiliary/non-scalp channel if present
    # VIS appears only in some subjects and is not part of the common
    # 63-channel EEG montage.
    # This must be done before pick_eeg(), because MNE may classify VIS as EEG.
    # ------------------------------------------------------------
    if "VIS" in raw.ch_names:
        print(f"[INFO] Dropping VIS channel from: {fp}")
        raw.drop_channels(["VIS"])

    raw = ensure_fs(raw, target_fs=target_fs)
    raw = pick_eeg(raw, average_reference=True)

    windows, fs = windows_from_raw(raw, win_sec=win_sec)
    if not windows:
        raise RuntimeError(
            f"Recording shorter than one {win_sec}s window at {fs}Hz -- no windows produced."
        )

    rows = [features_for_epoch(w, fs, wavelet) for w in windows]
    feats = pd.DataFrame(rows).mean(axis=0, numeric_only=True).to_dict()
    return feats


def aggregate_subject_features(
    files: List[Path], subj_id: str, target_fs: float, win_sec: float, wavelet: str
) -> Dict[str, float]:
    """
    Average per-file (already window-averaged) features across multiple
    files/runs for the same subject. No n_files_used/source_files
    bookkeeping columns -- matching the ADFTD/APAVA/TDBrain scripts.
    """
    rows = []

    for fp in files:
        try:
            feats = features_for_file(fp, target_fs=target_fs, win_sec=win_sec, wavelet=wavelet)
            rows.append(feats)
        except Exception as e:
            print(f"[WARN] Skipping file due to error: {fp}")
            print(f"       {e}")

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    mean_feats = df.mean(axis=0, numeric_only=True).to_dict()
    mean_feats["subj_ID"] = subj_id

    return mean_feats


# ============================================================
# Per-subject worker (top-level function -- must stay picklable for
# ProcessPoolExecutor)
# ============================================================

def _process_one_subject(
    subject_dir: Path, target_fs: float, win_sec: float, wavelet: str
) -> Tuple[str, Dict[str, float] | None]:
    """
    Worker unit for one subject: find its EEG files, run
    aggregate_subject_features, return (subj_id, feats_or_None). Printing
    from inside a worker process is fine but can interleave across
    subjects when n_jobs > 1 -- if you need to tell "still running" apart
    from "actually stuck" on a huge run, this is the function to add
    timestamped [START]/[DONE] prints to (same pattern used in timing.py).
    """
    subj_id = subject_dir.name
    eeg_files = find_eeg_files(subject_dir)

    # Prefer .vhdr for BrainVision so the linked .eeg/.vmrk are read correctly.
    vhdr_files = [fp for fp in eeg_files if fp.suffix.lower() == ".vhdr"]
    files_to_process = vhdr_files if vhdr_files else eeg_files

    if not files_to_process:
        print(f"[INFO] No EEG files found for {subj_id}. Skipping.")
        return subj_id, None

    print(f"[PROCESS] {subj_id}  ({len(files_to_process)} file(s))")

    feats = aggregate_subject_features(
        files_to_process, subj_id, target_fs=target_fs, win_sec=win_sec, wavelet=wavelet
    )

    if not feats:
        print(f"[WARN] No valid features computed for {subj_id}.")
        return subj_id, None

    return subj_id, feats


# ============================================================
# Main loop
# ============================================================

def process_all_subjects(
    root_dir: str | Path, output_csv: str | Path,
    target_fs: float = TARGET_FS, win_sec: float = WIN_SEC, wavelet: str = WAVELET,
    n_jobs: int = 1,
) -> pd.DataFrame:
    root = Path(root_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    subject_dirs = discover_subject_dirs(root)

    print(f"[INFO] Root: {root}")
    print(f"[INFO] Found {len(subject_dirs)} subject folders with EEG files.")
    print(f"[INFO] n_jobs={n_jobs}")

    if n_jobs is None or n_jobs <= 1:
        results = [
            _process_one_subject(sd, target_fs, win_sec, wavelet) for sd in subject_dirs
        ]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(_process_one_subject, sd, target_fs, win_sec, wavelet)
                for sd in subject_dirs
            ]
            results = [f.result() for f in futures]

    rows = [feats for _subj_id, feats in results if feats is not None]

    if not rows:
        raise RuntimeError("No features computed for any subject.")

    cols = ["subj_ID"] + FEATURE_ORDER

    df = pd.DataFrame(rows)

    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[cols].sort_values("subj_ID").reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    print("\n[DONE]")
    print(f"[OK] Wrote: {output_csv}")
    print(f"[OK] Shape: {df.shape}")
    print(f"[OK] Feature columns: {len(FEATURE_ORDER)}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute FFT, DWT, and spectral features from AMP-SCZ raw EEG files "
            "using non-overlapping windows, same pattern as "
            "tdbrain_features_from_csv.py. Output is one CSV row per subject."
        )
    )

    parser.add_argument(
        "--root", type=str, default=ROOT_DIR,
        help="Root directory containing AMP-SCZ subject folders, e.g. Eyes_Closed_visit1_only.",
    )
    parser.add_argument("--out", type=str, default=OUTPUT_CSV, help="Output CSV path.")
    parser.add_argument("--fs", type=float, default=TARGET_FS,
                         help="Target sampling rate (Hz) after resampling.")
    parser.add_argument("--win-sec", type=float, default=WIN_SEC,
                         help="Non-overlapping window length in seconds.")
    parser.add_argument("--wavelet", type=str, default=WAVELET,
                         help="Wavelet used for DWT band-energy features (e.g. db4, sym5).")
    parser.add_argument("--n_jobs", type=int, default=8,
                         help="Number of worker processes for subject-level parallelism. "
                              "Set to 1 for fully serial execution. With 661 AMP-SCZ "
                              "subjects, this is the main lever for wall-clock time.")

    args = parser.parse_args()

    process_all_subjects(
        args.root, args.out, target_fs=args.fs, win_sec=args.win_sec, wavelet=args.wavelet,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
