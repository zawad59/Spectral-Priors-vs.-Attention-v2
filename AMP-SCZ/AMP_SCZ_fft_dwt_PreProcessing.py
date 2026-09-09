#!/usr/bin/env python3
# SCZ_PreProcessing_ADFTD_aggregated.py
# Calculation of Spectral Feat

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
import pywt
from scipy.signal import welch
from scipy import interpolate

# np.trapz was removed in numpy>=2.0 (renamed to np.trapezoid). Pre-existing
# issue in this script, unrelated to the resampling/epoching changes below --
# fixed here so the script actually runs.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ============================================================
# Defaults
# ============================================================

ROOT_DIR = r"Eyes_Closed_visit1_only"
OUTPUT_CSV = r"SCZ_FFT_DWT.csv"

TARGET_FS = 256.0
WIN_SEC = 4.0
STRIDE_SEC = 2.0

# Match ADFTD/TDBrain script: keep gamma unless you explicitly remove it.
BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

SUMMARY_FMIN, SUMMARY_FMAX = 0.5, 45.0

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
# Label loading (Label/label.npy -> {subj_id: label})
# ============================================================

def load_labels(label_path: str | Path) -> Dict[str, object]:
    """
    Loads subject-level labels from a .npy file and returns a
    {subj_id: label} dict.

    Accepts, in this order:
      1. A dict (or 0-d object array wrapping a dict) mapping subj_id -> label.
      2. A structured/record numpy array with a subject-id field
         (name containing "subj" or "id") and a label field
         (name containing "label").
      3. A plain 2-column array/sequence of [subj_id, label] pairs.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    raw = np.load(label_path, allow_pickle=True)

    # Case: 0-d object array wrapping a dict (typical of np.save(dict)).
    if isinstance(raw, np.ndarray) and raw.shape == () and isinstance(raw.item(), dict):
        raw = raw.item()

    # Case: already a dict.
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}

    # Case: structured/record array.
    if isinstance(raw, np.ndarray) and raw.dtype.names:
        names = raw.dtype.names
        id_field = next((n for n in names if "subj" in n.lower() or "id" in n.lower()), None)
        label_field = next((n for n in names if "label" in n.lower()), None)
        if id_field is None or label_field is None:
            raise ValueError(
                f"Could not identify subject-id/label fields in structured array "
                f"with fields {names}. Expected a field name containing "
                f"'subj'/'id' and one containing 'label'."
            )
        return {str(row[id_field]): row[label_field] for row in raw}

    # Case: plain 2-column array/sequence of [subj_id, label] pairs.
    arr = np.asarray(raw, dtype=object)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return {str(sid): lab for sid, lab in arr}

    raise ValueError(
        f"Unrecognized label.npy format (type={type(raw)}, "
        f"shape={getattr(raw, 'shape', None)}). Expected a dict, a structured "
        f"array with id/label fields, or a 2-column [subj_id, label] array."
    )


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
    (referencing/channel selection don't depend on sample rate).
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
# Resampling (linear interpolation, matching TDBrain_fft_dwt.py exactly)
# ============================================================
# Replaces MNE's raw.resample() (which applies its own anti-aliasing filter)
# with the SAME plain linear-interpolation resampling used by the
# ADFTD/TDBrain CSV-loading scripts, so all three scripts' resampling
# behavior is consistent. Operated on the extracted (C, T) numpy array
# rather than the MNE Raw object, since interp1d works per-channel.
def resampling(array: np.ndarray, freq: float, target_fs: float, kind: str = "linear") -> np.ndarray:
    t = np.linspace(1, len(array), len(array))
    f = interpolate.interp1d(t, array, kind=kind)
    t_new = np.linspace(1, len(array), int(len(array) / freq * target_fs))
    return f(t_new)


def resample_channels(data_CxT: np.ndarray, native_fs: float, target_fs: float) -> np.ndarray:
    """Resamples every channel of a (C, T) array independently via linear
    interpolation. Returns the array unchanged if native_fs == target_fs."""
    if abs(native_fs - target_fs) <= 1e-6:
        return data_CxT
    resampled = [resampling(ch, freq=native_fs, target_fs=target_fs) for ch in data_CxT]
    return np.stack(resampled, axis=0)  # (C, T_new)


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
            epochs.append(seg.astype(float))
    return epochs if epochs else [data_CxT.astype(float)]


# ============================================================
# FFT / DWT / spectral features (now computed PER EPOCH)
# ============================================================

def bandpower_welch(
    data: np.ndarray,
    fs: float,
    band: Tuple[float, float],
    nperseg: int = 1024,
    noverlap: int = 512,
) -> np.ndarray:
    """
    Absolute bandpower per channel using Welch PSD.
    data shape: channels x samples
    """
    fmin, fmax = band
    pw = []

    for ch in data:
        if len(ch) == 0:
            pw.append(np.nan)
            continue

        nseg = min(nperseg, len(ch))
        nseg = max(nseg, int(fs))
        nseg = min(nseg, len(ch))

        if nseg < 2:
            pw.append(np.nan)
            continue

        freqs, psd = welch(
            ch,
            fs=fs,
            nperseg=nseg,
            noverlap=min(noverlap, nseg // 2),
            window="hann",
        )

        idx = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(idx):
            pw.append(np.nan)
        else:
            pw.append(float(_trapz(psd[idx], freqs[idx])))

    return np.asarray(pw, dtype=float)


def relative_bandpowers_fft(data: np.ndarray, fs: float) -> Dict[str, float]:
    """
    ADFTD-style FFT relative band powers:
    - compute bandpower per channel
    - normalize by sum of all five band powers per channel
    - average relative band powers across channels
    """
    nperseg = int(4 * fs)
    noverlap = nperseg // 2

    abs_powers: Dict[str, np.ndarray] = {}
    for name, band in BANDS.items():
        abs_powers[name] = bandpower_welch(data, fs, band, nperseg=nperseg, noverlap=noverlap)

    total_power_per_ch = np.zeros(data.shape[0], dtype=float)
    for name in BANDS:
        total_power_per_ch += np.nan_to_num(abs_powers[name], nan=0.0)

    rel: Dict[str, float] = {}
    for name in BANDS:
        rp_ch = abs_powers[name] / np.maximum(total_power_per_ch, 1e-12)
        rel[f"FFT_{name}"] = float(np.nanmean(rp_ch))

    return rel


def dwt_band_energies(
    data: np.ndarray,
    fs: float,
    wavelet: str = "db4",
    max_level: int = 7,
) -> Dict[str, float]:
    """
    ADFTD-style DWT relative band energies:
    - compute DWT coefficient energies per channel
    - map wavelet subband frequency ranges into delta/theta/alpha/beta/gamma
    - normalize by total mapped band energy per channel
    - average across channels
    """
    w = pywt.Wavelet(wavelet)
    rel_energies_all_ch = {f"DWT_{k}": [] for k in BANDS.keys()}

    for ch in data:
        if len(ch) < w.dec_len:
            for k in BANDS.keys():
                rel_energies_all_ch[f"DWT_{k}"].append(np.nan)
            continue

        lmax = pywt.dwt_max_level(len(ch), w.dec_len)
        level = min(max_level, max(1, lmax))

        coeffs = pywt.wavedec(ch, wavelet=w, level=level, mode="periodization")
        cA = coeffs[0]
        cDs = coeffs[1:]

        subband_ranges = [(0.0, fs / (2 ** (level + 1)))]
        subband_energies = [float(np.dot(cA, cA))]

        for i, cD in enumerate(cDs):
            level_j = level - i
            f_low = fs / (2 ** (level_j + 1))
            f_high = fs / (2 ** level_j)
            subband_ranges.append((f_low, f_high))
            subband_energies.append(float(np.dot(cD, cD)))

        mapped_E = {k: 0.0 for k in BANDS.keys()}

        for (f_low, f_high), energy in zip(subband_ranges, subband_energies):
            width = f_high - f_low
            if width <= 0:
                continue

            for band_name, (b_min, b_max) in BANDS.items():
                overlap_min = max(f_low, b_min)
                overlap_max = min(f_high, b_max)

                if overlap_max > overlap_min:
                    fraction = (overlap_max - overlap_min) / width
                    mapped_E[band_name] += energy * fraction

        total_E = sum(mapped_E.values()) + 1e-12

        for band_name in BANDS.keys():
            rel_energies_all_ch[f"DWT_{band_name}"].append(mapped_E[band_name] / total_E)

    return {k: float(np.nanmean(v)) for k, v in rel_energies_all_ch.items()}


def spectral_summary_features(data: np.ndarray, fs: float) -> Dict[str, float]:
    """
    ADFTD-style spectral summary features:
    - SPEC_mean_freq
    - SPEC_median_freq / SEF50
    - SPEC_entropy
    Each is computed per channel and averaged across channels.
    """
    nperseg = int(4 * fs)
    noverlap = nperseg // 2

    fmeans, fmeds, ents = [], [], []

    for ch in data:
        if len(ch) == 0:
            fmeans.append(np.nan)
            fmeds.append(np.nan)
            ents.append(np.nan)
            continue

        nseg = min(nperseg, len(ch))
        nseg = max(nseg, int(fs))
        nseg = min(nseg, len(ch))

        if nseg < 2:
            fmeans.append(np.nan)
            fmeds.append(np.nan)
            ents.append(np.nan)
            continue

        freqs, psd = welch(
            ch,
            fs=fs,
            nperseg=nseg,
            noverlap=min(noverlap, nseg // 2),
            window="hann",
        )

        mask = (freqs >= SUMMARY_FMIN) & (freqs <= SUMMARY_FMAX)
        f = freqs[mask]
        P = psd[mask].clip(min=0)

        if f.size == 0 or np.all(P == 0):
            fmeans.append(np.nan)
            fmeds.append(np.nan)
            ents.append(np.nan)
            continue

        p = P / np.sum(P)
        cdf = np.cumsum(p)

        fmeans.append(float(np.sum(f * p)))
        fmeds.append(float(np.interp(0.5, cdf, f)))

        nz = p[p > 0]
        ents.append(float(-np.sum(nz * np.log(nz))))

    return {
        "SPEC_mean_freq": float(np.nanmean(fmeans)),
        "SPEC_median_freq": float(np.nanmean(fmeds)),
        "SPEC_entropy": float(np.nanmean(ents)),
    }


def detrend_dc(epoch_CxT: np.ndarray) -> np.ndarray:
    return epoch_CxT - np.mean(epoch_CxT, axis=1, keepdims=True)


def features_for_epoch(epoch_CxT: np.ndarray, fs: float) -> Dict[str, float]:
    X = detrend_dc(epoch_CxT)
    out: Dict[str, float] = {}
    out.update(relative_bandpowers_fft(X, fs))
    out.update(dwt_band_energies(X, fs, wavelet="db4", max_level=7))
    out.update(spectral_summary_features(X, fs))
    return out


def features_for_file(
    fp: Path, target_fs: float, win_sec: float, stride_sec: float
) -> Tuple[Dict[str, float], int]:
    """
    Loads one raw recording, resamples via linear interpolation, segments
    into sliding-window epochs, computes FFT/DWT/spectral features per
    epoch, and averages across epochs.

    Returns (mean_feature_dict, n_epochs).
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

    raw = pick_eeg(raw, average_reference=True)  # channel selection/referencing at native fs

    data_native = raw.get_data()  # (C, T), MNE units are volts for BrainVision EEG
    native_fs = float(raw.info["sfreq"])

    data = resample_channels(data_native, native_fs=native_fs, target_fs=target_fs)
    fs = target_fs

    epochs = segment_sliding(data, fs=fs, win_sec=win_sec, stride_sec=stride_sec)
    n_epochs = len(epochs)

    rows = [features_for_epoch(ep, fs) for ep in epochs]
    mean_feats = pd.DataFrame(rows).mean(axis=0, numeric_only=True).to_dict()
    return mean_feats, n_epochs


def aggregate_subject_features(
    files: List[Path], subj_id: str, target_fs: float, win_sec: float, stride_sec: float
) -> Dict[str, float]:
    """
    Averages features across multiple files/runs for the same subject, and
    sums n_epochs across those files.
    """
    rows = []
    total_n_epochs = 0

    for fp in files:
        try:
            feats, n_epochs = features_for_file(fp, target_fs=target_fs, win_sec=win_sec, stride_sec=stride_sec)
            feats["source_file"] = str(fp)
            rows.append(feats)
            total_n_epochs += n_epochs
            print(f"[EPOCHS] subj {subj_id}  file={fp.name}  n_epochs={n_epochs}")
        except Exception as e:
            print(f"[WARN] Skipping file due to error: {fp}")
            print(f"       {e}")

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    mean_feats = df.mean(axis=0, numeric_only=True).to_dict()
    mean_feats["subj_ID"] = subj_id
    mean_feats["n_files_used"] = len(rows)
    mean_feats["n_epochs"] = total_n_epochs
    mean_feats["source_files"] = "|".join(df["source_file"].astype(str).tolist())

    return mean_feats


# ============================================================
# Main loop
# ============================================================

def process_all_subjects(
    root_dir: str | Path, output_csv: str | Path,
    target_fs: float, win_sec: float, stride_sec: float,
    label_path: str | Path | None = None,
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

        print(f"\n[PROCESS] {subj_id}")
        print(f"          files: {len(files_to_process)}")

        feats = aggregate_subject_features(
            files_to_process, subj_id, target_fs=target_fs, win_sec=win_sec, stride_sec=stride_sec
        )

        if feats:
            rows.append(feats)
            print(f"[OK] subj {subj_id}  files={feats['n_files_used']}  total_n_epochs={feats['n_epochs']}")
        else:
            print(f"[WARN] No valid features computed for {subj_id}.")

    if not rows:
        raise RuntimeError("No features computed for any subject.")

    feature_cols = (
        [f"FFT_{k}" for k in ["delta", "theta", "alpha", "beta", "gamma"]]
        + [f"DWT_{k}" for k in ["delta", "theta", "alpha", "beta", "gamma"]]
        + ["SPEC_mean_freq", "SPEC_median_freq", "SPEC_entropy"]
    )

    cols = ["subj_ID"] + feature_cols + ["n_epochs", "n_files_used", "source_files"]

    df = pd.DataFrame(rows)

    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[cols].sort_values("subj_ID").reset_index(drop=True)

    # ---- optional: attach subject-specific labels from label.npy ----
    if label_path is not None:
        label_map = load_labels(label_path)
        df["label"] = df["subj_ID"].map(label_map)

        n_missing = int(df["label"].isna().sum())
        if n_missing:
            missing_ids = df.loc[df["label"].isna(), "subj_ID"].tolist()
            print(f"[WARN] {n_missing} subject(s) had no matching entry in {label_path}: {missing_ids}")

        # Keep label immediately after subj_ID, ahead of the feature columns.
        cols = ["subj_ID", "label"] + feature_cols + ["n_epochs", "n_files_used", "source_files"]
        df = df[cols]

    df.to_csv(output_csv, index=False)

    # ---- summary stats across all subjects (same table format as ADFTD) ----
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
    print(f"[OK] Feature columns: {len(feature_cols)}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute ADFTD/TDBrain-style aggregated FFT, DWT, and spectral features "
            "from AMP-SCZ raw EEG files, using linear-interpolation resampling and "
            "sliding-window epoching. Output is one CSV row per subject."
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

    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Path to a label .npy file (e.g. Label/label.npy) mapping subject IDs to "
             "diagnostic labels. If provided, a 'label' column is added to the output CSV.",
    )

    args = parser.parse_args()

    process_all_subjects(
        args.root, args.out,
        target_fs=args.fs, win_sec=args.win_sec, stride_sec=args.stride_sec,
        label_path=args.label,
    )


if __name__ == "__main__":
    main()
