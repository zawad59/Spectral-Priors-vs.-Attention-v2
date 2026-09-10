from __future__ import annotations
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import welch, resample as scipy_resample
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


def _norm_sid(x) -> str:
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    if m:
        return m.group(1).lstrip("0") or "0"
    return s


# ---------------------- feature extraction (unchanged math -- identical to ADFTD) ----------------------
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


# ---------------------- loading preproctrialsNN.mat files ----------------------
def find_subject_files(root: Path) -> List[Path]:
    """preproctrials*.mat files directly under root, sorted by the numeric
    index embedded in the filename (preproctrials01.mat, preproctrials02.mat, ...)."""
    files = [p for p in root.glob("preproctrials*.mat") if p.is_file()]

    def _num(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else -1

    return sorted(files, key=_num)


def _iter_trials(trial_field) -> List[np.ndarray]:
    """
    data.trial normally loads (via scipy.io.loadmat with squeeze_me=True,
    struct_as_record=False) as a 1D object array of (n_channels,
    n_samples) arrays -- one per trial. Defensively also handle the
    edge case of a file with exactly one trial, where scipy's squeeze can
    collapse that array-of-one down to a single bare 2D array instead.
    """
    if isinstance(trial_field, np.ndarray) and trial_field.dtype == object:
        return list(trial_field)
    # single-trial edge case: a bare (C, T) array instead of an array of one
    arr = np.asarray(trial_field)
    if arr.ndim == 2:
        return [arr]
    raise ValueError(f"Unexpected data.trial shape/dtype: {arr.shape}, {arr.dtype}")


def load_windows_CxT(fp: Path, win_sec: float, fs_override: float | None) -> Tuple[List[np.ndarray], float]:
    """
    Load a FieldTrip-style preproctrialsNN.mat file and split EACH of its
    already-segmented trials into non-overlapping (C, T) windows of length
    win_sec (trailing samples in a trial too short for one more window are
    dropped, same as ADFTD's segment handling -- a trial here plays the
    same role a contiguous segment played there). Returns
    (list_of_windows_across_all_trials, fs_used).
    """
    mat = sio.loadmat(str(fp), squeeze_me=True, struct_as_record=False)
    data = mat["data"]
    native_fs = float(data.fsample)
    trials = _iter_trials(data.trial)

    fs = fs_override if fs_override else native_fs
    resample_needed = bool(fs_override) and abs(fs_override - native_fs) > 1e-6

    win_len = int(round(win_sec * fs))

    windows: List[np.ndarray] = []
    for trial in trials:
        trial = np.asarray(trial, dtype=float)  # (C, T)
        if resample_needed:
            n_new = int(round(trial.shape[1] * fs_override / native_fs))
            trial = scipy_resample(trial, n_new, axis=1)

        T = trial.shape[1]
        n_windows = T // win_len
        for i in range(n_windows):
            windows.append(trial[:, i * win_len:(i + 1) * win_len])

    return windows, fs


def features_for_subject(
    fp: Path, win_sec: float, fs_override: float | None, wavelet: str
) -> Tuple[Dict[str, float], float, int]:
    windows, fs = load_windows_CxT(fp, win_sec, fs_override)
    rows = [features_for_epoch(w, fs, wavelet) for w in windows]
    feats = pd.DataFrame(rows).mean(axis=0, numeric_only=True).to_dict()
    return feats, fs, len(windows)


def main():
    ap = ArgumentParser(
        description="Run FFT/DWT/spectral-summary feature extraction directly on "
                    "APAVA preproctrialsNN.mat files (FieldTrip-style structs with "
                    "already-segmented trials), using 1s non-overlapping windows."
    )
    ap.add_argument("--root", type=str, required=True,
                     help="Path to the directory containing preproctrialsNN.mat files")
    ap.add_argument("--out", type=str, default="apava_fft_dwt.csv", help="Output CSV path")
    ap.add_argument(
        "--labels", type=str, required=True,
        help="Path to label.npy OR a folder containing label.npy (shape (N,2): [label, subj_ID])."
    )
    ap.add_argument("--win-sec", type=float, default=1.0,
                     help="Non-overlapping window length in seconds.")
    ap.add_argument("--fs", type=float, default=256.0,
                     help="Force this sampling rate (resamples each file's trials if its "
                          "native fsample differs). Default 256.0 is a no-op for data "
                          "already at 256Hz.")
    ap.add_argument("--wavelet", type=str, default="db4", help="Wavelet (e.g. db4, sym5).")
    args = ap.parse_args()

    root = Path(args.root)
    sub_files = find_subject_files(root)

    print(f"[DEBUG] root = {root.resolve()}")
    if not sub_files:
        print(f"[ERROR] No preproctrials*.mat files found under {root}")
        return
    print(f"[INFO] Found {len(sub_files)} subject file(s) under {root}")
    print(f"[DEBUG] {sub_files[0].name}  (first file to be loaded to confirm channels/fs)")

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
            f"these should match 1:1 by preproctrialsNN numeric order if the row-order "
            f"assumption holds."
        )

    n_pairs = min(len(sub_files), labels_arr.shape[0])

    print("\n[VERIFY] First 5 and last 5 (subject file -> assumed label.npy row) -- "
          "check this against what you know about the file<->subj_ID mapping before trusting the output:")
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
            feats, fs_used, n_windows = features_for_subject(
                fp, win_sec=args.win_sec, fs_override=args.fs, wavelet=args.wavelet
            )
        except Exception as e:
            print(f"[WARN] Skipping {fp.name}: {e}")
            continue

        row = {k: feats.get(k, np.nan) for k in FEATURE_ORDER}
        row["subj_ID"] = _norm_sid(sid)

        lab_s = str(lab).strip()
        row["Group"] = int(lab_s) if lab_s.lstrip("-").isdigit() else lab

        rows.append(row)
        print(f"[OK] {fp.name}  fs={fs_used:.1f}Hz  n_windows={n_windows}  ->  subj_ID={row['subj_ID']}")

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
