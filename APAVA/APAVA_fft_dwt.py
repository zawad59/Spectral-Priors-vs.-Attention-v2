from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.io import loadmat
import pywt

# Optional: MNE for EDF/BrainVision/EEGLAB/FIF
try:
    import mne  # type: ignore
    _HAS_MNE = True
except Exception:
    _HAS_MNE = False

# ---------------------- configuration ----------------------
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
SUMMARY_FMIN, SUMMARY_FMAX = 0.5, 45.0
EEG_EXTS = (".vhdr", ".edf", ".set", ".fif")
NUMPY_EXTS = (".npy", ".npz")
MAT_EXTS = (".mat",)

# ---------------------- Welch helpers ----------------------
def _welch_band_power(x: np.ndarray, fs: float, band: Tuple[float, float]) -> float:
    # Absolute band power via Welch
    nperseg = max(int(min(4 * fs, len(x))), int(fs))   # ~4s or <= len(x); at least 1s
    noverlap = min(nperseg // 2, max(nperseg // 4, 0))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann")
    idx = (f >= band[0]) & (f <= band[1])
    return float(np.trapezoid(Pxx[idx], f[idx]))

def relative_bandpowers_fft(epoch_CxT: np.ndarray, fs: float) -> Dict[str, float]:
    # Channel-averaged relative FFT bandpowers from a (C, T) epoch.
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
    """Mean freq, median freq (SEF50), spectral entropy (Shannon), channel-averaged."""
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
        ents.append(float(-np.sum(nz * np.log(nz))))  # nats

    return {
        "SPEC_mean_freq": float(np.nanmean(means)),
        "SPEC_median_freq": float(np.nanmean(medians)),
        "SPEC_entropy": float(np.nanmean(ents)),
    }

# ---------------------- DWT (robust overlap mapping) ----------------------
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
        E_levels = [float(np.dot(c, c)) for c in cDs]  # [E_D_L, ..., E_D_1]

        # D_j ~ (fs/2^(j+1), fs/2^j)
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

        # Add A_L (0..fs/2^(L+1)) partly into delta
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

# ---------------------- Epoch normalization ----------------------
def _as_epochs_CxT(arr: np.ndarray) -> List[np.ndarray]:
    """
    Normalize any array into list of epochs shaped (C, T):
      (T, C) -> [(C, T)] ; (C, T) -> [(C, T)]
      (N, T, C) or (N, C, T) -> list of (C, T)
      1x object dict with 'X'/'data'/'signals'/'eeg' -> recurse
    """
    if isinstance(arr, np.ndarray):
        if arr.ndim == 2:
            T, C = arr.shape
            return [arr.T.astype(float)] if T >= C else [arr.astype(float)]
        if arr.ndim == 3:
            if arr.shape[-1] <= 512:   # (N, T, C)
                return [ep.T.astype(float) for ep in arr]
            return [ep.astype(float) for ep in arr]  # (N, C, T)
        if arr.ndim == 1 and arr.dtype == object and arr.size == 1 and isinstance(arr.item(), dict):
            obj = arr.item()
            for k in ("X", "data", "signals", "eeg"):
                if k in obj and isinstance(obj[k], np.ndarray):
                    return _as_epochs_CxT(obj[k])
    raise ValueError(f"Unsupported array shape/content: {type(arr)} shape={getattr(arr,'shape',None)}")

def detrend_dc(epoch_CxT: np.ndarray) -> np.ndarray:
    return epoch_CxT - np.mean(epoch_CxT, axis=1, keepdims=True)

# ---------------------- Subject ID ----------------------
SUB_RE = re.compile(r"sub-(\d+)")
def subject_id_from_path(p: Path) -> str:
    for parent in [p, *p.parents]:
        m = SUB_RE.search(parent.name)
        if m:
            return m.group(1).lstrip("0") or "0"
    m = re.search(r"(\d+)", p.stem)
    return (m.group(1).lstrip("0") or "0") if m else p.stem

def _norm_sid(x) -> str:
    """
    Make subject IDs comparable across sources:
      'sub-001' / '001' / 1 / 1.0 -> '1'
    """
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    if m:
        return m.group(1).lstrip("0") or "0"
    return s

# --------------- Load Numpy (C*T) -------------------------
def load_np_any(fp: Path) -> List[np.ndarray]:
    if fp.suffix.lower() == ".npy":
        arr = np.load(fp, allow_pickle=True)
        return _as_epochs_CxT(arr)
    z = np.load(fp, allow_pickle=True)  # .npz
    for key in ("X", "data", "signals", "eeg"):
        if key in z.files:
            return _as_epochs_CxT(z[key])
    return _as_epochs_CxT(z[z.files[0]])

# --------------- Load MATLAB .mat (C*T) --------------------
def _extract_epochs_from_matvar(val) -> List[np.ndarray]:
    """
    Recursively pull numeric epochs (C, T) out of a loaded MATLAB variable.
    Handles:
      - FieldTrip-style structs with a 'trial' field (cell array of
        channel x time matrices, e.g. fsample/sampleinfo/trial/time/label/cfg)
      - Struct arrays / object arrays (recurse into each element)
      - Plain numeric arrays (routed through the existing _as_epochs_CxT)
    """
    # Object array: could be a cell array of trials, or an array of structs
    if isinstance(val, np.ndarray) and val.dtype == object:
        collected: List[np.ndarray] = []
        for item in np.atleast_1d(val).ravel():
            collected.extend(_extract_epochs_from_matvar(item))
        return collected

    # FieldTrip-style struct: pull out the 'trial' cell array
    if hasattr(val, "trial"):
        trial = val.trial
        raw_epochs: List[np.ndarray] = []
        if isinstance(trial, np.ndarray) and trial.dtype == object:
            for t in np.atleast_1d(trial).ravel():
                raw_epochs.append(np.asarray(t, dtype=float))
        else:
            raw_epochs.append(np.asarray(trial, dtype=float))
        epochs: List[np.ndarray] = []
        for ep in raw_epochs:
            epochs.extend(_as_epochs_CxT(ep))
        return epochs

    # Plain numeric array
    if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number) and val.size > 0:
        return _as_epochs_CxT(val.astype(float))

    return []

def load_mat_any(fp: Path) -> List[np.ndarray]:
    """
    Load a MATLAB .mat file and normalize its contents into epochs (C, T).
    Supports classic (v4-v7) .mat via scipy.io.loadmat -- including
    FieldTrip-style structs (fsample/sampleinfo/trial/time/label/cfg), where
    'trial' is a cell array of channel x time matrices -- and falls back to
    h5py for MATLAB v7.3 (HDF5-based) .mat files.
    """
    try:
        mat = loadmat(fp, struct_as_record=False, squeeze_me=True)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if not keys:
            raise ValueError(f"No usable variable found in {fp}")
        preferred = [k for k in ("data", "X", "signals", "eeg", "EEG") if k in mat]
        ordered_keys = preferred + [k for k in keys if k not in preferred]

        for key in ordered_keys:
            epochs = _extract_epochs_from_matvar(mat[key])
            if epochs:
                return epochs
        raise ValueError(f"No usable EEG data found in {fp} (checked keys: {keys})")

    except NotImplementedError:
        # v7.3 .mat files are HDF5; scipy.io.loadmat can't read these.
        import h5py

        def _deref(f: "h5py.File", obj):
            if isinstance(obj, h5py.Reference):
                obj = f[obj]
            if isinstance(obj, h5py.Dataset):
                return np.asarray(obj)
            return obj

        with h5py.File(fp, "r") as f:
            keys = [k for k in f.keys() if not k.startswith("__")]
            if not keys:
                raise ValueError(f"No usable variable found in {fp}")
            preferred = [k for k in ("data", "X", "signals", "eeg", "EEG") if k in f]
            ordered_keys = preferred + [k for k in keys if k not in preferred]

            for key in ordered_keys:
                grp = f[key]
                # FieldTrip struct stored as an HDF5 group with a 'trial' entry
                if hasattr(grp, "keys") and "trial" in grp.keys():
                    trial_ref = grp["trial"]
                    raw_epochs: List[np.ndarray] = []
                    arr = np.asarray(trial_ref)
                    if arr.dtype == h5py.ref_dtype or arr.dtype == object:
                        for ref in arr.ravel():
                            raw_epochs.append(np.asarray(_deref(f, ref), dtype=float))
                    else:
                        raw_epochs.append(arr.astype(float))
                    epochs: List[np.ndarray] = []
                    for ep in raw_epochs:
                        epochs.extend(_as_epochs_CxT(ep))
                    if epochs:
                        return epochs
                elif isinstance(grp, h5py.Dataset):
                    arr = np.asarray(grp)
                    if np.issubdtype(arr.dtype, np.number) and arr.size > 0:
                        return _as_epochs_CxT(arr.astype(float))
            raise ValueError(f"No usable EEG data found in {fp} (checked keys: {keys})")

def load_eeg_mne(fp: Path, fs: float, win_sec: float, stride_sec: float) -> List[np.ndarray]:
    if not _HAS_MNE:
        raise RuntimeError("mne is not installed; cannot read EEG files.")
    if fp.suffix.lower() == ".edf":
        raw = mne.io.read_raw_edf(fp, preload=True, verbose="ERROR")
    elif fp.suffix.lower() == ".vhdr":
        raw = mne.io.read_raw_brainvision(fp, preload=True, verbose="ERROR")
    elif fp.suffix.lower() == ".set":
        raw = mne.io.read_raw_eeglab(fp, preload=True, verbose="ERROR")
    elif fp.suffix.lower() == ".fif":
        raw = mne.io.read_raw_fif(fp, preload=True, verbose="ERROR")
    else:
        raise ValueError(f"Unsupported EEG file: {fp}")

    picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    raw.pick(picks)
    raw.set_eeg_reference("average", projection=False)
    if abs(raw.info["sfreq"] - fs) > 1e-6:
        raw.resample(fs)

    X = raw.get_data()  # (C, T)
    win = int(win_sec * fs)
    hop = int(stride_sec * fs)
    T = X.shape[1]
    epochs: List[np.ndarray] = []
    for start in range(0, max(T - win + 1, 1), max(hop, 1)):
        seg = X[:, start:start + win]
        if seg.shape[1] >= max(int(fs), 32):
            epochs.append(seg.astype(float))
    return epochs if epochs else [X.astype(float)]

# ---------------------- Feature extraction ----------------------
def features_for_epoch(epoch_CxT: np.ndarray, fs: float, wavelet: str) -> Dict[str, float]:
    X = detrend_dc(epoch_CxT)
    out: Dict[str, float] = {}
    out.update(relative_bandpowers_fft(X, fs))
    out.update(dwt_band_energies_overlap(X, fs, wavelet=wavelet))
    out.update(spectral_summary_features(X, fs))
    return out

def features_for_file(fp: Path, fs: float, wavelet: str, win_sec: float, stride_sec: float) -> Tuple[Dict[str, float], int]:
    """
    Returns (mean_feature_dict, n_epochs) for this file -- n_epochs is the
    number of epochs/segments this file was cut into before averaging.
    """
    if fp.suffix.lower() in NUMPY_EXTS:
        epochs = load_np_any(fp)
    elif fp.suffix.lower() in MAT_EXTS:
        epochs = load_mat_any(fp)
    elif fp.suffix.lower() in EEG_EXTS:
        epochs = load_eeg_mne(fp, fs=fs, win_sec=win_sec, stride_sec=stride_sec)
    else:
        raise ValueError(f"Unsupported file type: {fp.suffix}")
    n_epochs = len(epochs)
    rows = [features_for_epoch(ep, fs, wavelet) for ep in epochs]
    mean_feats = pd.DataFrame(rows).mean(axis=0, numeric_only=True).to_dict()
    return mean_feats, n_epochs

# ---------------------- Labels loader ----------------------
def load_group_labels_for_subjects(label_fp: Path, subj_ids_sorted: List[str]) -> np.ndarray:

    if label_fp.suffix.lower() in MAT_EXTS:
        try:
            mat = loadmat(label_fp, struct_as_record=False, squeeze_me=True)
            keys = [k for k in mat.keys() if not k.startswith("__")]
            for key in ("group", "Group", "label", "labels", "y"):
                if key in mat:
                    labels_arr = np.asarray(mat[key])
                    break
            else:
                if not keys:
                    raise ValueError(f"No usable variable found in {label_fp}")
                labels_arr = np.asarray(mat[keys[0]])
        except NotImplementedError:
            import h5py
            with h5py.File(label_fp, "r") as f:
                keys = [k for k in f.keys() if not k.startswith("__")]
                for key in ("group", "Group", "label", "labels", "y"):
                    if key in f:
                        labels_arr = np.asarray(f[key])
                        break
                else:
                    if not keys:
                        raise ValueError(f"No usable variable found in {label_fp}")
                    labels_arr = np.asarray(f[keys[0]])
        labels_arr = np.squeeze(labels_arr)
        if labels_arr.ndim == 0:
            labels_arr = labels_arr.reshape(1)
    else:
        labels = np.load(label_fp, allow_pickle=True)
        labels_arr = np.asarray(labels)

    # (N,) already
    if labels_arr.ndim == 1:
        group = labels_arr

    # (N,1) -> (N,)
    elif labels_arr.ndim == 2 and labels_arr.shape[1] == 1:
        group = labels_arr[:, 0]

    # (N,2)
    elif labels_arr.ndim == 2 and labels_arr.shape[1] == 2:

        unique_vals = set(np.unique(labels_arr).tolist())
        is_binary = unique_vals.issubset({0, 1})
        row_sums = labels_arr.sum(axis=1)

        if is_binary and np.all((row_sums == 1) | (row_sums == 0)):
            group = labels_arr.argmax(axis=1)
        else:
  
            lab_col = labels_arr[:, 0]
            sid_col = labels_arr[:, 1]

            sid_to_lab: Dict[str, object] = {}
            for lab, sid in zip(lab_col, sid_col):
                sidn = _norm_sid(sid)
                lab_s = str(lab).strip()
                lab_v: object = int(lab_s) if lab_s.lstrip("-").isdigit() else lab
                sid_to_lab[sidn] = lab_v

            subj_norm = [_norm_sid(s) for s in subj_ids_sorted]
            missing = [s for s in subj_norm if s not in sid_to_lab]
            if missing:
                example_keys = list(sid_to_lab.keys())[:8]
                raise ValueError(
                    f"label file looks like (label,subj_ID) pairs but missing {len(missing)} IDs. "
                    f"Example missing: {missing[:5]}. Example label IDs: {example_keys}"
                )

            group = np.array([sid_to_lab[s] for s in subj_norm], dtype=object)

    else:
        raise ValueError(f"Unsupported label shape: {labels_arr.shape}")

    return np.asarray(group)

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser(description="EEG features (FFT bandpowers, robust DWT energies, spectral summaries).")
    ap.add_argument("--in", dest="in_path", required=True, help="File or folder (dataset root OK)")
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument("--fs", type=float, default=256.0, help="Sampling rate to assume/resample to (Hz).")
    ap.add_argument("--wavelet", type=str, default="db4", help="Wavelet (e.g., db4, sym5).")
    ap.add_argument("--win-sec", type=float, default=1.0, help="Window length (s) for raw EEG files.")
    ap.add_argument("--stride-sec", type=float, default=1.0, help="Stride (s) for raw EEG files.")
    ap.add_argument("--recursive", action="store_true", help="Search subfolders recursively.")
    ap.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to label.npy/label.mat OR a folder containing one (e.g., .../Label or .../Label/label.mat).",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    files: List[Path] = []
    if in_path.is_dir():
        walker = in_path.rglob if args.recursive else in_path.glob
        for ext in (*NUMPY_EXTS, *MAT_EXTS, *EEG_EXTS):
            files.extend(sorted(walker(f"*{ext}")))
    elif in_path.is_file():
        files = [in_path]
    else:
        raise FileNotFoundError(in_path)

    if not files:
        print("[ERROR] No supported files found.")
        return

    # group files by detected subject id (prefers nearest sub-### ancestor)
    by_subj: DefaultDict[str, List[Path]] = defaultdict(list)
    for fp in files:
        sid = subject_id_from_path(fp)
        by_subj[sid].append(fp)

    subject_rows = []
    subject_epoch_counts: Dict[str, int] = {}  # sid -> total n_epochs across that subject's files

    # sort numeric subject IDs numerically, others lexicographically
    def _sort_key(pair):
        sid = pair[0]
        return (0, int(sid)) if sid.isdigit() else (1, sid)

    for sid, fps in sorted(by_subj.items(), key=_sort_key):
        per_file = []
        n_epochs_this_subject = 0
        for fp in fps:
            try:
                feats, n_epochs = features_for_file(
                    fp, fs=args.fs, wavelet=args.wavelet,
                    win_sec=args.win_sec, stride_sec=args.stride_sec
                )
                per_file.append(feats)
                n_epochs_this_subject += n_epochs
                print(f"[EPOCHS] subj {sid}  file={fp.name}  n_epochs={n_epochs}")
            except Exception as e:
                print(f"[WARN] {sid}: skip {fp.name}: {e}")
        if not per_file:
            continue
        df = pd.DataFrame(per_file)
        feats = df.mean(axis=0, numeric_only=True).to_dict()
        feats["subj_ID"] = sid
        feats["n_epochs"] = n_epochs_this_subject
        subject_rows.append(feats)
        subject_epoch_counts[sid] = n_epochs_this_subject
        print(f"[OK] subj {sid}  files={len(fps)}  total_n_epochs={n_epochs_this_subject}")

    if not subject_rows:
        print("[ERROR] No features computed.")
        return

    # ---- summary stats across all subjects ----
    counts = np.array(list(subject_epoch_counts.values()), dtype=float)
    print("\n[SUMMARY] n_epochs per subject:")
    print(f"  n_subjects = {len(counts)}")
    print(f"  mean       = {counts.mean():.2f}")
    print(f"  median     = {np.median(counts):.2f}")
    print(f"  min        = {counts.min():.0f}")
    print(f"  max        = {counts.max():.0f}")
    print(f"  std        = {counts.std(ddof=1):.2f}" if len(counts) > 1 else "  std        = n/a (only 1 subject)")

    cols = (
        ["subj_ID"] +
        [f"FFT_{k}" for k in ["delta", "theta", "alpha", "beta", "gamma"]] +
        [f"DWT_{k}" for k in ["delta", "theta", "alpha", "beta", "gamma"]] +
        ["SPEC_mean_freq", "SPEC_median_freq", "SPEC_entropy"] +
        ["n_epochs"]
    )

    df_all = pd.DataFrame(subject_rows)
    for c in cols:
        if c not in df_all.columns:
            df_all[c] = np.nan

    # First sort by subj_ID (NUMERIC ascending when possible)
    df_all = df_all[cols].copy()
    df_all["_sid_num"] = df_all["subj_ID"].astype(str).map(_norm_sid).astype(int)
    df_all = df_all.sort_values(["_sid_num", "subj_ID"], kind="mergesort").drop(columns=["_sid_num"]).reset_index(drop=True)

    # Resolve label path
    if args.labels is not None:
        label_base = Path(args.labels)
        if label_base.is_dir():
            # prefer .mat, fall back to .npy
            cand_mat = label_base / "label.mat"
            cand_npy = label_base / "label.npy"
            label_fp = cand_mat if cand_mat.exists() else cand_npy
        else:
            label_fp = label_base
    else:
        # default behavior if --labels not provided: prefer .mat, fall back to .npy
        label_dir = (in_path / "Label") if in_path.is_dir() else (in_path.parent / "Label")
        cand_mat = label_dir / "label.mat"
        cand_npy = label_dir / "label.npy"
        label_fp = cand_mat if cand_mat.exists() else cand_npy

    if not label_fp.exists():
        raise FileNotFoundError(f"Label file not found: {label_fp}")

    # Load group labels aligned to sorted subject IDs
    subj_ids_sorted = df_all["subj_ID"].astype(str).tolist()
    group = load_group_labels_for_subjects(label_fp, subj_ids_sorted)

    if len(group) != len(df_all):
        raise ValueError(f"Label vector length {len(group)} != number of subjects {len(df_all)}")

    df_all["Group"] = group
    final_cols = cols + ["Group"]
    df_all = df_all[final_cols]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(args.out, index=False)
    print(f"[OK] Wrote {args.out} with shape {df_all.shape}")

if __name__ == "__main__":
    main()