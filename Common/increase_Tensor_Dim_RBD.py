#!/usr/bin/env python3
"""
Convert Feature_*.npy tensors from (N_win, samples, C) into:
  - (N_win, samples, C, 5)  [default], where 5 = {delta,theta,alpha,beta,gamma}
or
  - (N_win, samples, C*5)   [--flatten]

Filtering is applied along the 'samples' axis for every (window, channel).

"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from scipy.signal import butter, sosfiltfilt, sosfilt

CANONICAL_BANDS = [
    ("band1", 101.0, 103.0),
    ("band2", 104.0, 108.0),
    ("band3", 108.0, 112.0),
    ("band4", 115.0, 120.0),
    ("band5", 120.0, 127.0),
]
BANDS = CANONICAL_BANDS  # kept for backward compatibility

def random_bands(fs: float, n_bands: int = 5, seed: int | None = None,
                  band_width: float = 10.0, margin: float = 0.5):
    """
    Generate n_bands non-overlapping passbands with random center placement
    (fixed width, random position) spanning (margin, nyquist - margin).

    Returns a list of (name, lo, hi) tuples in the same shape as
    CANONICAL_BANDS, so it drops straight into to_5band_tensor. Band names
    embed their actual edges (e.g. "rand1_37.2-47.2Hz") so the specific
    random draw is always recoverable from the name alone, in addition to
    whatever is written to band_edges.json.

    A fixed seed makes the draw reproducible across the random-band run and
    across repeated invocations meant to compare against the same draw.
    """
    rng = np.random.default_rng(seed)
    nyq = 0.5 * fs
    lo_bound, hi_bound = margin, nyq - margin
    span = hi_bound - lo_bound
    total_width = n_bands * band_width
    if total_width >= span:
        raise ValueError(
            f"Cannot fit {n_bands} bands of width {band_width} Hz "
            f"in (0, nyquist)=({0}, {nyq}) for fs={fs}."
        )

    # Randomly distribute the leftover span as n_bands+1 gaps (before,
    # between, and after the bands) via a Dirichlet split, then lay the
    # fixed-width bands out left-to-right using those gaps.
    slack = span - total_width
    gaps = rng.dirichlet(np.ones(n_bands + 1)) * slack

    bands = []
    cursor = lo_bound + gaps[0]
    for i in range(n_bands):
        lo = cursor
        hi = lo + band_width
        bands.append((f"rand{i + 1}_{lo:.1f}-{hi:.1f}Hz", lo, hi))
        cursor = hi + gaps[i + 1]
    return bands

def validate_bands_for_fs(bands, fs: float, margin: float = 0.0):
    """Raise a clear error up front if any band's upper edge doesn't fit
    under Nyquist for the given fs, instead of failing deep inside
    bandpass_sos on whichever file happens to be processed first."""
    nyq = 0.5 * fs
    bad = [(n, lo, hi) for n, lo, hi in bands if hi >= nyq - margin]
    if bad:
        bad_str = "; ".join(f"{n} [{lo}, {hi}] Hz" for n, lo, hi in bad)
        raise ValueError(
            f"{len(bad)} band(s) exceed Nyquist ({nyq} Hz) for fs={fs}: {bad_str}. "
            f"Either lower these band edges, or use --fs matching the sampling rate "
            f"these bands were actually designed for."
        )


def bandpass_sos(fs: float, lo: float, hi: float, order: int = 4):
    nyq = 0.5 * fs
    if lo <= 0:
        lo = 0.001
    if hi >= nyq:
        hi = nyq - 0.001
    if not (0 < lo < hi < nyq):
        raise ValueError(f"Bad band [{lo}, {hi}] for fs={fs} (nyq={nyq}).")
    return butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")

def safe_sosfiltfilt(sos, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    sosfiltfilt can fail if the signal is too short for padding.
    """
    T = x.shape[axis]
    # conservative padlen; must be < T
    padlen = min(max(0, 3 * (2 * sos.shape[0])), max(0, T - 1))
    if T < 8 or padlen <= 0:
        return sosfilt(sos, x, axis=axis)
    try:
        return sosfiltfilt(sos, x, axis=axis, padlen=padlen)
    except ValueError:
        # last-resort fallback
        return sosfilt(sos, x, axis=axis)

def to_5band_tensor(arr: np.ndarray, fs: float, order: int = 4, bands=None) -> np.ndarray:
    """
    arr: (N_win, samples, C)
    out: (N_win, samples, C, len(bands))  with band order matching `bands`

    `bands` defaults to CANONICAL_BANDS if not supplied, but can be any
    list of (name, lo, hi) tuples -- e.g. the output of random_bands().
    """
    if bands is None:
        bands = CANONICAL_BANDS
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D (N_win, samples, C). Got {arr.shape}")
    n_win, T, C = arr.shape

    # reshape to (N_win*C, T) so we can filter all channels/windows in one go
    X = np.transpose(arr, (0, 2, 1)).reshape(n_win * C, T).astype(np.float32, copy=False)

    band_out = []
    for _, lo, hi in bands:
        sos = bandpass_sos(fs, lo, hi, order=order)
        Y = safe_sosfiltfilt(sos, X, axis=1)  # filter along time axis
        # back to (N_win, T, C)
        Y = Y.reshape(n_win, C, T).transpose(0, 2, 1)
        band_out.append(Y)

    # stack along new band dimension -> (N_win, T, C, len(bands))
    return np.stack(band_out, axis=-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder containing Feature_*.npy")
    ap.add_argument("--out_dir", required=True, help="Where to write converted .npy files")
    ap.add_argument("--pattern", default="Feature_*.npy", help="Glob pattern (default: Feature_*.npy)")
    ap.add_argument("--fs", type=float, required=True, help="Sampling rate in Hz (e.g., 256)")
    ap.add_argument("--order", type=int, default=4, help="Butterworth order (default: 4)")
    ap.add_argument("--flatten", action="store_true",
                    help="If set, output shape is (N_win, samples, C*5) instead of (N_win, samples, C, 5)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--overwrite", action="store_true",
                    help="If set, write into --in_dir instead of --out_dir (overwrites files).")
    ap.add_argument("--band_mode", default="canonical", choices=["canonical", "random"],
                    help="'canonical' (default) uses the fixed delta/theta/alpha/beta/gamma bands. "
                         "'random' draws --n_bands non-overlapping random passbands instead, for "
                         "an ablation against the canonical-band experiment.")
    ap.add_argument("--n_bands", type=int, default=5,
                    help="Number of bands to use in --band_mode random (default: 5, to match "
                         "the canonical 5-band setup). Ignored in --band_mode canonical.")
    ap.add_argument("--random_band_width", type=float, default=10.0,
                    help="Width in Hz of each random band (default: 10.0). Ignored in "
                         "--band_mode canonical.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for --band_mode random, so the same random bands can be "
                         "reproduced across runs. Ignored in --band_mode canonical.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = in_dir if args.overwrite else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {args.pattern!r} in {in_dir}")

    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    if args.band_mode == "random":
        bands = random_bands(
            fs=args.fs, n_bands=args.n_bands, seed=args.seed,
            band_width=args.random_band_width,
        )
    else:
        bands = CANONICAL_BANDS
    band_names = [b[0] for b in bands]
    validate_bands_for_fs(bands, args.fs)

    # Record exactly which bands were used -- essential for the random mode,
    # since the draw differs run to run unless --seed is fixed, and this is
    # what you'd cite when comparing the random-band experiment's results
    # against the canonical-band experiment's.
    with open(out_dir / "band_edges.json", "w") as f:
        json.dump(
            {
                "band_mode": args.band_mode,
                "fs": args.fs,
                "seed": args.seed,
                "bands": [{"name": n, "lo_hz": lo, "hi_hz": hi} for n, lo, hi in bands],
            },
            f, indent=2,
        )

    for p in files:
        arr = np.load(p, allow_pickle=False)

        out = to_5band_tensor(arr, fs=args.fs, order=args.order, bands=bands).astype(out_dtype, copy=False)

        if args.flatten:
            # (N_win, T, C, len(bands)) -> (N_win, T, C*len(bands))
            # ordering: for each channel, bands are contiguous: [ch0_band1,...,ch0_bandN, ch1_band1,...]
            n_win, T, C, B = out.shape
            out = out.reshape(n_win, T, C * B)

        out_path = out_dir / p.name
        np.save(out_path, out)

        if args.verbose:
            print(f"{p.name}: {arr.shape} -> {out.shape} | bands={band_names}")

    print(f"[OK] Processed {len(files)} files.")
    print(f"[OK] Output dir: {out_dir}")
    print(f"[OK] Output shape: {'(N_win, samples, C*B)' if args.flatten else '(N_win, samples, C, B)'} "
          f"where B={len(bands)}")
    print(f"[OK] Band mode: {args.band_mode} | bands={band_names}")

if __name__ == "__main__":
    main()