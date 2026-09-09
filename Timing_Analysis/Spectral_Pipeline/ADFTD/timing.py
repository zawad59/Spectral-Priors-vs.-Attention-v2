#!/usr/bin/env python3
"""
Benchmark_ADFTD_Pipeline_Runtime.py

Times the ADFTD spectral-prior pipeline as ONE simple, linear process:

  1. Compute spectral features for every subject, write ONE CSV.
     (adftd_from_derivatives.py -- unmodified.)
  2. Run gen_Stratified_Sets.py (unmodified) on that CSV to produce the
     stratified_holdout_splits/split_1 ... split_5 directories
     (each with train.csv/val.csv/test.csv + subject-ID files).
  3. Run RDA (RDA_Test.py -- unmodified) on every one of those splits.
     Each call already aggregates across the 5 seeds baked into
     RDA_Test.py's own defaults ([41, 42, 43, 44, 45]), and internally
     grid-searches (prior_mode, alpha, gamma) via inner CV per seed.

ONE continuous clock covers steps 1-3 in full: it starts right as spectral
feature generation begins and stops right after RDA finishes on the last
split. There is no per-split or per-seed breakdown -- RDA-vs-Transformer
runtime should be compared as one aggregate total across all splits and
seeds, not split-by-split, so a single total number is what's reported.

No pipeline logic is touched anywhere. This script only:
  - discovers sub-XXX/eeg/*.set files and matches them to Label/label.npy
    EXACTLY the way adftd_from_derivatives.main() does -- by POSITION
    (numeric sub-XXX order <-> label.npy row order), NOT by any subj_ID
    lookup. subj_ID in the output CSV comes from label.npy's own subj_ID
    column (via prep._norm_sid), not the sub-XXX folder name. This is a
    real behavioral difference from the AMP-SCZ pipeline (which matched
    subj_ID by dict lookup) -- ADFTD's own script works this way, so this
    benchmark reproduces it rather than "fixing" it. A [VERIFY] preview is
    printed before the timed region starts; check it,
  - writes spectral features to the exact filename/columns
    gen_Stratified_Sets.py expects (adftd_fft_dwt_from_derivatives.csv,
    subj_ID, Group),
  - runs gen_Stratified_Sets.py exactly as its own code defines it
    (via runpy, in a working directory, so its hardcoded
    INPUT_CSV/OUTPUT_DIR resolve correctly, without editing a single line
    of that file),
  - runs RDA_Test.run_one_split() on each resulting split,
  - and times the whole thing as one block.

Preprocessing source: adftd_from_derivatives.py (imported as `prep` below).
Unlike SCZ_PreProcessing.py (AMP-SCZ), this module does not expose raw-load
and windowing as separate functions -- prep.load_windows_CxT() bundles
MNE .set load, optional resample, and non-overlapping window segmentation
into one call. Rather than reach into its internals, this benchmark calls
that function wholesale and treats the bundle as NOT TIMED (windowing
itself is a cheap reshape, so lumping it in with file I/O introduces
negligible skew) -- only per-window spectral feature computation
(prep.features_for_epoch) is TIMED, same role FFT/DWT feature extraction
played in the AMP-SCZ version.

Windowing is always non-overlapping (win_sec only -- prep.load_windows_CxT
takes no stride/overlap parameter), matching the fixed 1 s, non-overlapping
epoching scheme used across all datasets in this project.

Clock scope:
    NOT TIMED (happens before the clock starts):
        - reading Label/label.npy + sub-XXX/eeg/*.set discovery
          (prep.find_subject_dirs / find_set_file), by-position pairing
        - raw .set load, optional resample to --fs, and non-overlapping
          window segmentation (prep.load_windows_CxT)
    TIMED, as one continuous block:
        - per-window spectral feature generation for every subject
          (prep.features_for_epoch -- DC centering + FFT relative
          bandpowers + DWT relative band energies + spectral summary
          features, called as the single unmodified unit
          adftd_from_derivatives.py defines), averaged across windows
          per subject
        - attaching each subject's label and writing
          adftd_fft_dwt_from_derivatives.csv
        - running gen_Stratified_Sets.py end-to-end (unmodified)
        - running rda.run_one_split() on every resulting split (each
          call's own inner CV over the alpha/gamma/prior grid, across
          all seeds, counts toward the timed total)
    Confusion-matrix generation is NOT part of the timed region: it only
    ever happens in RDA_Test.main() (never inside run_one_split()) and
    only when plot_cm=True. rda_args below sets plot_cm=False, so
    save_cm() is never invoked anywhere in this benchmark.

Usage:
    python Benchmark_ADFTD_Pipeline_Runtime.py \
        --raw_root   "Z:/.../ADFTD/derivatives" \
        --label_npy  "Z:/.../TIMING_ANALYSIS/Label" \
        --work_dir   "Z:/.../TIMING_ANALYSIS/runtime_benchmark_work" \
        --out_csv    adftd_pipeline_runtime_total.csv \
        --n_jobs     8

Place this script in the same directory as adftd_from_derivatives.py,
RDA_Test.py, and gen_Stratified_Sets.py (or adjust the sys.path insert
below).
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Import the pipeline scripts as-is. No functions are redefined or
# modified below; we only call what already exists in them.
# ------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adftd_from_derivatives as prep   # noqa: E402
import RDA_Test as rda                  # noqa: E402

SUBJECT_COL = "subj_ID"
LABEL_COL = "Group"

# Matches gen_Stratified_Sets.py's own hardcoded INPUT_CSV / OUTPUT_DIR
# constants exactly (confirmed from the uploaded ADFTD copy of that
# script). Not configurable here on purpose: that script is run
# unmodified, so these names must match what it actually reads/writes.
GEN_SPLITS_SCRIPT_NAME = "gen_Stratified_Sets.py"
GEN_SPLITS_INPUT_CSV_NAME = "adftd_fft_dwt_from_derivatives_09-08.csv"
GEN_SPLITS_OUTPUT_DIR_NAME = "stratified_holdout_splits"


def _ts_print(msg: str) -> None:
    """
    Timestamped, force-flushed print. With n_jobs > 1, several worker
    PROCESSES print concurrently, and stdout buffering across processes
    (especially over a network-mounted working dir) can make the last
    line on screen NOT be the last thing that actually happened -- a
    subject can look "stuck" while 7 others are quietly finishing in the
    background. flush=True plus a timestamp on every start/done line
    removes that ambiguity: if the clock stops advancing and no new
    [DONE] lines appear, something is actually hung, not just buffered.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# Helpers (orchestration only -- no feature/RDA/split logic here)
# ============================================================

def resolve_subject_files_and_labels(
    raw_root: Path, label_npy_path: Path, task: str
) -> dict[str, tuple[Path, object]]:
    """
    NOT TIMED. Mirrors adftd_from_derivatives.main()'s subject discovery +
    label alignment EXACTLY -- same functions (prep.find_subject_dirs,
    prep.find_set_file, prep._norm_sid), same POSITIONAL pairing (i-th
    sorted sub-XXX dir <-> i-th label.npy row), same label casting -- so
    this benchmark can't silently drift out of sync with what that script
    actually does when run standalone.

    Returns {subj_ID: (set_file_path, Group_value)}, where subj_ID comes
    from label.npy's own subj_ID column via prep._norm_sid -- NOT from
    the sub-XXX folder name.
    """
    sub_dirs = prep.find_subject_dirs(raw_root)
    if not sub_dirs:
        raise FileNotFoundError(f"No sub-* directories found under {raw_root}")
    print(f"[INFO] Found {len(sub_dirs)} subject directories under {raw_root}")

    sub_files: list[tuple[Path, Path]] = []
    for sub_dir in sub_dirs:
        fp = prep.find_set_file(sub_dir, task)
        if fp is None:
            print(f"[WARN] {sub_dir.name}: no matching .set file found, skipping.")
            continue
        sub_files.append((sub_dir, fp))

    if not sub_files:
        raise FileNotFoundError("No usable .set files found.")

    label_base = Path(label_npy_path)
    label_fp = (label_base / "label.npy") if label_base.is_dir() else label_base
    if not label_fp.exists():
        raise FileNotFoundError(f"Label file not found: {label_fp}")

    labels_arr = np.asarray(np.load(label_fp, allow_pickle=True), dtype=object)
    if labels_arr.ndim != 2 or labels_arr.shape[1] != 2:
        raise ValueError(
            f"Expected label.npy shape (N,2) [label, subj_ID], got {labels_arr.shape}"
        )

    if len(sub_files) != labels_arr.shape[0]:
        print(
            f"[WARN] {len(sub_files)} subject files but {labels_arr.shape[0]} label rows -- "
            f"these should match 1:1 by sub-XXX numeric order if the row-order assumption holds."
        )

    n_pairs = min(len(sub_files), labels_arr.shape[0])

    print(
        "\n[VERIFY] First 5 and last 5 (sub-dir -> assumed label.npy row) -- "
        "check this against what you know about the folder<->subj_ID mapping "
        "before trusting the output:"
    )
    preview_idxs = sorted(set(list(range(min(5, n_pairs))) + list(range(max(0, n_pairs - 5), n_pairs))))
    for i in preview_idxs:
        lab, sid = labels_arr[i, 0], labels_arr[i, 1]
        sub_dir, fp = sub_files[i]
        print(f"  {sub_dir.name} ({fp.name})  ->  label.npy row {i}: label={lab}, subj_ID={sid}")
    print()

    subject_entries: dict[str, tuple[Path, object]] = {}
    for i in range(n_pairs):
        sub_dir, fp = sub_files[i]
        lab, sid = labels_arr[i, 0], labels_arr[i, 1]
        subj_id = prep._norm_sid(sid)
        lab_s = str(lab).strip()
        label_val = int(lab_s) if lab_s.lstrip("-").isdigit() else lab
        subject_entries[subj_id] = (fp, label_val)

    return subject_entries


def _preprocess_one_subject(
    subj_id: str, fp: Path, win_sec: float, fs_override: float | None
) -> tuple[str, tuple[list, float, Path] | None]:
    """
    Worker unit for one subject's raw .set load + optional resample +
    non-overlapping window segmentation (win_sec length) -- calls
    prep.load_windows_CxT unmodified. Must stay a top-level function (not
    a closure/nested def) so it's picklable for ProcessPoolExecutor.
    """
    _ts_print(f"[START] {subj_id} ({fp.name}) -- raw load / resample / windowing")
    t0 = time.perf_counter()
    try:
        windows, fs = prep.load_windows_CxT(
            fp, win_sec=win_sec, fs_override=fs_override
        )
    except Exception as e:
        _ts_print(f"[WARN] Skipping {subj_id} ({fp.name}) due to error during preprocessing: {e}")
        return subj_id, None

    _ts_print(f"[DONE]  {subj_id} ({fp.name}) -- {len(windows)} window(s) in {time.perf_counter() - t0:.1f}s")
    return subj_id, (windows, fs, fp)


def preprocess_subjects(
    subject_entries: dict[str, tuple[Path, object]],
    win_sec: float, fs_override: float | None, n_jobs: int = 1,
    subject_timeout: float | None = None,
) -> dict[str, tuple[list, float, Path]]:
    """
    NOT TIMED: raw .set load + optional resample + non-overlapping window
    segmentation, via prep.load_windows_CxT unmodified (this stage bundles
    what were two separate NOT_TIMED steps in the AMP-SCZ pipeline -- see
    module docstring for why). Parallelized across subjects via
    ProcessPoolExecutor when n_jobs > 1: this stage is I/O-bound, so this
    is where parallelism cuts wall-clock the most, and since it sits
    outside the timed block, parallelizing it has no effect on the
    reported runtime number.

    If subject_timeout is set, a single subject exceeding it raises a
    clear TimeoutError naming which subject/file stalled -- e.g. a network
    share stalling on load_raw -- instead of the whole run hanging
    silently forever with no indication of where. Without a timeout,
    ProcessPoolExecutor.result() will wait indefinitely.

    Returns: {subj_id: (windows, fs, source_file_path)}
    """
    preprocessed: dict[str, tuple[list, float, Path]] = {}
    items = [(sid, fp) for sid, (fp, _lab) in subject_entries.items()]

    if n_jobs is None or n_jobs <= 1:
        results = [
            _preprocess_one_subject(sid, fp, win_sec, fs_override)
            for sid, fp in items
        ]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            future_to_subj = {
                ex.submit(_preprocess_one_subject, sid, fp, win_sec, fs_override): (sid, fp)
                for sid, fp in items
            }
            results = []
            for fut in future_to_subj:
                sid, fp = future_to_subj[fut]
                try:
                    results.append(fut.result(timeout=subject_timeout))
                except FutureTimeoutError:
                    raise TimeoutError(
                        f"Subject {sid} ({fp.name}) exceeded --subject_timeout="
                        f"{subject_timeout}s during raw load/resample/windowing. "
                        f"This is very likely a stalled network read (raw_root "
                        f"looks like it may be on a network share) rather than a "
                        f"slow computation -- check connectivity to that file."
                    )

    for subj_id, payload in results:
        if payload is not None:
            preprocessed[subj_id] = payload

    return preprocessed


def _features_for_one_subject(
    subj_id: str, windows: list, fs: float, wavelet: str
) -> tuple[str, dict | None]:
    """
    Worker unit for one subject's per-window spectral feature generation --
    calls prep.features_for_epoch unmodified per window, then averages
    across windows the same way prep.features_for_subject() does
    internally (that function isn't called directly here since it also
    does the NOT_TIMED file load -- see module docstring). Must stay a
    top-level function so it's picklable for ProcessPoolExecutor.
    """
    if not windows:
        return subj_id, None

    _ts_print(f"[START] {subj_id} -- spectral feature extraction ({len(windows)} window(s))")
    t0 = time.perf_counter()
    rows = [prep.features_for_epoch(w, fs, wavelet) for w in windows]
    subj_feats = pd.DataFrame(rows).mean(axis=0, numeric_only=True).to_dict()
    subj_feats["subj_ID"] = subj_id
    _ts_print(f"[DONE]  {subj_id} -- spectral features computed in {time.perf_counter() - t0:.1f}s")
    return subj_id, subj_feats


def generate_features_from_preprocessed(
    preprocessed: dict[str, tuple[list, float, Path]], wavelet: str, n_jobs: int = 1,
    subject_timeout: float | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    TIMED (part of the single overall clock): per-window spectral feature
    generation (prep.features_for_epoch -- DC centering + FFT relative
    bandpowers + DWT relative band energies + spectral summary features,
    called as the single unmodified unit adftd_from_derivatives.py
    defines), averaged across windows per subject. Output columns are the
    spectral feature columns (prep.FEATURE_ORDER) plus subj_ID only
    (Group is attached by the caller).

    Parallelized across subjects via ProcessPoolExecutor when n_jobs > 1.
    This stage IS inside the timed block, so n_jobs here directly affects
    the reported runtime number -- set to match the num_workers used for
    the Transformer benchmark (e.g. 8) so the RDA-vs-Transformer runtime
    comparison uses the same degree of parallelism on both sides. Note
    that if subject_timeout fires here, the time already spent waiting is
    still inside the timed window -- it's a safety net against a silent
    hang, not something to rely on for a clean benchmark number.

    Returns the per-subject feature DataFrame plus a {subj_ID: n_windows}
    dict (all attempted subjects, including those with 0 windows) for the
    end-of-run window-count distribution summary.
    """
    rows = []
    window_counts: dict[str, int] = {}
    items = [(sid, windows, fs) for sid, (windows, fs, _fp) in preprocessed.items()]

    if n_jobs is None or n_jobs <= 1:
        results = [_features_for_one_subject(sid, windows, fs, wavelet) for sid, windows, fs in items]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            future_to_subj = {
                ex.submit(_features_for_one_subject, sid, windows, fs, wavelet): sid
                for sid, windows, fs in items
            }
            results = []
            for fut in future_to_subj:
                sid = future_to_subj[fut]
                try:
                    results.append(fut.result(timeout=subject_timeout))
                except FutureTimeoutError:
                    raise TimeoutError(
                        f"Subject {sid} exceeded --subject_timeout={subject_timeout}s "
                        f"during spectral feature extraction."
                    )

    n_windows_by_subj = {sid: len(windows) for sid, windows, _fs in items}

    for subj_id, subj_feats in results:
        window_counts[subj_id] = n_windows_by_subj.get(subj_id, 0)
        if subj_feats is None:
            _ts_print(f"[WARN] subject_{subj_id} -- FAILED (no valid windows/features)")
            continue
        rows.append(subj_feats)

    return pd.DataFrame(rows), window_counts


def print_window_count_distribution(window_counts: dict[str, int]) -> None:
    """
    End-of-run summary: distributional statistics of window (epoch) count
    per subject (non-overlapping win_sec windows).
    """
    if not window_counts:
        print("\n[WARN] No window counts to summarize (no subjects processed).")
        return

    counts = pd.Series(window_counts, name="n_windows")
    print(f"\n{'=' * 60}")
    print("Window (epoch) count per subject -- distribution")
    print(f"{'=' * 60}")
    print(f"  n_subjects : {counts.shape[0]}")
    print(f"  total      : {int(counts.sum())}")
    print(f"  mean       : {counts.mean():.2f}")
    print(f"  std        : {counts.std():.2f}")
    print(f"  min        : {int(counts.min())}")
    print(f"  25%        : {counts.quantile(0.25):.1f}")
    print(f"  median     : {counts.median():.1f}")
    print(f"  75%        : {counts.quantile(0.75):.1f}")
    print(f"  max        : {int(counts.max())}")
    n_zero = int((counts == 0).sum())
    if n_zero:
        print(f"  [WARN] {n_zero} subject(s) contributed 0 windows.")


def run_stratified_split_generation(work_dir: Path) -> Path:
    """
    TIMED (part of the single overall clock): runs gen_Stratified_Sets.py
    exactly as written, unmodified -- via runpy so its top-level code
    executes in place, with the process's working directory switched to
    `work_dir` so its hardcoded relative INPUT_CSV
    ("adftd_fft_dwt_from_derivatives.csv") and OUTPUT_DIR
    ("stratified_holdout_splits") resolve inside work_dir. Returns the
    resulting splits directory.
    """
    script_path = Path(__file__).resolve().parent / GEN_SPLITS_SCRIPT_NAME
    if not script_path.exists():
        raise FileNotFoundError(
            f"{GEN_SPLITS_SCRIPT_NAME} not found next to this script: {script_path}"
        )

    prev_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        os.chdir(prev_cwd)

    return work_dir / GEN_SPLITS_OUTPUT_DIR_NAME


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_root", required=True,
                     help="Path to the ADFTD derivatives directory containing "
                          "sub-XXX/eeg/*.set files.")
    ap.add_argument("--label_npy", required=True,
                     help="Path to label.npy OR a folder containing label.npy "
                          "(shape (N,2): [label, subj_ID]). Matched to sub-XXX "
                          "dirs BY POSITION, not by subj_ID lookup -- see "
                          "module docstring.")
    ap.add_argument("--work_dir", required=True,
                     help="Working directory where adftd_fft_dwt_from_derivatives.csv "
                          "gets written and where gen_Stratified_Sets.py will run "
                          "(producing stratified_holdout_splits inside it).")
    ap.add_argument("--out_csv", default="adftd_pipeline_runtime_total.csv")

    # Passed straight through to adftd_from_derivatives, matching its own
    # argparse defaults exactly. That module has no top-level constants for
    # these (unlike SCZ_PreProcessing's TARGET_FS/WIN_SEC), so the matching
    # defaults are hardcoded here to mirror its main().
    ap.add_argument("--task", type=str, default="eyesclosed",
                     help="Substring used to pick the right .set file when a "
                          "subject has more than one.")
    ap.add_argument("--win-sec", dest="win_sec", type=float, default=1.0,
                     help="Non-overlapping window length in seconds. "
                          "prep.load_windows_CxT does not support a "
                          "stride/overlap parameter, so windows are always "
                          "non-overlapping, matching the 1 s scheme used "
                          "across all datasets in this project.")
    ap.add_argument("--fs", type=float, default=256.0,
                     help="Force this sampling rate (resamples each file). "
                          "Pass 0 to use each file's native rate instead.")
    ap.add_argument("--wavelet", type=str, default="db4",
                     help="Wavelet used for DWT band-energy features (e.g. db4, sym5).")

    # Passed straight through to RDA_Test.run_one_split, matching its own
    # defaults exactly so RDA behavior is unchanged. seeds already
    # aggregates across 5 seeds per split, and each seed's own inner CV
    # already grid-searches prior_mode/alpha/gamma internally.
    ap.add_argument("--seeds", nargs="*", type=int, default=[41, 42, 43, 44, 45])
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--rda_alpha_grid", type=float, nargs="*", default=rda.ALPHA_GRID,
                     help="Candidate alpha values (0=QDA-like, 1=LDA-like) searched "
                          "inside RDA_Test's inner CV, jointly with prior_mode and gamma.")
    ap.add_argument("--rda_gamma_grid", type=float, nargs="*", default=rda.GAMMA_GRID,
                     help="Candidate gamma values (shrinkage toward scaled identity) "
                          "searched inside RDA_Test's inner CV alongside alpha and prior_mode.")

    ap.add_argument("--n_jobs", type=int, default=8,
                     help="Number of worker processes for subject-level "
                          "parallelism (raw loading + windowing in the "
                          "not-timed preprocessing stage, and spectral "
                          "feature generation in the timed stage). Set to "
                          "1 for fully serial execution. Default 8 to "
                          "match the num_workers used for the Transformer "
                          "benchmark, so the RDA-vs-Transformer runtime "
                          "comparison uses the same degree of parallelism "
                          "on both sides.")
    ap.add_argument("--subject_timeout", type=float, default=None,
                     help="Seconds to wait for a single subject's raw-load/"
                          "windowing or feature-extraction step before raising "
                          "a clear TimeoutError naming that subject, instead of "
                          "the whole run hanging silently forever. Especially "
                          "useful when raw_root is a network share, where a "
                          "single stalled read can look identical to normal "
                          "progress with n_jobs > 1 (other workers keep "
                          "printing while one hangs). Unset (None) by default "
                          "= wait indefinitely, matching prior behavior.")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    label_npy_path = Path(args.label_npy)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    fs_override = args.fs if args.fs else None

    # ---- NOT TIMED: subject discovery + by-position label alignment ----
    subject_entries = resolve_subject_files_and_labels(raw_root, label_npy_path, task=args.task)
    labels = {sid: lab for sid, (fp, lab) in subject_entries.items()}
    print(f"[INFO] {len(subject_entries)} subject(s) resolved. "
          f"Spectral features will be computed once for all of them.")

    # ---- NOT TIMED: raw load + resample + non-overlapping windowing ----
    preprocessed = preprocess_subjects(
        subject_entries, win_sec=args.win_sec,
        fs_override=fs_override, n_jobs=args.n_jobs,
        subject_timeout=args.subject_timeout,
    )
    if not preprocessed:
        raise RuntimeError(
            f"0 of {len(subject_entries)} subject(s) survived preprocessing "
            f"(raw load / resample / windowing) -- see [WARN] lines above for "
            f"the per-subject errors. Nothing to do."
        )

    # plot_cm=False -> save_cm() (confusion-matrix generation) is never
    # invoked anywhere in this benchmark; it only ever runs from
    # RDA_Test.main(), not from run_one_split(), and only when plot_cm=True.
    rda_args = SimpleNamespace(
        train_name=None, test_name=None, val_name=None,
        subject_col=SUBJECT_COL, label_col=LABEL_COL,
        seeds=args.seeds, cv_folds=args.cv_folds,
        rda_alpha_grid=args.rda_alpha_grid, rda_gamma_grid=args.rda_gamma_grid,
        out_csv="rda_results_tmp.csv", plot_cm=False,
        cm_dir="confusion_matrices_tmp", cm_cmap="viridis",
    )

    # ================= START TIMER =================
    # Spectral feature generation for ALL subjects ->
    # write adftd_fft_dwt_from_derivatives.csv -> run gen_Stratified_Sets.py
    # -> RDA on every resulting split.
    t0 = time.perf_counter()

    master_features_df, window_counts = generate_features_from_preprocessed(
        preprocessed, wavelet=args.wavelet, n_jobs=args.n_jobs,
        subject_timeout=args.subject_timeout,
    )
    if master_features_df.empty:
        raise RuntimeError(
            f"0 of {len(preprocessed)} preprocessed subject(s) produced "
            f"features -- see [WARN] lines above. Nothing to write."
        )
    master_features_df[LABEL_COL] = master_features_df[SUBJECT_COL].astype(str).map(labels)

    # adftd_from_derivatives.py's own main() sorts by subj_ID before writing
    # its CSV (df.sort_values("subj_ID").reset_index(drop=True)). Without
    # this, master_features_df's row order here is sub-XXX numeric FOLDER
    # order (from resolve_subject_files_and_labels), not subj_ID order --
    # a different ordering, since subj_ID comes from label.npy, not the
    # folder name. gen_Stratified_Sets.py's StratifiedShuffleSplit uses a
    # fixed random_state over ROW POSITION, so a different row order here
    # silently produces different train/test membership per split, even
    # with identical subjects and seed -- matching this sort exactly is
    # required for this benchmark's splits (and RDA numbers) to match what
    # running adftd_from_derivatives.py -> gen_Stratified_Sets.py ->
    # RDA_Test.py by hand would produce.
    master_features_df = master_features_df.sort_values(SUBJECT_COL).reset_index(drop=True)

    input_csv_path = work_dir / GEN_SPLITS_INPUT_CSV_NAME
    master_features_df.to_csv(input_csv_path, index=False)
    print(f"\n[INFO] Wrote spectral feature CSV for all subjects: {input_csv_path}")

    new_splits_dir = run_stratified_split_generation(work_dir)
    print(f"[INFO] Stratified splits generated at: {new_splits_dir}")

    new_split_dirs = sorted(
        [p for p in new_splits_dir.iterdir() if p.is_dir() and p.name.lower().startswith("split_")],
        key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else p.name
    )
    if not new_split_dirs:
        raise RuntimeError(f"No split_* folders found under {new_splits_dir} "
                            f"after running {GEN_SPLITS_SCRIPT_NAME}")

    for sd in new_split_dirs:
        print(f"\n{'=' * 60}\nRDA for {sd.name}\n{'=' * 60}")
        split_args = SimpleNamespace(**vars(rda_args))
        rda.run_one_split(sd, split_args)

    elapsed = time.perf_counter() - t0
    # ================== END TIMER ===================

    print(f"\n{'=' * 60}")
    print(f"TOTAL pipeline runtime (spectral feature generation -> "
          f"stratified split generation -> RDA across {len(new_split_dirs)} "
          f"split(s) x {len(args.seeds)} seed(s)): {elapsed:.2f} s")

    pd.DataFrame([{
        "n_subjects": len(master_features_df),
        "n_splits": len(new_split_dirs),
        "n_seeds": len(args.seeds),
        "total_runtime_seconds": elapsed,
    }]).to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")

    # ---- End-of-run summary: window (epoch) count per subject ----
    print_window_count_distribution(window_counts)


if __name__ == "__main__":
    main()