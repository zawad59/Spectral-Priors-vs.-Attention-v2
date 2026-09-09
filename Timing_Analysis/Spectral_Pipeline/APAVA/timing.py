#!/usr/bin/env python3
"""
Benchmark_APAVA_Pipeline_Runtime.py

Times the APAVA spectral-prior pipeline as ONE simple, linear process:

  1. Compute spectral features for every subject, write ONE CSV.
     (apava_Non_Overlap.py -- unmodified.)
  2. Run gen_Stratified_Sets.py (unmodified) on that CSV to produce the
     stratified_holdout_splits_No_Overlap/split_1 ... split_N directories
     (N_SPLITS=3 in the uploaded copy of that script -- this benchmark
     reads however many split_* directories it actually produces, it
     does not hardcode a count).
  3. Run RDA (RDA_Test.py -- unmodified) on every one of those splits.
     Each call already aggregates across the 5 seeds baked into
     RDA_Test.py's own defaults ([41, 42, 43, 44, 45]), and internally
     grid-searches (prior_mode, alpha, gamma) via inner CV per seed.
     APAVA's RDA_Test.py copy has a 2-class LABEL_MAP ({0:0, 1:1},
     CLASS_NAMES {0:"HC", 1:"AD"}) -- Group values are kept as raw ints,
     no HC/CHR-style string mapping is needed.

ONE continuous clock covers steps 1-3 in full: it starts right as spectral
feature generation begins and stops right after RDA finishes on the last
split. There is no per-split or per-seed breakdown -- RDA-vs-Transformer
runtime should be compared as one aggregate total across all splits and
seeds, not split-by-split, so a single total number is what's reported.

No pipeline logic is touched anywhere. This script only:
  - discovers preproctrialsNN.mat files and matches them to label.npy
    EXACTLY the way apava_Non_Overlap.main() does -- by POSITION (numeric
    NN order <-> label.npy row order), NOT by any subj_ID lookup. subj_ID
    in the output CSV comes from label.npy's own subj_ID column (via
    prep._norm_sid), not the filename. A [DEBUG]/[VERIFY] preview is
    printed before the timed region starts, exactly as
    apava_Non_Overlap.py's own main() does -- check it before trusting
    the output,
  - writes spectral features to the exact filename/columns
    gen_Stratified_Sets.py expects (apava_Non_Overlap.csv, subj_ID,
    Group), SORTED BY subj_ID before writing -- apava_Non_Overlap.py's
    own main() does this (df.sort_values("subj_ID").reset_index(drop=True))
    right before its to_csv call, and gen_Stratified_Sets.py's
    StratifiedShuffleSplit uses a fixed random_state over ROW POSITION,
    so reproducing this exact row order is required for the resulting
    splits (and RDA numbers) to match what running apava_Non_Overlap.py
    -> gen_Stratified_Sets.py -> RDA_Test.py by hand would produce. (This
    was a real bug found and fixed in the ADFTD version of this benchmark
    -- carried over as a fix here from the start rather than reintroduced.)
  - runs gen_Stratified_Sets.py exactly as its own code defines it
    (via runpy, in a working directory, so its hardcoded
    INPUT_CSV/OUTPUT_DIR resolve correctly, without editing a single line
    of that file),
  - runs RDA_Test.run_one_split() on each resulting split,
  - and times the whole thing as one block.

Preprocessing source: apava_Non_Overlap.py (imported as `prep` below).
Like ADFTD's adftd_from_derivatives.py (and unlike AMP-SCZ's
SCZ_PreProcessing.py or TDBrain's TDB_No_Overlap.py), this module does not
expose raw-load and windowing as separate functions -- prep.load_windows_CxT()
bundles the .mat load, optional resample, and non-overlapping window
segmentation (across ALL of a subject's already-segmented trials) into one
call. Rather than reach into its internals, this benchmark calls that
function wholesale and treats the bundle as NOT TIMED (windowing itself is
a cheap reshape, so lumping it in with file I/O introduces negligible
skew) -- only per-window spectral feature computation
(prep.features_for_epoch) is TIMED, same role FFT/DWT feature extraction
played in the AMP-SCZ/ADFTD/TDBrain versions.

apava_Non_Overlap.py's load_windows_CxT(fp, win_sec, fs_override) has the
EXACT SAME signature as ADFTD's version (unlike TDBrain's, which also
takes fs and drop_channels) -- APAVA has no --task or --drop-channels
concept at all (one .mat file per subject, no auxiliary-channel drop
documented), so neither appears in this benchmark's CLI.

UNVERIFIED ASSUMPTIONS carried over unchanged from apava_Non_Overlap.py's
own module docstring -- this script does not resolve any of them, it just
surfaces the same [DEBUG]/[VERIFY] printouts that script already has:
  1. Sampling rate is read from each file's own data.fsample (NOT
     hardcoded); --fs only forces a resample if it differs from a given
     file's native rate.
  2. Subject-ID/label alignment is by-position (numeric preproctrialsNN
     order <-> label.npy row order) -- unconfirmed, same caveat as ADFTD.
  3. Each of a subject's trials is windowed independently (never straddling
     a trial boundary); trailing samples that don't fill a whole window
     are dropped per trial.
  4. There is no single "recording length" per subject -- only a set of
     independent trials, structurally different from ADFTD's one
     continuous recording.

Clock scope:
    NOT TIMED (happens before the clock starts):
        - reading label.npy + preproctrialsNN.mat discovery
          (prep.find_subject_files), by-position pairing, [DEBUG]/[VERIFY]
          printouts
        - .mat load, optional resample, and non-overlapping window
          segmentation across all trials (prep.load_windows_CxT)
    TIMED, as one continuous block:
        - per-window spectral feature generation for every subject
          (prep.features_for_epoch -- DC centering + FFT relative
          bandpowers + DWT relative band energies + spectral summary
          features, called as the single unmodified unit
          apava_Non_Overlap.py defines), averaged across ALL windows from
          ALL trials per subject
        - attaching each subject's label and writing apava_Non_Overlap.csv
          (subj_ID-sorted, matching the standalone script)
        - running gen_Stratified_Sets.py end-to-end (unmodified)
        - running rda.run_one_split() on every resulting split (each
          call's own inner CV over the alpha/gamma/prior grid, across
          all seeds, counts toward the timed total)
    Confusion-matrix generation is NOT part of the timed region: it only
    ever happens in RDA_Test.main() (never inside run_one_split()) and
    only when plot_cm=True. rda_args below sets plot_cm=False, so
    save_cm() is never invoked anywhere in this benchmark.

Usage:
    python Benchmark_APAVA_Pipeline_Runtime.py \
        --raw_root   "Z:/.../ML4H/APAVA" \
        --label_npy  "Z:/.../TIMING_ANALYSIS/Label" \
        --work_dir   "Z:/.../TIMING_ANALYSIS/runtime_benchmark_work" \
        --out_csv    apava_pipeline_runtime_total.csv \
        --n_jobs     8

Place this script in the same directory as apava_Non_Overlap.py,
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
import apava_Non_Overlap as prep   # noqa: E402
import RDA_Test as rda             # noqa: E402

SUBJECT_COL = "subj_ID"
LABEL_COL = "Group"

# Matches gen_Stratified_Sets.py's own hardcoded INPUT_CSV / OUTPUT_DIR
# constants exactly (confirmed from the uploaded APAVA copy of that
# script). Not configurable here on purpose: that script is run
# unmodified, so these names must match what it actually reads/writes.
GEN_SPLITS_SCRIPT_NAME = "gen_Stratified_Sets.py"
GEN_SPLITS_INPUT_CSV_NAME = "apava_Non_Overlap.csv"
GEN_SPLITS_OUTPUT_DIR_NAME = "stratified_holdout_splits_No_Overlap"


def _ts_print(msg: str) -> None:
    """
    Timestamped, force-flushed print. With n_jobs > 1, several worker
    PROCESSES print concurrently, so stdout buffering across processes
    can make the last line on screen NOT be the last thing that actually
    happened. flush=True plus a timestamp on every start/done line
    removes that ambiguity.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# Helpers (orchestration only -- no feature/RDA/split logic here)
# ============================================================

def resolve_subject_files_and_labels(
    raw_root: Path, label_npy_path: Path
) -> dict[str, tuple[Path, object]]:
    """
    NOT TIMED. Mirrors apava_Non_Overlap.main()'s subject discovery +
    label alignment EXACTLY -- same functions (prep.find_subject_files,
    prep._norm_sid), same POSITIONAL pairing (i-th preproctrialsNN.mat
    file, sorted numerically <-> i-th label.npy row), same label casting,
    same [DEBUG]/[VERIFY] printouts -- so this benchmark can't silently
    drift out of sync with what that script actually does when run
    standalone.

    Returns {subj_ID: (mat_path, Group_value)}, where subj_ID comes from
    label.npy's own subj_ID column via prep._norm_sid -- NOT from the
    filename.
    """
    sub_files = prep.find_subject_files(raw_root)

    print(f"[DEBUG] root = {raw_root.resolve()}")
    if not sub_files:
        raise FileNotFoundError(f"No preproctrials*.mat files found under {raw_root}")
    print(f"[INFO] Found {len(sub_files)} subject file(s) under {raw_root}")
    print(f"[DEBUG] {sub_files[0].name}  (first file to be loaded to confirm channels/fs)")

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
            f"these should match 1:1 by preproctrialsNN numeric order if the row-order "
            f"assumption holds."
        )

    n_pairs = min(len(sub_files), labels_arr.shape[0])

    print(
        "\n[VERIFY] First 5 and last 5 (subject file -> assumed label.npy row) -- "
        "check this against what you know about the file<->subj_ID mapping before "
        "trusting the output:"
    )
    preview_idxs = sorted(set(list(range(min(5, n_pairs))) + list(range(max(0, n_pairs - 5), n_pairs))))
    for i in preview_idxs:
        lab, sid = labels_arr[i, 0], labels_arr[i, 1]
        fp = sub_files[i]
        print(f"  {fp.name}  ->  label.npy row {i}: label={lab}, subj_ID={sid}")
    print()

    subject_entries: dict[str, tuple[Path, object]] = {}
    for i in range(n_pairs):
        fp = sub_files[i]
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
    Worker unit for one subject's .mat load + optional resample +
    non-overlapping window segmentation across ALL of that subject's
    already-segmented trials -- calls prep.load_windows_CxT unmodified.
    Must stay a top-level function (not a closure/nested def) so it's
    picklable for ProcessPoolExecutor.
    """
    _ts_print(f"[START] {subj_id} ({fp.name}) -- .mat load / resample / windowing")
    t0 = time.perf_counter()
    try:
        windows, fs = prep.load_windows_CxT(fp, win_sec=win_sec, fs_override=fs_override)
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
    NOT TIMED: .mat load + optional resample + non-overlapping window
    segmentation, via prep.load_windows_CxT unmodified (this stage bundles
    what were two separate NOT_TIMED steps in the AMP-SCZ pipeline -- see
    module docstring for why). Parallelized across subjects via
    ProcessPoolExecutor when n_jobs > 1: this stage is I/O-bound, so this
    is where parallelism cuts wall-clock the most, and since it sits
    outside the timed block, parallelizing it has no effect on the
    reported runtime number.

    If subject_timeout is set, a single subject exceeding it raises a
    clear TimeoutError naming which subject/file stalled -- e.g. a
    network share stalling on the .mat read -- instead of the whole run
    hanging silently forever with no indication of where. Without a
    timeout, ProcessPoolExecutor.result() will wait indefinitely.

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
                        f"{subject_timeout}s during .mat load/resample/windowing. "
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
    does the NOT_TIMED .mat load -- see module docstring). Must stay a
    top-level function so it's picklable for ProcessPoolExecutor.
    """
    if not windows:
        return subj_id, None

    _ts_print(f"[START] {subj_id} -- spectral feature extraction ({len(windows)} window(s))")
    t0 = time.perf_counter()
    rows = [prep.features_for_epoch(w, fs, wavelet) for w in windows]
    feats = pd.DataFrame(rows).mean(axis=0, numeric_only=True).to_dict()
    subj_feats = {k: feats.get(k, np.nan) for k in prep.FEATURE_ORDER}
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
    called as the single unmodified unit apava_Non_Overlap.py defines),
    averaged across all windows (from all trials) per subject. Output
    columns are subj_ID plus prep.FEATURE_ORDER (Group is attached by the
    caller).

    Parallelized across subjects via ProcessPoolExecutor when n_jobs > 1.
    This stage IS inside the timed block, so n_jobs here directly affects
    the reported runtime number -- set to match the num_workers used for
    the Transformer benchmark (e.g. 8) so the RDA-vs-Transformer runtime
    comparison uses the same degree of parallelism on both sides.

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
    per subject (non-overlapping win_sec windows, summed across all of
    each subject's trials).
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
    `work_dir` so its hardcoded relative INPUT_CSV ("apava_Non_Overlap.csv")
    and OUTPUT_DIR ("stratified_holdout_splits_No_Overlap") resolve inside
    work_dir. Returns the resulting splits directory.
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
                     help="Path to the directory containing preproctrialsNN.mat files.")
    ap.add_argument("--label_npy", required=True,
                     help="Path to label.npy OR a folder containing label.npy "
                          "(shape (N,2): [label, subj_ID]). Matched to subject "
                          "files BY POSITION, not by subj_ID lookup -- see "
                          "module docstring.")
    ap.add_argument("--work_dir", required=True,
                     help="Working directory where apava_Non_Overlap.csv gets "
                          "written and where gen_Stratified_Sets.py will run "
                          "(producing stratified_holdout_splits_No_Overlap "
                          "inside it).")
    ap.add_argument("--out_csv", default="apava_pipeline_runtime_total.csv")

    # Passed straight through to apava_Non_Overlap, matching its own
    # argparse defaults exactly. That module has no top-level constants for
    # these (unlike SCZ_PreProcessing's TARGET_FS/WIN_SEC), so the matching
    # defaults are hardcoded here to mirror its main(). No --task or
    # --drop-channels here -- apava_Non_Overlap.py has neither (one .mat
    # file per subject, no auxiliary-channel drop concept).
    ap.add_argument("--win-sec", dest="win_sec", type=float, default=1.0,
                     help="Non-overlapping window length in seconds.")
    ap.add_argument("--fs", type=float, default=256.0,
                     help="Force this sampling rate (resamples each file's "
                          "trials if its native fsample differs). Default "
                          "256.0 is a no-op for data already at 256Hz. "
                          "Pass 0 to always use each file's native rate.")
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
                          "parallelism (.mat loading + windowing in the "
                          "not-timed preprocessing stage, and spectral "
                          "feature generation in the timed stage). Set to "
                          "1 for fully serial execution. Default 8 to "
                          "match the num_workers used for the Transformer "
                          "benchmark, so the RDA-vs-Transformer runtime "
                          "comparison uses the same degree of parallelism "
                          "on both sides.")
    ap.add_argument("--subject_timeout", type=float, default=None,
                     help="Seconds to wait for a single subject's .mat-load/"
                          "windowing or feature-extraction step before raising "
                          "a clear TimeoutError naming that subject, instead of "
                          "the whole run hanging silently forever. Especially "
                          "useful when raw_root is a network share. Unset "
                          "(None) by default = wait indefinitely, matching "
                          "prior behavior.")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    label_npy_path = Path(args.label_npy)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    fs_override = args.fs if args.fs else None

    # ---- NOT TIMED: subject discovery + by-position label alignment ----
    subject_entries = resolve_subject_files_and_labels(raw_root, label_npy_path)
    labels = {sid: lab for sid, (fp, lab) in subject_entries.items()}
    print(f"[INFO] {len(subject_entries)} subject(s) resolved. "
          f"Spectral features will be computed once for all of them.")

    # ---- NOT TIMED: .mat load + resample + non-overlapping windowing ----
    preprocessed = preprocess_subjects(
        subject_entries, win_sec=args.win_sec, fs_override=fs_override,
        n_jobs=args.n_jobs, subject_timeout=args.subject_timeout,
    )
    if not preprocessed:
        raise RuntimeError(
            f"0 of {len(subject_entries)} subject(s) survived preprocessing "
            f"(.mat load / resample / windowing) -- see [WARN] lines above "
            f"for the per-subject errors. Nothing to do."
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
    # write apava_Non_Overlap.csv -> run gen_Stratified_Sets.py ->
    # RDA on every resulting split.
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
    # Group is kept as the raw label value (int 0/1 for the HC/AD scheme)
    # -- RDA_Test.py's LABEL_MAP here is already {0:0, 1:1}, so no string
    # mapping is needed (unlike the AMP-SCZ HC/CHR case).
    master_features_df[LABEL_COL] = master_features_df[SUBJECT_COL].astype(str).map(labels)

    # Column order + row order match apava_Non_Overlap.py's own main()
    # output exactly, INCLUDING the subj_ID sort -- see module docstring
    # for why this sort is required, not cosmetic.
    cols = [SUBJECT_COL] + list(prep.FEATURE_ORDER) + [LABEL_COL]
    master_features_df = master_features_df[cols].sort_values(SUBJECT_COL).reset_index(drop=True)

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
