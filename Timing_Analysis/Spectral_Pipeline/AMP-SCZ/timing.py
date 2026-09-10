#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import runpy
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Import the pipeline scripts as-is. No functions are redefined or
# modified below; we only call what already exists in them.
# ------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import SCZ_PreProcessing as prep   # noqa: E402
import RDA_Test as rda             # noqa: E402

SUBJECT_COL = "subj_ID"
LABEL_COL = "Group"

# RDA_Test.py's own LABEL_MAP is the single source of truth for what the
# Group column must contain ("HC"/"CHR" strings, not raw 0/1 ints). Built
# from rda.LABEL_MAP directly (not hardcoded) so this never silently goes
# stale if RDA_Test.py's mapping changes.
REVERSE_LABEL_MAP = {v: k for k, v in rda.LABEL_MAP.items()}

# Matches gen_Stratified_Sets.py's own hardcoded INPUT_CSV / OUTPUT_DIR
# constants exactly. Not configurable here on purpose: that script is run
# unmodified, so these names must match what it actually reads/writes.
GEN_SPLITS_SCRIPT_NAME = "gen_Stratified_Sets.py"
GEN_SPLITS_INPUT_CSV_NAME = "AmpSCZ_no_Overlap.csv"
GEN_SPLITS_OUTPUT_DIR_NAME = "stratified_holdout_splits_no_Overlap"


# ============================================================
# Helpers (orchestration only -- no feature/RDA/split logic here)
# ============================================================

def resolve_label_npy_path(label_npy_path: Path) -> Path:
    """
    If --label_npy was given as a directory (e.g. '.', or a folder path)
    rather than the file itself, look inside it for label.npy or
    Label/label.npy instead of failing with a confusing error from
    np.load() trying to open a directory as a file.
    """
    if label_npy_path.is_file():
        return label_npy_path

    if label_npy_path.is_dir():
        candidates = [
            label_npy_path / "label.npy",
            label_npy_path / "Label" / "label.npy",
        ]
        for c in candidates:
            if c.is_file():
                return c
        raise FileNotFoundError(
            f"--label_npy '{label_npy_path}' is a directory and neither "
            f"'label.npy' nor 'Label/label.npy' was found inside it. "
            f"Pass the path to the label.npy file itself."
        )

    raise FileNotFoundError(f"--label_npy path does not exist: {label_npy_path}")


def load_labels_from_npy(label_npy_path: Path) -> dict[str, int]:
    """
    NOT TIMED. Reads Label/label.npy the same way the original preprocessing
    script does: an (N, 2) object array where each row is [label, subj_ID]
    (label as 0/1 int, subj_ID as str). Returns {subj_ID: label}. The
    subject universe for this whole run is exactly the set of IDs in this
    file -- the stratified_holdout_splits directory is not read at all.
    """
    label_npy_path = resolve_label_npy_path(label_npy_path)
    arr = np.load(label_npy_path, allow_pickle=True)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"Expected label.npy to be an (N, 2) array of [label, subj_ID] "
            f"rows, got shape {arr.shape} from {label_npy_path}"
        )

    labels: dict[str, int] = {}
    for row in arr:
        a, b = row[0], row[1]
        # Row order is [label, subj_ID] as observed in label.npy; detect by
        # type just in case, rather than assuming position blindly.
        if isinstance(a, str) and not isinstance(b, str):
            subj_id, label = a, b
        else:
            label, subj_id = a, b
        labels[str(subj_id)] = int(label)

    return labels


def validate_labels_against_rda(labels: dict[str, int]) -> None:
    """
    NOT TIMED, and deliberately runs BEFORE the expensive raw-load/VIS-drop/
    resample stage: fails immediately and clearly if label.npy's int labels
    don't match RDA_Test.LABEL_MAP's value set, instead of letting the
    mismatch surface an hour later as a cryptic ValueError deep inside
    RDA_Test.get_Xy() after all preprocessing has already run.
    """
    seen = set(labels.values())
    expected = set(rda.LABEL_MAP.values())
    unexpected = seen - expected
    if unexpected:
        raise ValueError(
            f"label.npy contains label value(s) {sorted(unexpected)} that "
            f"RDA_Test.LABEL_MAP doesn't know how to map to a class name. "
            f"RDA_Test.LABEL_MAP = {rda.LABEL_MAP} (expects int values "
            f"{sorted(expected)} only). Fix label.npy or RDA_Test.LABEL_MAP "
            f"before running -- this is checked now, before any "
            f"preprocessing, specifically so you don't have to wait an "
            f"hour to find out."
        )


def resolve_subject_files(raw_root: Path, subject_ids: list[str]) -> dict[str, list[Path]]:
    """
    Find which raw EEG files belong to which requested subject, using
    prep.discover_subject_dirs / prep.find_eeg_files unmodified. Pure
    bookkeeping -- no I/O on the recordings themselves yet.
    """
    all_subject_dirs = prep.discover_subject_dirs(raw_root)
    wanted = set(subject_ids)
    target_dirs = [d for d in all_subject_dirs if d.name in wanted]

    missing = wanted - {d.name for d in target_dirs}
    if missing:
        preview = sorted(missing)[:5]
        print(f"[WARN] {len(missing)} subject(s) not found under raw_root: "
              f"{preview}{'...' if len(missing) > 5 else ''}")

    subject_files: dict[str, list[Path]] = {}
    for subject_dir in target_dirs:
        subj_id = subject_dir.name
        eeg_files = prep.find_eeg_files(subject_dir)
        vhdr_files = [fp for fp in eeg_files if fp.suffix.lower() == ".vhdr"]
        files_to_process = vhdr_files if vhdr_files else eeg_files
        if files_to_process:
            subject_files[subj_id] = files_to_process
        else:
            print(f"[INFO] No EEG files found for {subj_id}. Skipping.")

    return subject_files


def _preprocess_one_subject(
    subj_id: str, files: list[Path], target_fs: float
) -> tuple[str, list[tuple]]:
    """
    Worker unit for one subject's raw load + channel-level conditioning --
    identical logic to what preprocess_subjects used to run inline, just
    factored out so it can be dispatched to a process pool. Must stay a
    top-level function (not a closure/nested def) so it's picklable for
    ProcessPoolExecutor.
    """
    entries = []
    for fp in files:
        try:
            raw = prep.load_raw(fp, preload=True)

            if "VIS" in raw.ch_names:
                print(f"[INFO] Dropping VIS channel from: {fp}")
                raw.drop_channels(["VIS"])

            raw = prep.pick_eeg(raw, average_reference=True)  # native fs
            raw = prep.ensure_fs(raw, target_fs=target_fs)    # resample to target_fs

            entries.append((raw, target_fs, fp))
        except Exception as e:
            print(f"[WARN] Skipping file due to error during preprocessing: {fp}")
            print(f"       {e}")

    return subj_id, entries


def preprocess_subjects(
    subject_files: dict[str, list[Path]], target_fs: float, n_jobs: int = 1
) -> dict[str, list[tuple]]:
    """
    NOT TIMED: raw load + channel-level conditioning only. Parallelized
    across subjects via ProcessPoolExecutor when n_jobs > 1.
    """
    preprocessed: dict[str, list[tuple]] = {}
    items = list(subject_files.items())

    if n_jobs is None or n_jobs <= 1:
        results = [_preprocess_one_subject(sid, files, target_fs) for sid, files in items]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_preprocess_one_subject, sid, files, target_fs) for sid, files in items]
            results = [f.result() for f in futures]

    for subj_id, entries in results:
        if entries:
            preprocessed[subj_id] = entries

    return preprocessed


def _features_for_one_subject(
    subj_id: str, entries: list[tuple], win_sec: float
) -> tuple[str, dict | None, int]:
    """
    Worker unit for one subject's non-overlapping-window epoching + per-epoch
    spectral feature generation. Must stay a top-level function so it's
    picklable for ProcessPoolExecutor.

    Windowing uses prep.windows_from_raw (non-overlapping only -- there is
    no stride/overlap parameter in SCZ_PreProcessing.py), consistent with
    the fixed 1 s, non-overlapping epoching scheme.

    Returns (subj_id, mean feature dict or None, total epoch count across
    this subject's file(s)) -- the epoch count is used for the end-of-run
    distributional summary, not for the feature values themselves.
    """
    file_rows = []
    n_epochs_total = 0
    for raw, target_fs, fp in entries:
        epochs, fs = prep.windows_from_raw(raw, win_sec=win_sec)
        n_epochs_total += len(epochs)
        if not epochs:
            continue
        epoch_rows = [prep.features_for_epoch(ep, fs, prep.WAVELET) for ep in epochs]

        mean_feats = pd.DataFrame(epoch_rows).mean(axis=0, numeric_only=True).to_dict()
        file_rows.append(mean_feats)

    if not file_rows:
        return subj_id, None, n_epochs_total

    df = pd.DataFrame(file_rows)
    subj_feats = df.mean(axis=0, numeric_only=True).to_dict()
    subj_feats["subj_ID"] = subj_id
    return subj_id, subj_feats, n_epochs_total


def generate_features_from_preprocessed(
    preprocessed: dict[str, list[tuple]], win_sec: float, n_jobs: int = 1
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    TIMED (part of the single overall clock): non-overlapping-window
    epoching + per-epoch spectral feature generation. Parallelized across
    subjects via ProcessPoolExecutor when n_jobs > 1 -- this stage IS
    inside the timed block, so n_jobs here directly affects the reported
    runtime number.

    Returns the per-subject feature DataFrame plus a {subj_ID: n_epochs}
    dict (all attempted subjects, including those with 0 epochs) for the
    end-of-run epoch-count distribution summary.
    """
    rows = []
    epoch_counts: dict[str, int] = {}
    items = list(preprocessed.items())

    if n_jobs is None or n_jobs <= 1:
        results = [_features_for_one_subject(sid, entries, win_sec) for sid, entries in items]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(_features_for_one_subject, sid, entries, win_sec)
                for sid, entries in items
            ]
            results = [f.result() for f in futures]

    for subj_id, subj_feats, n_epochs in results:
        epoch_counts[subj_id] = n_epochs
        if subj_feats is None:
            print(f"[WARN] subject_{subj_id} -- FAILED (no valid epochs/features)")
            continue
        rows.append(subj_feats)
        print(f"[OK] subject_{subj_id} -- spectral features computed ({n_epochs} epochs)")

    return pd.DataFrame(rows), epoch_counts


def print_epoch_count_distribution(epoch_counts: dict[str, int]) -> None:
    """
    End-of-run summary: distributional statistics of epoch count per
    subject (non-overlapping win_sec windows, summed across each
    subject's file(s)).
    """
    if not epoch_counts:
        print("\n[WARN] No epoch counts to summarize (no subjects processed).")
        return

    counts = pd.Series(epoch_counts, name="n_epochs")
    print(f"\n{'=' * 60}")
    print("Epoch count per subject -- distribution")
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
        print(f"  [WARN] {n_zero} subject(s) contributed 0 epochs.")


def run_stratified_split_generation(work_dir: Path) -> Path:
    """
    TIMED (part of the single overall clock): runs gen_Stratified_Sets.py
    exactly as written, unmodified.
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
                     help="Root directory of raw AMP-SCZ EEG recordings "
                          "(the folder containing per-subject BrainVision folders).")
    ap.add_argument("--label_npy", required=True,
                     help="Path to Label/label.npy -- an (N, 2) array of "
                          "[label, subj_ID] rows.")
    ap.add_argument("--work_dir", required=True,
                     help="Working directory where AmpSCZ_no_Overlap.csv.csv gets written "
                          "and where gen_Stratified_Sets.py will run.")
    ap.add_argument("--out_csv", default="amp_scz_pipeline_runtime_total.csv")

    ap.add_argument("--fs", type=float, default=prep.TARGET_FS,
                     help="Target sampling rate (Hz) after linear-interpolation resampling.")
    ap.add_argument("--win-sec", dest="win_sec", type=float, default=prep.WIN_SEC,
                     help="Non-overlapping window epoch length (s). "
                          "SCZ_PreProcessing.windows_from_raw does not support "
                          "a stride/overlap parameter, so windows are always "
                          "non-overlapping.")

    ap.add_argument("--seeds", nargs="*", type=int, default=[41, 42, 43, 44, 45])
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--rda_alpha_grid", type=float, nargs="*", default=rda.ALPHA_GRID)
    ap.add_argument("--rda_gamma_grid", type=float, nargs="*", default=rda.GAMMA_GRID)

    ap.add_argument("--n_jobs", type=int, default=8,
                     help="Number of worker processes for subject-level parallelism.")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    label_npy_path = Path(args.label_npy)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # ---- NOT TIMED: subject universe + labels, entirely from label.npy ----
    labels = load_labels_from_npy(label_npy_path)
    all_ids = sorted(labels.keys())
    print(f"[INFO] {len(all_ids)} unique subject(s) found in {label_npy_path}. "
          f"Spectral features will be computed once for all of them.")

    # ---- NOT TIMED, FAIL FAST: validate labels BEFORE the expensive
    # raw-load/VIS-drop/resample stage runs, so a label-mapping mismatch
    # surfaces in seconds instead of after an hour of preprocessing. ----
    validate_labels_against_rda(labels)

    # ---- NOT TIMED: locate files + raw load / channel removal / downsample ----
    subject_files = resolve_subject_files(raw_root, all_ids)
    preprocessed = preprocess_subjects(subject_files, target_fs=args.fs, n_jobs=args.n_jobs)

    rda_args = SimpleNamespace(
        train_name=None, test_name=None, val_name=None,
        subject_col=SUBJECT_COL, label_col=LABEL_COL,
        seeds=args.seeds, cv_folds=args.cv_folds,
        rda_alpha_grid=args.rda_alpha_grid, rda_gamma_grid=args.rda_gamma_grid,
        out_csv="rda_results_tmp.csv", plot_cm=False,
        cm_dir="confusion_matrices_tmp", cm_cmap="viridis",
    )

    # ================= START TIMER =================
    t0 = time.perf_counter()

    master_features_df, epoch_counts = generate_features_from_preprocessed(
        preprocessed, win_sec=args.win_sec, n_jobs=args.n_jobs
    )
    # Attach labels as the strings RDA_Test.LABEL_MAP expects ("HC"/"CHR"),
    # not raw ints -- RDA_Test.py is never modified, so this conversion has
    # to happen on our side.
    master_features_df[LABEL_COL] = (
        master_features_df[SUBJECT_COL].astype(str).map(labels).map(REVERSE_LABEL_MAP)
    )

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

    # ---- End-of-run summary: epoch count per subject ----
    print_epoch_count_distribution(epoch_counts)


if __name__ == "__main__":
    main()
