#!/usr/bin/env python3
"""
Null baseline for Table 2: KL divergence from each OBSERVED attention row
to the uniform distribution 1/T.

This is the "how spread-out is attention, period" reference scale --
independent of the Cor/Inc distinction entirely. It answers a different
question than KL(Cor||Cor)/KL(Inc||Inc) (how much do same-group subjects'
attention patterns differ from EACH OTHER): here we ask how far a single
row's attention pattern sits from maximally-uninformative (flat) attention.
Reported alongside the within-group and cross-group KL values in Table 2
so a reader can judge whether those values are "small" in absolute terms.

Input: attn_rows_{correct,incorrect}_seed{X}_split{Y}.csv, one row per
(subject, head, query), with metadata columns (subj_id, y_true, y_pred,
correct, head, q) followed by key columns (k0..k{T-1}) that sum to ~1
across each row. T is inferred from the number of k* columns actually
present in each file -- not hardcoded -- so this works whether a file's
attention rows include or exclude a CLS key column.

Output: one companion CSV per input file (metadata columns + a new
KL_to_uniform column, one value per row), plus a single combined long CSV
across all files (with seed/split/group columns parsed from the filename)
for convenient pooling into Table 2.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

EPS = 1e-12

# -----------------------
# Config
# -----------------------
INPUT_DIR = Path(".")
OUTPUT_DIR = Path("kl_to_uniform_output")

# Matches both attn_rows_correct_seed{X}_split{Y}.csv and
# attn_rows_incorrect_seed{X}_split{Y}.csv.
FILE_RE = re.compile(r"^attn_rows_(correct|incorrect)_seed(\d+)_split(\d+)\.csv$")

K_COL_RE = re.compile(r"^k\d+$")

META_COLS = ["subj_id", "y_true", "y_pred", "correct", "head", "q"]


def discover_files(input_dir: Path):
    """Find every attn_rows_{correct,incorrect}_seed{X}_split{Y}.csv file."""
    files = []
    for f in sorted(input_dir.glob("attn_rows_*_seed*_split*.csv")):
        m = FILE_RE.match(f.name)
        if not m:
            continue
        group, seed, split = m.group(1), m.group(2), m.group(3)
        files.append((group, seed, split, f))
    return files


def kl_to_uniform(P: np.ndarray) -> np.ndarray:
    """
    KL(row || uniform) for every row in P, where P is (n_rows, T) and each
    row is (approximately) a probability distribution over T keys.

    KL(p || 1/T) = sum_k p_k * log(p_k * T)
                 = sum_k p_k*log(p_k) + log(T)     (since sum_k p_k == 1)
                 = -H(p) + log(T)

    i.e. log(T) minus each row's own Shannon entropy -- exactly 0 only for
    a perfectly uniform row, and larger the more peaked/concentrated a
    row's attention is.
    """
    T = P.shape[1]
    P = np.clip(P, EPS, 1.0)
    P = P / P.sum(axis=1, keepdims=True)  # defensive renormalization
    return np.sum(P * np.log(P), axis=1) + np.log(T)


def process_file(group: str, seed: str, split: str, fp: Path, output_dir: Path):
    """
    Full per-file pipeline: load, compute per-row KL-to-uniform, write a
    companion CSV. Runs standalone (no shared state) so it can be
    dispatched to a separate worker process.
    """
    tag = f"{group}_seed{seed}_split{split}"
    df = pd.read_csv(fp)

    k_cols = [c for c in df.columns if K_COL_RE.match(c)]
    if not k_cols:
        print(f"[WARN] {tag}: no k* columns found in {fp.name}, skipping")
        return None
    # Preserve k0, k1, ... k{T-1} order rather than whatever order pandas
    # happened to read the header in.
    k_cols = sorted(k_cols, key=lambda c: int(c[1:]))
    T = len(k_cols)

    meta_cols_present = [c for c in META_COLS if c in df.columns]
    missing_meta = [c for c in META_COLS if c not in df.columns]
    if missing_meta:
        print(f"[WARN] {tag}: missing expected metadata column(s) {missing_meta} "
              f"(continuing with what's available)")

    P = df[k_cols].to_numpy(dtype=np.float64)
    kl = kl_to_uniform(P)

    out = df[meta_cols_present].copy()
    out["seed"] = seed
    out["split"] = split
    out["group"] = group
    out["T"] = T
    out["KL_to_uniform"] = kl

    out_path = output_dir / f"kl_to_uniform.{tag}.csv"
    out.to_csv(out_path, index=False, float_format="%.18e")

    print(f"[OK] {tag}: wrote {out_path.name} | rows={len(out)} | T={T} | "
          f"KL_to_uniform mean={kl.mean():.6f} std={kl.std(ddof=1) if len(kl) > 1 else 0.0:.6f}")

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_dir", type=Path, default=INPUT_DIR,
                     help="Directory containing attn_rows_{correct,incorrect}_seed{X}_split{Y}.csv files.")
    ap.add_argument("--output_dir", type=Path, default=OUTPUT_DIR,
                     help="Directory to write per-file and combined KL-to-uniform CSVs.")
    ap.add_argument("--combined_out", default="kl_to_uniform_all.csv",
                     help="Filename (inside output_dir) for the single combined long CSV "
                          "across every file, for convenient pooling into Table 2.")
    ap.add_argument("--n_jobs", type=int, default=-1,
                    help="Parallel worker processes across files (-1 = all cores, "
                         "1 = sequential). Each file's load+compute+write is fully "
                         "independent, so this scales close to linearly with cores.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = discover_files(args.input_dir)

    print(f"Discovered {len(files)} attn_rows_*.csv file(s) "
          f"({sum(1 for g,_,_,_ in files if g=='correct')} correct, "
          f"{sum(1 for g,_,_,_ in files if g=='incorrect')} incorrect)")

    if not files:
        print("[INFO] Nothing to process.")
        return

    # Each file's load+compute+write is fully independent (no shared state),
    # so this parallelizes cleanly across processes. loky (the default
    # joblib backend) starts separate worker processes -- appropriate here
    # since the KL computation is CPU-bound numpy work, not I/O-bound once
    # the CSV is loaded, so threading would be capped by the GIL.
    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(process_file)(group, seed, split, fp, args.output_dir)
        for group, seed, split, fp in files
    )

    frames = [r for r in results if r is not None]
    if not frames:
        print("[ERROR] No files produced output.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined_path = args.output_dir / args.combined_out
    combined.to_csv(combined_path, index=False, float_format="%.18e")

    print(f"\nWrote {len(frames)}/{len(files)} per-file CSVs and combined "
          f"{combined_path.name} ({len(combined)} rows total) to {args.output_dir}/")

    print("\n--- KL_to_uniform summary by group (for reference / sanity-checking only) ---")
    for group, sub in combined.groupby("group"):
        vals = sub["KL_to_uniform"].to_numpy()
        print(f"  {group}: n_rows={len(vals)}  mean={vals.mean():.3e}  "
              f"std={vals.std(ddof=1):.3e}  median={np.median(vals):.3e}")

    # -----------------------
    # THE Table 2 reference value: pooled across correct+incorrect together.
    #
    # Unlike KL(Cor||Cor)/KL(Inc||Inc)/KL(Cor||Inc)/KL(Inc||Cor) -- which are
    # each inherently defined with respect to the correct/incorrect
    # distinction -- "how far is a typical attention row from uniform" is
    # not a group-specific question, so it should be reported as ONE
    # absolute reference scale, not two separate numbers. Correctly pools
    # ALL rows together directly (not a naive average of the two group
    # means above, which would misweight them given correct/incorrect have
    # very different row counts).
    # -----------------------
    vals_all = combined["KL_to_uniform"].to_numpy()
    overall_mean = vals_all.mean()
    overall_std = vals_all.std(ddof=1)
    overall_median = np.median(vals_all)

    print(f"\n--- KL_to_uniform: OVERALL reference value for Table 2 (correct+incorrect pooled) ---")
    print(f"  n_rows={len(vals_all)}  mean={overall_mean:.3e}  std={overall_std:.3e}  "
          f"median={overall_median:.3e}")
    print(f"  -> report this single value as the divergence-from-uniform baseline; "
          f"the per-group breakdown above is for sanity-checking that correct/incorrect "
          f"aren't wildly different from each other, not for reporting separately.")


if __name__ == "__main__":
    main()