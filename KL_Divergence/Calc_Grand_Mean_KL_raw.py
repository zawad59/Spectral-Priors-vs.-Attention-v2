import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
INPUT_DIR = Path("kl_raw_pairs_output")          # folder holding the 25 per-split CSVs
PATTERN = "kl_raw_by_head_query_pairs.seed*_split*.csv"
SUMMARY_CSV = Path("kl_mean_by_seed_split.csv")  # the 25-row output requested
PLOT_PATH = Path("kl_mean_by_split_index.png")

FILE_RE = re.compile(r"kl_raw_by_head_query_pairs\.seed(\d+)_split(\d+)\.csv$")

COL_CI = "KL_Correct||Incorrect"
COL_IC = "KL_Incorrect||Correct"


def per_file_means(csv_path: Path):
    """Same computation as the provided snippet: sum/size mean over finite values."""
    df = pd.read_csv(csv_path)

    ci = df[COL_CI].to_numpy(dtype=np.float64)
    ic = df[COL_IC].to_numpy(dtype=np.float64)

    ci = ci[np.isfinite(ci)]
    ic = ic[np.isfinite(ic)]

    mean_ci = ci.sum() / ci.size
    mean_ic = ic.sum() / ic.size

    return mean_ci, mean_ic


def main():
    files = sorted(INPUT_DIR.glob(PATTERN))
    print(f"Found {len(files)} CSVs matching pattern in {INPUT_DIR}/")

    rows = []
    for f in files:
        m = FILE_RE.search(f.name)
        if not m:
            print(f"[WARN] filename doesn't match expected pattern, skipping: {f.name}")
            continue
        seed, split = int(m.group(1)), int(m.group(2))
        mean_ci, mean_ic = per_file_means(f)
        rows.append({
            "seed": seed,
            "split": split,
            "KL_mean(C||I)": mean_ci,
            "KL_mean(I||C)": mean_ic,
        })
        print(f"  seed{seed}_split{split}: KL(C||I) mean={mean_ci:.6e} | KL(I||C) mean={mean_ic:.6e}")

    summary = pd.DataFrame(rows).sort_values(["seed", "split"]).reset_index(drop=True)

    if len(summary) != 25:
        print(f"[WARN] expected 25 rows, got {len(summary)} - check for missing/misnamed files")

    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.18e")
    print(f"\n[OK] wrote {SUMMARY_CSV} ({len(summary)} rows)")

    # -----------------------
    # Grand mean across the 25 per-split means
    # -----------------------
    grand_mean_ci = summary["KL_mean(C||I)"].to_numpy().sum() / len(summary)
    grand_mean_ic = summary["KL_mean(I||C)"].to_numpy().sum() / len(summary)

    # Between-split std: spread of the 25 per-split MEANS themselves (not the within-split stds).
    # This is the number that answers "is the effect stable across splits" - see prior discussion.
    grand_std_ci = summary["KL_mean(C||I)"].to_numpy().std(ddof=1)
    grand_std_ic = summary["KL_mean(I||C)"].to_numpy().std(ddof=1)

    print("\n=== Grand mean across 25 splits ===")
    print(f"KL(C||I): grand mean = {grand_mean_ci:.18e} | between-split std = {grand_std_ci:.18e}")
    print(f"KL(I||C): grand mean = {grand_mean_ic:.18e} | between-split std = {grand_std_ic:.18e}")

    # -----------------------
    # Plot: per-split mean, x = split index 1..25
    # -----------------------
    x = np.arange(1, len(summary) + 1)
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, summary["KL_mean(C||I)"], width=width, label="KL(C||I)")
    ax.bar(x + width / 2, summary["KL_mean(I||C)"], width=width, label="KL(I||C)")
    ax.axhline(grand_mean_ci, linestyle="--", linewidth=1, color="C0", alpha=0.7,
               label=f"grand mean C||I = {grand_mean_ci:.3e}")
    ax.axhline(grand_mean_ic, linestyle="--", linewidth=1, color="C1", alpha=0.7,
               label=f"grand mean I||C = {grand_mean_ic:.3e}")
    ax.set_xlabel("Split index (1-25)")
    ax.set_ylabel("Mean KL divergence")
    ax.set_title("Per-split mean KL(Correct||Incorrect) vs KL(Incorrect||Correct)")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"[OK] wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()