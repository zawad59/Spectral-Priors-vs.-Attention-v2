import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
INPUT_DIR = Path("kl_raw_within_group_output")   # folder holding the within-group CSVs
COR_PATTERN = "kl_raw_within_cor.seed*_split*.csv"
INC_PATTERN = "kl_raw_within_inc.seed*_split*.csv"
SUMMARY_CSV = Path("kl_within_mean_by_seed_split.csv")   # the 25-row output requested
PLOT_PATH = Path("kl_within_mean_by_split_index.png")

COR_FILE_RE = re.compile(r"kl_raw_within_cor\.seed(\d+)_split(\d+)\.csv$")
INC_FILE_RE = re.compile(r"kl_raw_within_inc\.seed(\d+)_split(\d+)\.csv$")

COL_CorA_CorB = "KL_CorA||CorB"
COL_CorB_CorA = "KL_CorB||CorA"
COL_IncA_IncB = "KL_IncA||IncB"
COL_IncB_IncA = "KL_IncB||IncA"


def pooled_mean(csv_path: Path, col_a: str, col_b: str):
    """
    KL(Group||Group) = (1 / (2 * n_rows)) * (sum(col_a) + sum(col_b))
    i.e. concatenate both directions into one vector and take its mean.
    Equivalent to (sum_a + sum_b) / (n_a + n_b) when n_a == n_b (they always
    are here, since col_a and col_b come from the same set of cross-half pairs).
    """
    df = pd.read_csv(csv_path)

    a = df[col_a].to_numpy(dtype=np.float64)
    b = df[col_b].to_numpy(dtype=np.float64)

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    pooled = np.concatenate([a, b])
    return pooled.sum() / pooled.size


def collect_summary(files, file_re, col_a, col_b, group_label):
    rows = []
    for f in files:
        m = file_re.search(f.name)
        if not m:
            print(f"[WARN] filename doesn't match expected pattern, skipping: {f.name}")
            continue
        seed, split = int(m.group(1)), int(m.group(2))
        mean_kl = pooled_mean(f, col_a, col_b)
        rows.append({
            "seed": seed,
            "split": split,
            f"KL_mean({group_label}||{group_label})": mean_kl,
        })
        print(f"  seed{seed}_split{split}: KL({group_label}||{group_label}) mean={mean_kl:.6e}")
    return pd.DataFrame(rows).sort_values(["seed", "split"]).reset_index(drop=True)


def main():
    cor_files = sorted(INPUT_DIR.glob(COR_PATTERN))
    inc_files = sorted(INPUT_DIR.glob(INC_PATTERN))

    print(f"Found {len(cor_files)} Cor CSVs and {len(inc_files)} Inc CSVs in {INPUT_DIR}/")

    print("\n--- KL(Cor||Cor) per split ---")
    cor_summary = collect_summary(cor_files, COR_FILE_RE, COL_CorA_CorB, COL_CorB_CorA, "Cor")

    print("\n--- KL(Inc||Inc) per split ---")
    inc_summary = collect_summary(inc_files, INC_FILE_RE, COL_IncA_IncB, COL_IncB_IncA, "Inc")

    summary = pd.merge(cor_summary, inc_summary, on=["seed", "split"], how="outer").sort_values(
        ["seed", "split"]
    ).reset_index(drop=True)

    if len(summary) != 25:
        print(f"\n[WARN] expected 25 rows, got {len(summary)} - check for missing/misnamed files")

    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.18e")
    print(f"\n[OK] wrote {SUMMARY_CSV} ({len(summary)} rows)")

    # -----------------------
    # Grand mean across the 25 per-split pooled means
    # -----------------------
    grand_mean_cor = summary["KL_mean(Cor||Cor)"].to_numpy().sum() / summary["KL_mean(Cor||Cor)"].notna().sum()
    grand_mean_inc = summary["KL_mean(Inc||Inc)"].to_numpy().sum() / summary["KL_mean(Inc||Inc)"].notna().sum()

    # Between-split std: spread of the 25 per-split pooled MEANS themselves (not the
    # within-split/within-pair stds). Same interpretation as the original script's
    # between-split std for KL(C||I) / KL(I||C) - answers "is the null-baseline stable
    # across splits."
    grand_std_cor = summary["KL_mean(Cor||Cor)"].to_numpy(dtype=np.float64)
    grand_std_cor = np.nanstd(grand_std_cor, ddof=1)
    grand_std_inc = summary["KL_mean(Inc||Inc)"].to_numpy(dtype=np.float64)
    grand_std_inc = np.nanstd(grand_std_inc, ddof=1)

    print("\n=== Grand mean across 25 splits ===")
    print(f"KL(Cor||Cor): grand mean = {grand_mean_cor:.18e} | between-split std = {grand_std_cor:.18e}")
    print(f"KL(Inc||Inc): grand mean = {grand_mean_inc:.18e} | between-split std = {grand_std_inc:.18e}")

    # -----------------------
    # Plot: per-split pooled mean, x = split index 1..25
    # -----------------------
    x = np.arange(1, len(summary) + 1)
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, summary["KL_mean(Cor||Cor)"], width=width, label="KL(Cor||Cor)")
    ax.bar(x + width / 2, summary["KL_mean(Inc||Inc)"], width=width, label="KL(Inc||Inc)")
    ax.axhline(grand_mean_cor, linestyle="--", linewidth=1, color="C0", alpha=0.7,
               label=f"grand mean Cor||Cor = {grand_mean_cor:.3e}")
    ax.axhline(grand_mean_inc, linestyle="--", linewidth=1, color="C1", alpha=0.7,
               label=f"grand mean Inc||Inc = {grand_mean_inc:.3e}")
    ax.set_xlabel("Split index (1-25)")
    ax.set_ylabel("Mean KL divergence (pooled, within-group null)")
    ax.set_title("Per-split pooled mean KL(Cor||Cor) vs KL(Inc||Inc)")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"[OK] wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()