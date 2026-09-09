import re
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12

# -----------------------
# Config
# -----------------------
INPUT_DIR = Path(".")                      # where correct_seed{X}_split{Y}.npy / incorrect_... live
OUTPUT_DIR = Path("kl_raw_within_group_output")    # will be created; holds the CSVs
EXPECTED_PAIRS = 25

# ASSUMPTION: per-pair subject metadata CSV naming. Adjust if yours differs.
SUBJECT_PREDS_PATTERN = "subject_preds_seed{seed}_split{split}.csv"

CORRECT_RE = re.compile(r"^correct_seed(\d+)_split(\d+)\.npy$")

# Number of random disjoint-half partitions to draw per group, per (seed,split).
# Each partition contributes its own set of cross-half pairs; final KL(Cor||Cor) /
# KL(Inc||Inc) is the average over ALL of these (all splits, all pairs, all heads/queries
# -- or average however you like downstream from the long CSV).
N_RANDOM_SPLITS = 20

# Base RNG seed for reproducibility of the random half-partitions (NOT the same as the
# "seed{X}" in filenames, which refers to the model training seed).
RNG_SEED = 12345


def _prep_keys(attn_hnn: np.ndarray) -> np.ndarray:
    """
    attn_hnn: (H, N, N) attention probs (already softmaxed).
    Returns: (H, Q, K) where K=64 keys and Q is N (if N=65) or N (if N=64)
    - If N==65: keep queries 0..64, drop CLS key column -> keys=1..64 (K=64)
    - If N==64: keep queries 0..63, keys 0..63 (K=64)
    Renormalizes each (H,Q,:) row to sum to 1 across keys.
    """
    H, N, _ = attn_hnn.shape
    if N == 65:
        # queries: 0..64 (65), keys: 1..64 (64)
        A = attn_hnn[:, :, 1:]        # (H, 65, 64)
    elif N == 64:
        A = attn_hnn                  # (H, 64, 64)
    else:
        # If your N is different, still compute KL over all keys
        A = attn_hnn                  # (H, N, N)

    A = A.astype(np.float64)
    A = np.clip(A, EPS, 1.0)

    # renormalize rows over keys
    A = A / A.sum(axis=-1, keepdims=True)
    return A  # (H,Q,K)


def kl_vec(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    KL(p||q) over last axis.
    p,q: (..., K) probability vectors
    returns: (...) KL values
    """
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    return np.sum(p * np.log(p / q), axis=-1)


def discover_pairs(input_dir: Path):
    """Find every correct_seed{X}_split{Y}.npy and its incorrect_ counterpart."""
    pairs = []
    for f in sorted(input_dir.glob("correct_seed*_split*.npy")):
        m = CORRECT_RE.match(f.name)
        if not m:
            continue
        seed, split = m.group(1), m.group(2)
        inc_path = input_dir / f"incorrect_seed{seed}_split{split}.npy"
        if not inc_path.exists():
            print(f"[WARN] no incorrect counterpart for {f.name} (expected {inc_path.name}) - skipping")
            continue
        pairs.append((seed, split, f, inc_path))
    return pairs


def disjoint_half_indices(n: int, rng: np.random.Generator):
    """
    Randomly partition indices [0..n-1] into two disjoint halves, as close to
    50/50 as possible (sizes n//2 and n - n//2). No index is dropped, no index
    appears in both halves, so cross-half pairs never include a self-pair.
    """
    idx = rng.permutation(n)
    n_a = n // 2
    half_a = idx[:n_a]
    half_b = idx[n_a:]
    return half_a, half_b


def within_group_kl(
    group_name: str,
    ids: list,
    A: np.ndarray,           # (N, H, Q, K)
    n_random_splits: int,
    rng: np.random.Generator,
):
    """
    For a single group (Cor or Inc), repeatedly draw a random disjoint half-split
    of its N subjects, compute KL(half_A || half_B) and KL(half_B || half_A) for
    every cross-half subject pair, every head, every query -- never comparing a
    subject to itself and never reusing within-half pairs. Returns a long DataFrame
    with a `split_iter` column so downstream code can average within a split first
    or pool everything directly (both give the same overall mean).
    """
    ids_arr = np.array(ids, dtype=int)
    n = A.shape[0]
    H, Q, K = A.shape[1], A.shape[2], A.shape[3]

    frames = []
    for split_iter in range(n_random_splits):
        half_a, half_b = disjoint_half_indices(n, rng)
        n_a, n_b = len(half_a), len(half_b)
        if n_a == 0 or n_b == 0:
            print(f"[WARN] {group_name}: n={n} too small for a disjoint half-split, skipping split_iter={split_iter}")
            continue

        A_half = A[half_a]  # (n_a, H, Q, K)
        B_half = A[half_b]  # (n_b, H, Q, K)

        P = A_half[:, None, :, :, :]  # (n_a, 1, H, Q, K)
        R = B_half[None, :, :, :, :]  # (1, n_b, H, Q, K)

        kl_a_given_b = kl_vec(P, R)   # (n_a, n_b, H, Q)  == KL(A||B)
        kl_b_given_a = kl_vec(R, P)   # (n_a, n_b, H, Q)  == KL(B||A)

        subj_a = ids_arr[half_a]
        subj_b = ids_arr[half_b]

        a_arr = np.repeat(subj_a, n_b * H * Q)
        b_arr = np.tile(np.repeat(subj_b, H * Q), n_a)

        head_arr = np.tile(np.repeat(np.arange(H, dtype=int), Q), n_a * n_b)
        q_arr = np.tile(np.arange(Q, dtype=int), n_a * n_b * H)

        df = pd.DataFrame({
            "split_iter": split_iter,
            f"subj_ID({group_name}_A)": a_arr,
            f"subj_ID({group_name}_B)": b_arr,
            "Head": head_arr,
            "Query": q_arr,
            f"KL_{group_name}A||{group_name}B": kl_a_given_b.reshape(-1),
            f"KL_{group_name}B||{group_name}A": kl_b_given_a.reshape(-1),
        })
        frames.append(df)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def process_pair(seed: str, split: str, corr_path: Path, inc_path: Path, output_dir: Path, rng: np.random.Generator):
    tag = f"seed{seed}_split{split}"

    # -----------------------
    # Load arrays + subject IDs
    # -----------------------
    A_c_raw = np.load(corr_path)     # (Nc,H,N,N)
    A_i_raw = np.load(inc_path)      # (Ni,H,N,N)

    preds_path = INPUT_DIR / SUBJECT_PREDS_PATTERN.format(seed=seed, split=split)
    if not preds_path.exists():
        print(f"[SKIP] {tag}: missing subject preds file {preds_path.name}")
        return False
    meta = pd.read_csv(preds_path)   # subj_id, correct, ...

    corr_ids = meta.loc[meta["correct"] == 1, "subj_id"].to_list()
    inc_ids  = meta.loc[meta["correct"] == 0, "subj_id"].to_list()

    assert len(corr_ids) == A_c_raw.shape[0], f"{tag}: corr_ids={len(corr_ids)} vs A_c_raw={A_c_raw.shape[0]}"
    assert len(inc_ids)  == A_i_raw.shape[0], f"{tag}: inc_ids={len(inc_ids)} vs A_i_raw={A_i_raw.shape[0]}"

    # -----------------------
    # Convert to (Nc,H,Q,K=64) and (Ni,H,Q,K=64)
    # -----------------------
    A_c = np.stack([_prep_keys(A_c_raw[i]) for i in range(A_c_raw.shape[0])], axis=0)  # (Nc,H,Q,K)
    A_i = np.stack([_prep_keys(A_i_raw[i]) for i in range(A_i_raw.shape[0])], axis=0)  # (Ni,H,Q,K)

    # -----------------------
    # Within-group KL(Cor||Cor): random disjoint-half splits, cross-half pairs only,
    # averaged over N_RANDOM_SPLITS random partitions.
    # -----------------------
    df_cor = within_group_kl("Cor", corr_ids, A_c, N_RANDOM_SPLITS, rng)
    if df_cor is not None:
        out_cor = output_dir / f"kl_raw_within_cor.{tag}.csv"
        df_cor.to_csv(out_cor, index=False, float_format="%.18e")
        print(f"[OK] {tag}: wrote {out_cor.name} | rows={len(df_cor)} | Nc={A_c.shape[0]}")
    else:
        print(f"[WARN] {tag}: Cor group too small for any disjoint split, no file written")

    # -----------------------
    # Within-group KL(Inc||Inc): same procedure, independent random splits.
    # -----------------------
    df_inc = within_group_kl("Inc", inc_ids, A_i, N_RANDOM_SPLITS, rng)
    if df_inc is not None:
        out_inc = output_dir / f"kl_raw_within_inc.{tag}.csv"
        df_inc.to_csv(out_inc, index=False, float_format="%.18e")
        print(f"[OK] {tag}: wrote {out_inc.name} | rows={len(df_inc)} | Ni={A_i.shape[0]}")
    else:
        print(f"[WARN] {tag}: Inc group too small for any disjoint split, no file written")

    return (df_cor is not None) and (df_inc is not None)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = discover_pairs(INPUT_DIR)

    print(f"Discovered {len(pairs)} correct/incorrect .npy pairs "
          f"(expected {EXPECTED_PAIRS})")

    # One top-level RNG, advanced across all (seed,split) pairs and both groups so
    # every random half-split across the whole run is reproducible from RNG_SEED
    # but not identical to any other split.
    rng = np.random.default_rng(RNG_SEED)

    written = 0
    for seed, split, corr_path, inc_path in pairs:
        if process_pair(seed, split, corr_path, inc_path, OUTPUT_DIR, rng):
            written += 1

    print(f"\nWrote within-group CSVs for {written} pair(s) to {OUTPUT_DIR}/")
    if written != EXPECTED_PAIRS:
        print(f"[WARN] expected exactly {EXPECTED_PAIRS} pairs fully processed, got {written}. "
              f"Check for missing .npy pairs, missing subject_preds files, or groups too small "
              f"for a disjoint split above.")


if __name__ == "__main__":
    main()