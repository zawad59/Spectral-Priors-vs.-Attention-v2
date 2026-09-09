import re
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12

# -----------------------
# Config
# -----------------------
INPUT_DIR = Path(".")                      # where correct_seed{X}_split{Y}.npy / incorrect_... live
OUTPUT_DIR = Path("kl_raw_pairs_output")    # will be created; holds the 25 CSVs
EXPECTED_PAIRS = 25

# ASSUMPTION: per-pair subject metadata CSV naming. Adjust if yours differs.
SUBJECT_PREDS_PATTERN = "subject_preds_seed{seed}_split{split}.csv"

CORRECT_RE = re.compile(r"^correct_seed(\d+)_split(\d+)\.npy$")


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


def process_pair(seed: str, split: str, corr_path: Path, inc_path: Path, output_dir: Path):
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

    Nc, H, Q, K = A_c.shape
    Ni = A_i.shape[0]

    # -----------------------
    # Pairwise KL per (head,query), no averaging
    # Shapes:
    #   KL_C||I : (Nc,Ni,H,Q)
    #   KL_I||C : (Nc,Ni,H,Q)  (computed with swapped args)
    # -----------------------
    P = A_c[:, None, :, :, :]   # (Nc,1,H,Q,K)
    R = A_i[None, :, :, :, :]   # (1,Ni,H,Q,K)

    kl_c_given_i = kl_vec(P, R)   # (Nc,Ni,H,Q)
    kl_i_given_c = kl_vec(R, P)   # (Nc,Ni,H,Q)

    # -----------------------
    # Write long CSV:
    # columns: subj_ID(Correct), subj_ID(Incorrect), Head, Query, KL_Correct||Incorrect, KL_Incorrect||Correct
    # rows = Nc * Ni * H * Q
    # -----------------------
    corr_arr = np.repeat(np.array(corr_ids, dtype=int), Ni * H * Q)
    inc_arr  = np.tile(np.repeat(np.array(inc_ids, dtype=int), H * Q), Nc)

    head_arr = np.tile(np.repeat(np.arange(H, dtype=int), Q), Nc * Ni)
    q_arr    = np.tile(np.arange(Q, dtype=int), Nc * Ni * H)

    kl_ci_flat = kl_c_given_i.reshape(-1)   # aligns with (c,i,h,q) order
    kl_ic_flat = kl_i_given_c.reshape(-1)

    df = pd.DataFrame({
        "subj_ID(Correct)": corr_arr,
        "subj_ID(Incorrect)": inc_arr,
        "Head": head_arr,
        "Query": q_arr,
        "KL_Correct||Incorrect": kl_ci_flat,
        "KL_Incorrect||Correct": kl_ic_flat,
    })

    out_path = output_dir / f"kl_raw_by_head_query_pairs.{tag}.csv"
    df.to_csv(out_path, index=False, float_format="%.18e")

    print(f"[OK] {tag}: wrote {out_path.name} | rows={len(df)} | A_c={A_c.shape} A_i={A_i.shape}")
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = discover_pairs(INPUT_DIR)

    print(f"Discovered {len(pairs)} correct/incorrect .npy pairs "
          f"(expected {EXPECTED_PAIRS})")

    written = 0
    for seed, split, corr_path, inc_path in pairs:
        if process_pair(seed, split, corr_path, inc_path, OUTPUT_DIR):
            written += 1

    print(f"\nWrote {written} CSV(s) to {OUTPUT_DIR}/")
    if written != EXPECTED_PAIRS:
        print(f"[WARN] expected exactly {EXPECTED_PAIRS} CSVs, got {written}. "
              f"Check for missing .npy pairs or missing subject_preds files above.")


if __name__ == "__main__":
    main()