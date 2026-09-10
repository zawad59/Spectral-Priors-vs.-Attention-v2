# utils/vis_attn.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Optional
import csv
import math
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import pywt
from scipy.signal import butter, sosfiltfilt
from collections import defaultdict
# -----------------------------
# Utilities
# -----------------------------

def _core_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _extract_attn_tensors(obj) -> List[torch.Tensor]:
    """Recursively extract all tensors from possibly nested return objects (cycle-safe)."""
    out: List[torch.Tensor] = []
    seen = set()

    def rec(x):
        if x is None:
            return
        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)

        if torch.is_tensor(x):
            out.append(x)
            return
        if isinstance(x, (list, tuple)):
            for t in x:
                rec(t)
            return
        if isinstance(x, dict):
            for v in x.values():
                rec(v)
            return

    rec(obj)
    return out


def _split_logits_and_attns(model_out) -> Tuple[torch.Tensor, Any]:
    """Return (logits, attns_any) from model output."""
    if not isinstance(model_out, (tuple, list)):
        return model_out, None

    logits = model_out[0]
    attns = None

    # Heuristic: scan remaining items and pick the first that contains any tensor(s)
    for cand in model_out[1:]:
        tensors = _extract_attn_tensors(cand)
        if tensors:
            attns = tensors[0] if (len(tensors) == 1 and torch.is_tensor(tensors[0])) else tensors
            break

    return logits, attns


def _as_bhnn(a: torch.Tensor, *, batch_size: Optional[int]) -> Optional[torch.Tensor]:
    """Convert attention tensor to (B,H,N,N) if possible."""
    if not torch.is_tensor(a):
        return None

    # (B,H,N,N)
    if a.dim() == 4 and a.size(-1) == a.size(-2):
        if batch_size is None or a.size(0) == batch_size:
            return a
        # (H,B,N,N)
        if batch_size is not None and a.size(1) == batch_size:
            return a.permute(1, 0, 2, 3).contiguous()

    # (B,N,N) -> (B,1,N,N)
    if a.dim() == 3 and a.size(-1) == a.size(-2):
        if batch_size is None or a.size(0) == batch_size:
            return a.unsqueeze(1)

    # (N,N) -> (1,1,N,N)
    if a.dim() == 2 and a.size(0) == a.size(1):
        return a.unsqueeze(0).unsqueeze(0)

    return None


def _select_last_layer_attention_from_return(
    attns_any,
    *,
    args=None,
    batch_size: Optional[int] = None,
    preferred_n: int = 33,
) -> Optional[torch.Tensor]:
    """Pick the *last-layer* attention only (no averaging over layers/heads).

    """

    if attns_any is None:
        return None

    # Collect tensors in (rough) order
    tensors: List[torch.Tensor] = []
    if torch.is_tensor(attns_any):
        tensors = [attns_any]
    elif isinstance(attns_any, (list, tuple, dict)):
        tensors = _extract_attn_tensors(attns_any)
    else:
        return None

    if not tensors:
        return None

    # Group candidates by N
    candidates_by_n: Dict[int, List[torch.Tensor]] = {}
    for t in tensors:
        bhnn = _as_bhnn(t, batch_size=batch_size)
        if bhnn is None:
            continue
        N = int(bhnn.shape[-1])
        candidates_by_n.setdefault(N, []).append(bhnn)

    if not candidates_by_n:
        return None

    available = sorted(candidates_by_n.keys())
    pref = int(getattr(args, "attn_preferred_n", preferred_n)) if args is not None else preferred_n

    if pref in candidates_by_n:
        N_best = pref
    else:
        N_best = min(available, key=lambda n: abs(n - pref))

    # last occurrence for that N
    return candidates_by_n[N_best][-1]


def _collect_attn_modules(model: nn.Module):
    """Collect modules that expose `.last_attn`."""
    return [m for m in model.modules() if hasattr(m, "last_attn")]


def _select_last_layer_attention_from_modules(attn_modules, *, batch_size: Optional[int]) -> Optional[torch.Tensor]:
    """Fallback: use `.last_attn` from the last module that has it."""
    last = None
    for m in attn_modules:
        a = getattr(m, "last_attn", None)
        if a is not None:
            last = a

    if last is None:
        return None

    bhnn = _as_bhnn(last, batch_size=batch_size)
    return bhnn


def _detect_cls_token(model) -> bool:
    if model is None:
        return False
    for attr in ("cls_token", "class_token", "cls", "token_cls"):
        if hasattr(model, attr):
            return True
        if hasattr(model, "net") and hasattr(model.net, attr):
            return True
    return False


def _safe_prob(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0:
        return np.full_like(p, 1.0 / max(1, p.size), dtype=np.float64)
    p = p / s
    # avoid zeros for KL
    p = np.clip(p, eps, 1.0)
    p = p / float(p.sum())
    return p


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _safe_prob(p, eps=eps)
    q = _safe_prob(q, eps=eps)
    return float(np.sum(p * np.log(p / q)))


def _jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _safe_prob(p, eps=eps)
    q = _safe_prob(q, eps=eps)
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, eps=eps) + 0.5 * _kl(q, m, eps=eps)


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    p = _safe_prob(p)
    q = _safe_prob(q)
    return 0.5 * float(np.sum(np.abs(p - q)))


def _cosine(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _safe_prob(p, eps=eps)
    q = _safe_prob(q, eps=eps)
    denom = float(np.linalg.norm(p) * np.linalg.norm(q))
    if denom <= 0:
        return 0.0
    return float(np.dot(p, q) / denom)


# -----------------------------
# Plot helpers (optional)
# -----------------------------

def _plot_1d(y, out_png, title="", xlabel="", ylabel="value"):
    y = np.asarray(y, dtype=np.float32)
    plt.figure(figsize=(8, 3))
    plt.plot(y)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------
# Main Block For Attention Maps and Entropy for TP, FP, FN, TN
# -----------------------------

def save_transformer_attention_maps(
    *,
    model_eval: nn.Module,
    test_loader,
    out_dir: str,
    args,
    device,
    # Filtering is still supported, but by default we compute correct vs incorrect.
    focus_true_class: Optional[int] = None,
    focus_pred_class: Optional[int] = None,
    only_correct: bool = False,
    only_incorrect: bool = False,
    # Behavior controls
    preferred_n: int = 33,
    save_per_example_keydists: bool = True,
    save_mean_keydists: bool = True,
    save_head_separation_summary: bool = True,
    plot_mean_keydists: bool = False,
    max_examples: int = 0,  # keep 0 unless you still want per-example heatmaps elsewhere
):
    """Capture attention from the *last* classification layer only, without
    averaging over heads or layers.

    We then compute per-head *key distributions* (a probability distribution over
    key tokens) for each example, separated into correct vs incorrect predictions.

    Key distribution definition (per head):
      - If CLS token exists: use the CLS query row (query index 0) => A[CLS, :]
      - Else: mean over queries => mean_q A[q, :]
    Then normalize to sum to 1.

    Outputs (inside out_dir):
      aggregated_all_correct_preds/
        per_example_keydist_correct_headXX.csv
        mean_keydist_correct_headXX.csv
      aggregated_all_incorrect_preds/
        per_example_keydist_incorrect_headXX.csv
        mean_keydist_incorrect_headXX.csv
      head_separation_summary.csv  (optional)

    NOTE: If you call this function multiple times (e.g., per confusion bucket),
    it will produce correct/incorrect splits *within that subset*.
    """

    os.makedirs(out_dir, exist_ok=True)

    core = _core_model(model_eval)
    wrapper = core
    inner = core.net if hasattr(core, "net") else None

    # best-effort: ensure attention output enabled
    for m_ in (wrapper, inner):
        if m_ is not None and hasattr(m_, "output_attention"):
            try:
                m_.output_attention = True
            except Exception:
                pass

    wrapper.eval()
    if inner is not None:
        inner.eval()

    attn_modules = _collect_attn_modules(inner if inner is not None else wrapper)

    def _forward_flexible(m, batch_x, padding_mask):
        last_err = None
        for fargs in (
            (batch_x, padding_mask, None, None),
            (batch_x, padding_mask),
            (batch_x,),
        ):
            try:
                return m(*fargs)
            except TypeError as e:
                last_err = e
        raise last_err

    # Will become lists-of-arrays per head
    correct_by_head: List[List[np.ndarray]] = []
    incorrect_by_head: List[List[np.ndarray]] = []

    # Keep small debug sample if requested
    examples = []

    has_cls = _detect_cls_token(inner if inner is not None else wrapper)
    
    # accumulate windows -> subject
    subj_prob_sum = {}            # sid -> (C,)
    subj_attn_sum = {}            # sid -> (H,N,N)
    subj_count = defaultdict(int) # sid -> #windows
    subj_y = {}   
    with torch.no_grad():
        for batch in test_loader:
            # Expect (batch_x, label, padding_mask, subj_id)
            if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                batch_x, label, padding_mask, subj_id = batch[:4]
            else:
                raise ValueError(
                    "Need subj_id from DataLoader for subject-level attention. "
                    "Update Dataset to return (x, y, padding_mask, subj_id)."
                )

            batch_x = batch_x.float().to(device)
            label = label.to(device)
            try:
                padding_mask = padding_mask.float().to(device)
            except Exception:
                padding_mask = None

            # subj_id -> numpy int array
            if torch.is_tensor(subj_id):
                sid_np = subj_id.detach().cpu().numpy()
            else:
                sid_np = np.asarray(subj_id)

            out = _forward_flexible(wrapper, batch_x, padding_mask)
            logits, attns_ret = _split_logits_and_attns(out)

            probs = torch.softmax(logits, dim=1)  # (B,C)

            # attention (B,H,N,N)
            attn_bhnn = _select_last_layer_attention_from_return(
                attns_ret,
                args=args,
                batch_size=int(batch_x.size(0)),
                preferred_n=preferred_n,
            )

            if attn_bhnn is None and (str(getattr(args, "model", "")).lower() == "medformer") and (inner is not None):
                try:
                    out_alt = _forward_flexible(inner, batch_x, padding_mask)
                    logits_alt, attns_alt = _split_logits_and_attns(out_alt)
                    alt = _select_last_layer_attention_from_return(
                        attns_alt, args=args, batch_size=int(batch_x.size(0)), preferred_n=preferred_n
                    )
                    if alt is not None:
                        attn_bhnn = alt
                        probs = torch.softmax(logits_alt, dim=1)
                except TypeError:
                    pass

            if attn_bhnn is None:
                attn_bhnn = _select_last_layer_attention_from_modules(attn_modules, batch_size=int(batch_x.size(0)))

            if attn_bhnn is None or attn_bhnn.dim() != 4:
                continue

            attn_np = attn_bhnn.detach().cpu().numpy()        # (B,H,N,N)
            y_np = label.detach().cpu().numpy().astype(int)   # (B,)
            prob_np = probs.detach().cpu().numpy()            # (B,C)

            B = attn_np.shape[0]
            for i in range(B):
                sid = int(sid_np[i])
                yi = int(y_np[i])

                if sid not in subj_y:
                    subj_y[sid] = yi

                if sid not in subj_prob_sum:
                    subj_prob_sum[sid] = prob_np[i].astype(np.float64)
                    subj_attn_sum[sid] = attn_np[i].astype(np.float64)
                else:
                    subj_prob_sum[sid] += prob_np[i]
                    subj_attn_sum[sid] += attn_np[i]

                subj_count[sid] += 1

    # ---- Build correct/incorrect lists ONCE per subject ----
    # initialize collectors once we know H
    any_sid = next(iter(subj_attn_sum.keys()), None)
    if any_sid is None:
        print("[ATTN] No attention captured; nothing to save.")
        return

    H = subj_attn_sum[any_sid].shape[0]
    correct_by_head = [[] for _ in range(H)]
    incorrect_by_head = [[] for _ in range(H)]

    for sid, cnt in subj_count.items():
        if cnt <= 0:
            continue

        p_sid = subj_prob_sum[sid] / float(cnt)      # mean prob over windows
        pred_sid = int(np.argmax(p_sid))
        y_sid = int(subj_y[sid])

        A_sid = subj_attn_sum[sid] / float(cnt)      # mean attention over windows (H,N,N)

        is_correct = (pred_sid == y_sid)

        # apply your same filters, but at subject-level
        match = True
        if focus_true_class is not None and y_sid != int(focus_true_class):
            match = False
        if focus_pred_class is not None and pred_sid != int(focus_pred_class):
            match = False
        if only_correct and not is_correct:
            match = False
        if only_incorrect and is_correct:
            match = False
        if not match:
            continue

        for h in range(H):
            A = A_sid[h]  # (N,N)
            if has_cls and A.shape[0] > 0:
                key_scores = A[0, :]
            else:
                key_scores = A.mean(axis=0)

            p = _safe_prob(key_scores)
            if is_correct:
                correct_by_head[h].append(p)
            else:
                incorrect_by_head[h].append(p)
        

                if max_examples and len(examples) < max_examples:
                    examples.append({"y": yi, "p": pi, "correct": is_correct})

    # -------------------------
    # Save results
    # -------------------------
    corr_dir = os.path.join(out_dir, "aggregated_all_correct_preds")
    inc_dir = os.path.join(out_dir, "aggregated_all_incorrect_preds")
    os.makedirs(corr_dir, exist_ok=True)
    os.makedirs(inc_dir, exist_ok=True)

    if not correct_by_head:
        print("[ATTN] No samples matched your filters (nothing to save).")
        return

    H = len(correct_by_head)

    summary_rows: List[Dict[str, Any]] = []

    for h in range(H):
        corr = correct_by_head[h]
        inc = incorrect_by_head[h]

        corr_mat = np.stack(corr, axis=0) if len(corr) else None
        inc_mat = np.stack(inc, axis=0) if len(inc) else None

        # per-example
        if save_per_example_keydists:
            if corr_mat is not None:
                np.savetxt(
                    os.path.join(corr_dir, f"per_example_keydist_correct_head{h:02d}.csv"),
                    corr_mat,
                    delimiter=",",
                )
            if inc_mat is not None:
                np.savetxt(
                    os.path.join(inc_dir, f"per_example_keydist_incorrect_head{h:02d}.csv"),
                    inc_mat,
                    delimiter=",",
                )

        # means
        mu_c = _safe_prob(corr_mat.mean(axis=0)) if corr_mat is not None else None
        mu_i = _safe_prob(inc_mat.mean(axis=0)) if inc_mat is not None else None

        if save_mean_keydists:
            if mu_c is not None:
                np.savetxt(os.path.join(corr_dir, f"mean_keydist_correct_head{h:02d}.csv"), mu_c[None, :], delimiter=",")
                if plot_mean_keydists:
                    _plot_1d(mu_c, os.path.join(corr_dir, f"mean_keydist_correct_head{h:02d}.png"), title=f"Mean keydist (correct) head {h}")
            if mu_i is not None:
                np.savetxt(os.path.join(inc_dir, f"mean_keydist_incorrect_head{h:02d}.csv"), mu_i[None, :], delimiter=",")
                if plot_mean_keydists:
                    _plot_1d(mu_i, os.path.join(inc_dir, f"mean_keydist_incorrect_head{h:02d}.png"), title=f"Mean keydist (incorrect) head {h}")

        # separation summary for this head
        if save_head_separation_summary and (mu_c is not None) and (mu_i is not None):
            row = {
                "head": h,
                "n_correct": int(corr_mat.shape[0]) if corr_mat is not None else 0,
                "n_incorrect": int(inc_mat.shape[0]) if inc_mat is not None else 0,
                "JSD(mu_c,mu_i)": _jsd(mu_c, mu_i),
                "KL(mu_c||mu_i)": _kl(mu_c, mu_i),
                "KL(mu_i||mu_c)": _kl(mu_i, mu_c),
                "TV": _tv(mu_c, mu_i),
                "cosine": _cosine(mu_c, mu_i),
            }
            summary_rows.append(row)

    # Write head separation summary
    if save_head_separation_summary and summary_rows:
        # Sort by descending JSD and TV
        summary_rows_sorted = sorted(summary_rows, key=lambda r: float(r["JSD(mu_c,mu_i)"]), reverse=True)
        out_csv = os.path.join(out_dir, "head_separation_summary.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows_sorted[0].keys()))
            w.writeheader()
            w.writerows(summary_rows_sorted)

        topk = min(5, len(summary_rows_sorted))
        tops = ", ".join(
            f"h{r['head']} (JSD={r['JSD(mu_c,mu_i)']:.4g}, TV={r['TV']:.4g})" for r in summary_rows_sorted[:topk]
        )
        print(f"[ATTN] Saved per-head key distributions. Top separable heads: {tops}")
    else:
        print("[ATTN] Saved per-head key distributions (no head separation summary).")

def save_attn_qk_allCorrect_vs_allIncorrect(
    *,
    model_eval: nn.Module,
    test_loader,
    out_dir: str,
    args,
    device,
    preferred_n: int = 33,
    max_batches: int = 0,
    max_samples: int = 0,
):


    os.makedirs(out_dir, exist_ok=True)

    core = _core_model(model_eval)
    wrapper = core
    inner = core.net if hasattr(core, "net") else None

    wrapper.eval()
    if inner is not None:
        inner.eval()

    attn_modules = _collect_attn_modules(inner if inner is not None else wrapper)

    def _forward_flexible(m, batch_x, padding_mask):
        last_err = None
        for fargs in (
            (batch_x, padding_mask, None, None),
            (batch_x, padding_mask),
            (batch_x,),
        ):
            try:
                return m(*fargs)
            except TypeError as e:
                last_err = e
        raise last_err

    # accumulators (init after we know H,N)
    sum_correct = None  # (H,N,N)
    sum_incorrect = None
    n_correct = 0
    n_incorrect = 0

    seen_batches = 0
    seen_samples = 0

    # --- accumulate per-subject first ---
    subj_prob_sum = {}                 # sid -> (num_class,)
    subj_attn_sum = {}                 # sid -> (H,N,N)
    subj_count = defaultdict(int)      # sid -> #windows
    subj_y = {}                        # sid -> true label

    with torch.no_grad():
        for batch in test_loader:
            seen_batches += 1
            if max_batches and seen_batches > max_batches:
                break

            # Expect (batch_x, label, padding_mask, subj_id)
            if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                batch_x, label, padding_mask, subj_id = batch[:4]
            else:
                raise ValueError(
                    "Subject-level aggregation requires subj_id. "
                    "Update Dataset/DataLoader to yield (x, y, padding_mask, subj_id)."
                )

            batch_x = batch_x.float().to(device)
            label = label.to(device)
            try:
                padding_mask = padding_mask.float().to(device)
            except Exception:
                padding_mask = None

            # subj_id -> CPU numpy ints
            if torch.is_tensor(subj_id):
                sid_np = subj_id.detach().cpu().numpy()
            else:
                sid_np = np.asarray(subj_id)

            out = _forward_flexible(wrapper, batch_x, padding_mask)
            logits, attns_ret = _split_logits_and_attns(out)

            probs = torch.softmax(logits, dim=1)       # (B, num_class)
            B = int(batch_x.size(0))

            # last-layer attention (B,H,N,N)
            attn_last = _select_last_layer_attention_from_return(
                attns_ret, args=args, batch_size=B, preferred_n=preferred_n
            )
            if attn_last is None:
                attn_last = _select_last_layer_attention_from_modules(attn_modules, batch_size=B)
            if attn_last is None:
                continue
            if attn_last.dim() != 4 or attn_last.size(-1) != attn_last.size(-2):
                continue

            attn_np = attn_last.detach().cpu().numpy()  # (B,H,N,N)
            y_np = label.detach().cpu().numpy().astype(int)
            prob_np = probs.detach().cpu().numpy()

            for i in range(B):
                seen_samples += 1
                if max_samples and seen_samples > max_samples:
                    break

                sid = int(sid_np[i])
                yi = int(y_np[i])

                if sid not in subj_y:
                    subj_y[sid] = yi

                if sid not in subj_prob_sum:
                    subj_prob_sum[sid] = prob_np[i].astype(np.float64)
                    subj_attn_sum[sid] = attn_np[i].astype(np.float64)
                else:
                    subj_prob_sum[sid] += prob_np[i]
                    subj_attn_sum[sid] += attn_np[i]

                subj_count[sid] += 1

            if max_samples and seen_samples >= max_samples:
                break

    # --- now compute subject-level correctness ONCE per subject ---
    sum_correct = None
    sum_incorrect = None
    n_correct = 0
    n_incorrect = 0

    for sid, cnt in subj_count.items():
        if cnt <= 0:
            continue

        p_sid = subj_prob_sum[sid] / float(cnt)       # mean prob over windows
        pred_sid = int(np.argmax(p_sid))
        y_sid = int(subj_y[sid])

        A_sid = subj_attn_sum[sid] / float(cnt)       # mean attn over windows

        if sum_correct is None:
            H, N, _ = A_sid.shape
            sum_correct = np.zeros((H, N, N), dtype=np.float64)
            sum_incorrect = np.zeros((H, N, N), dtype=np.float64)

        if pred_sid == y_sid:
            sum_correct += A_sid
            n_correct += 1
        else:
            sum_incorrect += A_sid
            n_incorrect += 1

            if max_samples and seen_samples >= max_samples:
                break

    if pd is None:
        print("[ATTN][CorrectVsIncorrect] pandas not available; cannot write CSV.")
        return

    if sum_correct is None:
        print("[ATTN][CorrectVsIncorrect] No attention captured; nothing to save.")
        return

    # counts file
    pd.DataFrame(
        [{"n_correct": int(n_correct), "n_incorrect": int(n_incorrect)}]
    ).to_csv(os.path.join(out_dir, "attn_qk_counts_correct_incorrect.csv"), index=False)

    def _to_long(sum_mat: np.ndarray, count: int) -> pd.DataFrame:
        # sum_mat: (H,N,N)
        H, N, _ = sum_mat.shape
        heads = np.repeat(np.arange(H), N * N)
        q = np.tile(np.repeat(np.arange(N), N), H)
        k = np.tile(np.tile(np.arange(N), N), H)
        s = sum_mat.reshape(-1)

        df = pd.DataFrame({"head": heads, "q": q, "k": k, "sum": s})
        df["count"] = int(count)
        df["avg"] = (df["sum"] / float(count)) if count > 0 else np.nan
        return df

    df_corr = _to_long(sum_correct, n_correct)
    df_inc = _to_long(sum_incorrect, n_incorrect)

    df_corr.to_csv(os.path.join(out_dir, "attn_qk_allCorrect.csv"), index=False)
    df_inc.to_csv(os.path.join(out_dir, "attn_qk_allIncorrect.csv"), index=False)

    print(f"[ATTN][CorrectVsIncorrect] Saved to: {out_dir}")
    print(f"[ATTN][CorrectVsIncorrect] Wrote attn_qk_allCorrect.csv and attn_qk_allIncorrect.csv")

    from collections import defaultdict

def save_attn_qk_raw_per_subject_correct_incorrect(
    *,
    model_eval: nn.Module,
    test_loader,
    out_dir: str,
    args,
    device,
    seed: int,
    split_id: int,
    preferred_n: int = 33,
    drop_first_token: bool = False,   # set True if N includes CLS and you want 64 instead of 65
):
    """
    Saves per-subject attention matrices (NO averaging across subjects).

    `out_dir` should be ONE shared directory across all seeds/splits (do not
    nest a fresh subdirectory per run) -- seed/split are encoded directly in
    the filenames so nothing gets overwritten across runs.

    Outputs (all written directly into out_dir):
      - out_dir/correct_seed{seed}_split{split_id}.npy               shape: (n_corr, H, N, N)
      - out_dir/incorrect_seed{seed}_split{split_id}.npy             shape: (n_inc,  H, N, N)
      - out_dir/subject_preds_seed{seed}_split{split_id}.csv         one row per subject (sid, y_true, y_pred, correct, n_windows)
      - out_dir/attn_rows_correct_seed{seed}_split{split_id}.csv     wide CSV: one row per (sid, head, q), columns k0..k{N-1}
      - out_dir/attn_rows_incorrect_seed{seed}_split{split_id}.csv   same
    """

    os.makedirs(out_dir, exist_ok=True)

    core = _core_model(model_eval)
    wrapper = core
    inner = core.net if hasattr(core, "net") else None

    wrapper.eval()
    if inner is not None:
        inner.eval()

    attn_modules = _collect_attn_modules(inner if inner is not None else wrapper)

    def _forward_flexible(m, batch_x, padding_mask):
        last_err = None
        for fargs in (
            (batch_x, padding_mask, None, None),
            (batch_x, padding_mask),
            (batch_x,),
        ):
            try:
                return m(*fargs)
            except TypeError as e:
                last_err = e
        raise last_err

    # --- accumulate windows -> subject ---
    subj_prob_sum = {}                 # sid -> (K,)
    subj_attn_sum = {}                 # sid -> (H,N,N)
    subj_count = defaultdict(int)      # sid -> #windows
    subj_y = {}                        # sid -> true label

    with torch.no_grad():
        for batch in test_loader:
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 4):
                raise ValueError(
                    "Subject-level aggregation requires subj_id. "
                    "DataLoader must yield (x, y, padding_mask, subj_id)."
                )

            batch_x, label, padding_mask, subj_id = batch[:4]

            batch_x = batch_x.float().to(device)
            label = label.to(device)

            try:
                padding_mask = padding_mask.float().to(device)
            except Exception:
                padding_mask = None

            # subj_id -> numpy int array
            if torch.is_tensor(subj_id):
                sid_np = subj_id.detach().cpu().numpy()
            else:
                sid_np = np.asarray(subj_id)

            out = _forward_flexible(wrapper, batch_x, padding_mask)
            logits, attns_ret = _split_logits_and_attns(out)

            probs = torch.softmax(logits, dim=1)  # (B,K)
            B = int(batch_x.size(0))

            # last-layer attention (B,H,N,N)
            attn_last = _select_last_layer_attention_from_return(
                attns_ret, args=args, batch_size=B, preferred_n=preferred_n
            )
            if attn_last is None:
                attn_last = _select_last_layer_attention_from_modules(attn_modules, batch_size=B)
            if attn_last is None:
                continue
            if attn_last.dim() != 4 or attn_last.size(-1) != attn_last.size(-2):
                continue

            attn_np = attn_last.detach().cpu().numpy()  # (B,H,N,N)
            prob_np = probs.detach().cpu().numpy()      # (B,K)
            y_np = label.detach().cpu().numpy().astype(int)

            for i in range(B):
                sid = int(sid_np[i])
                yi = int(y_np[i])

                if sid not in subj_y:
                    subj_y[sid] = yi

                if sid not in subj_prob_sum:
                    subj_prob_sum[sid] = prob_np[i].astype(np.float64)
                    subj_attn_sum[sid] = attn_np[i].astype(np.float64)
                else:
                    subj_prob_sum[sid] += prob_np[i]
                    subj_attn_sum[sid] += attn_np[i]

                subj_count[sid] += 1

    if not subj_count:
        print("[ATTN][RAW] No subjects accumulated; nothing to save.")
        return

    # --- finalize per-subject prediction + per-subject mean attention ---
    corr_mats = []
    inc_mats = []
    corr_rows = []
    inc_rows = []
    meta_rows = []

    # stable ordering
    sids = sorted(subj_count.keys())

    # discover H,N from first subject
    sid0 = sids[0]
    H, N, _ = (subj_attn_sum[sid0] / float(subj_count[sid0])).shape

    # optionally drop CLS token if present and you want exactly 64x64
    def _maybe_drop(A_hnn: np.ndarray) -> np.ndarray:
        # A_hnn: (H,N,N)
        if drop_first_token and A_hnn.shape[-1] > 1:
            return A_hnn[:, 1:, 1:]
        return A_hnn

    for sid in sids:
        cnt = int(subj_count[sid])
        if cnt <= 0:
            continue

        p_sid = subj_prob_sum[sid] / float(cnt)        # (K,)
        pred_sid = int(np.argmax(p_sid))
        y_sid = int(subj_y[sid])

        A_sid = subj_attn_sum[sid] / float(cnt)        # (H,N,N)
        A_sid = _maybe_drop(A_sid)                     # maybe -> (H,N',N')

        correct = int(pred_sid == y_sid)

        meta_rows.append({
            "subj_id": sid,
            "y_true": y_sid,
            "y_pred": pred_sid,
            "correct": correct,
            "n_windows": cnt,
        })

        # save into correct/incorrect stack
        if correct:
            corr_mats.append(A_sid.astype(np.float32))
        else:
            inc_mats.append(A_sid.astype(np.float32))

        # build wide rows: one row per (head, q) with k0..k{N-1}
        H2, N2, _ = A_sid.shape
        for h in range(H2):
            for q in range(N2):
                row = {
                    "subj_id": sid,
                    "y_true": y_sid,
                    "y_pred": pred_sid,
                    "correct": correct,
                    "head": h,
                    "q": q,
                }
                # keys as columns
                for k in range(N2):
                    row[f"k{k}"] = float(A_sid[h, q, k])
                if correct:
                    corr_rows.append(row)
                else:
                    inc_rows.append(row)

    # --- save outputs ---
    corr_arr = np.stack(corr_mats, axis=0) if len(corr_mats) else np.zeros((0, H, N, N), dtype=np.float32)
    inc_arr  = np.stack(inc_mats, axis=0) if len(inc_mats)  else np.zeros((0, H, N, N), dtype=np.float32)

    correct_fname = f"correct_seed{seed}_split{split_id}.npy"
    incorrect_fname = f"incorrect_seed{seed}_split{split_id}.npy"

    np.save(os.path.join(out_dir, correct_fname), corr_arr)
    np.save(os.path.join(out_dir, incorrect_fname), inc_arr)

    if pd is not None:
        pd.DataFrame(meta_rows).to_csv(
            os.path.join(out_dir, f"subject_preds_seed{seed}_split{split_id}.csv"), index=False
        )
        pd.DataFrame(corr_rows).to_csv(
            os.path.join(out_dir, f"attn_rows_correct_seed{seed}_split{split_id}.csv"), index=False
        )
        pd.DataFrame(inc_rows).to_csv(
            os.path.join(out_dir, f"attn_rows_incorrect_seed{seed}_split{split_id}.csv"), index=False
        )

    print(f"[ATTN][RAW] Saved to: {out_dir}")
    print(f"[ATTN][RAW] {correct_fname} shape = {tuple(corr_arr.shape)}")
    print(f"[ATTN][RAW] {incorrect_fname} shape = {tuple(inc_arr.shape)}")


    
