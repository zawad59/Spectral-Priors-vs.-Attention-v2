#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize, StandardScaler

import matplotlib.pyplot as plt
import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=LinAlgWarning)

METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1 score", "AUROC", "AUPRC"]

# 3-class problem: labels are already numeric 0 / 1 / 2 in the Group column.
LABEL_MAP = {0: 0, 1: 1, 2: 2}
CLASSES   = sorted(LABEL_MAP.values())      # [0, 1, 2] — fixed column order for proba matrices
N_CLASSES = len(CLASSES)

# Display names only (used for confusion-matrix axis labels / printouts).
# Rename these if you know what 0/1/2 correspond to.
CLASS_NAMES = {0: "Class 0", 1: "Class 1", 2: "Class 2"}


# I/O
def read_csv_safely(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, encoding="latin1")


def get_Xy(df, subject_col, label_col):
    df = df.copy()
    df.columns = df.columns.str.strip()
    subject_col, label_col = subject_col.strip(), label_col.strip()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols    = [c for c in numeric_cols if c not in (subject_col, label_col)]

    X        = df[feat_cols].to_numpy(dtype=float)
    subjects = df[subject_col].astype(str).to_numpy()

    # Coerce labels to int regardless of whether the column arrived as
    # int64, float64 (0.0/1.0/2.0), or string ("0"/"1"/"2"), then validate
    # against LABEL_MAP.
    y_raw_num = pd.to_numeric(df[label_col], errors="raise").to_numpy()
    y_raw     = np.rint(y_raw_num).astype(int)

    unknown = set(np.unique(y_raw)) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Unknown labels in data: {unknown}. Expected one of {sorted(LABEL_MAP)}.")

    y = np.array([LABEL_MAP[v] for v in y_raw], dtype=int)
    return X, y, subjects, feat_cols


# Cleaning + scaling
def clean_train_apply(Xtr, Xte, feat_cols, var_eps=1e-10):
    Xtr, Xte = Xtr.copy(), Xte.copy()

    med = np.nanmedian(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    for X in (Xtr, Xte):
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(med, np.where(bad)[1])

    keep      = np.var(Xtr, axis=0) > var_eps
    feat_cols = [c for c, k in zip(feat_cols, keep) if k]
    Xtr, Xte  = Xtr[:, keep], Xte[:, keep]

    scaler = StandardScaler()
    Xtr    = scaler.fit_transform(Xtr)
    Xte    = scaler.transform(Xte)

    return Xtr, Xte, feat_cols


# Subject aggregation
def aggregate_subject_probs(subjects, y, proba, model_classes):
    """
    Aggregate row-level probabilities to subject-level.
    proba columns follow `model_classes` order (from model.classes_), which
    may be a subset of CLASSES if a fold happened to be missing a class —
    we re-expand to the full N_CLASSES column layout (missing class -> 0
    probability) so aggregation/metrics are always consistent across folds.
    """
    proba_full = np.zeros((proba.shape[0], N_CLASSES), dtype=float)
    for j, cls in enumerate(model_classes):
        proba_full[:, CLASSES.index(cls)] = proba[:, j]

    df = pd.DataFrame({"subj": subjects.astype(str), "y": y})
    for j in range(N_CLASSES):
        df[f"p{j}"] = proba_full[:, j]

    prob_cols   = [f"p{j}" for j in range(N_CLASSES)]
    proba_subj  = df.groupby("subj")[prob_cols].mean().to_numpy()
    y_true_subj = df.groupby("subj")["y"].agg(lambda s: s.value_counts().idxmax()).to_numpy()
    y_pred_subj = np.array(CLASSES)[np.argmax(proba_subj, axis=1)]
    return y_true_subj, y_pred_subj, proba_subj


# Metrics — multiclass, macro-averaged across all 3 classes
def metrics_6(y_true, y_pred, y_proba):
    out = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, labels=CLASSES,
                                     average="macro", zero_division=0),
        "Recall":    recall_score(y_true, y_pred, labels=CLASSES,
                                  average="macro", zero_division=0),
        "F1 score":  f1_score(y_true, y_pred, labels=CLASSES,
                              average="macro", zero_division=0),
    }

    try:
        out["AUROC"] = roc_auc_score(y_true, y_proba, labels=CLASSES,
                                     multi_class="ovr", average="macro")
    except Exception:
        out["AUROC"] = np.nan

    try:
        y_true_bin = label_binarize(y_true, classes=CLASSES)
        out["AUPRC"] = average_precision_score(y_true_bin, y_proba, average="macro")
    except Exception:
        out["AUPRC"] = np.nan

    return out


def mean_std_str(vals):
    v    = np.asarray(vals, dtype=float)
    m, s = np.nanmean(v), (np.nanstd(v, ddof=1) if np.sum(~np.isnan(v)) > 1 else 0.0)
    return f"{100*m:.2f}±{100*s:.2f}"


PRIOR_MODES = ["empricial", "uniform", "balanced"]

def make_lda(ytr, prior_mode):
    if prior_mode == "empricial":
        # Default LDA behavior:
        # priors are estimated from class frequencies in ytr
        priors = None
    if prior_mode == "uniform":
        priors = np.ones(N_CLASSES) / N_CLASSES
    elif prior_mode == "balanced":
        counts = np.bincount(ytr, minlength=N_CLASSES).astype(float)
        priors = (1.0 / counts) / (1.0 / counts).sum()
    return LinearDiscriminantAnalysis(solver="eigen", shrinkage='auto', priors=priors)


# Inner CV
def cv_score(Xtr_full, ytr, subj_tr, feat_cols, splits):
    best_metrics = None
    best_score   = -np.inf
    best_prior   = None

    for prior_mode in PRIOR_MODES:
        fold_metrics = []

        for tr_idx, va_idx in splits:
            Xtr_f, Xva_f = Xtr_full[tr_idx], Xtr_full[va_idx]
            ytr_f, yva_f = ytr[tr_idx],       ytr[va_idx]
            subj_va      = subj_tr[va_idx]

            # Require all 3 classes present in the training fold so the
            # fitted model's classes_ (and proba column layout) is complete.
            if len(np.unique(ytr_f)) < N_CLASSES:
                continue

            try:
                Xtr_c, Xva_c, _ = clean_train_apply(Xtr_f, Xva_f, feat_cols)
                model            = make_lda(ytr_f, prior_mode)
                model.fit(Xtr_c, ytr_f)
            except Exception as e:
                print(f"[FOLD FAIL] {type(e).__name__}: {e}")
                continue

            if len(model.classes_) < N_CLASSES:
                continue

            proba_va                    = model.predict_proba(Xva_c)
            y_true_s, y_pred_s, proba_s = aggregate_subject_probs(
                subj_va, yva_f, proba_va, model.classes_
            )

            # Need at least 2 distinct classes present at subject level in
            # the val fold for the metrics (esp. AUROC/AUPRC) to be defined.
            if len(np.unique(y_true_s)) < 2:
                continue

            fold_metrics.append(metrics_6(y_true_s, y_pred_s, proba_s))

        if not fold_metrics:
            continue

        mean_f1 = float(np.nanmean([m["F1 score"] for m in fold_metrics]))
        if mean_f1 > best_score:
            best_score   = mean_f1
            best_prior   = prior_mode
            best_metrics = {k: float(np.nanmean([m[k] for m in fold_metrics]))
                            for k in METRIC_NAMES}

    return best_metrics, best_prior


# Confusion matrix
def save_cm(y_true, y_pred, class_names, out_png, cmap="viridis"):
    cm    = confusion_matrix(y_true, y_pred, labels=CLASSES).astype(float)
    denom = cm.sum(axis=1, keepdims=True)
    cm    = np.divide(cm, denom, out=np.zeros_like(cm), where=denom != 0)

    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=300)
    im      = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True",      fontsize=10)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    thresh = (cm.max() + cm.min()) / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=15,
                    color="white" if cm[i, j] > thresh else "black")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# Single split
def run_one_split(splits_dir, args):
    train_name = args.train_name or (
        "train_scaled.csv" if (splits_dir / "train_scaled.csv").exists() else "train.csv"
    )
    test_name = args.test_name or (
        "test_scaled.csv" if (splits_dir / "test_scaled.csv").exists() else "test.csv"
    )
    val_name = args.val_name or (
        "val_scaled.csv" if (splits_dir / "val_scaled.csv").exists() else "val.csv"
    )

    train_df = read_csv_safely(splits_dir / train_name)
    test_df  = read_csv_safely(splits_dir / test_name)

    # Merge validation subjects into the training pool. There is no separate
    # val-based model selection step in this script (model selection is done
    # via inner CV on the training set), so val subjects are just additional
    # training data.
    val_path = splits_dir / val_name
    if val_path.exists():
        val_df   = read_csv_safely(val_path)
        train_df = pd.concat([train_df, val_df], ignore_index=True)
    else:
        print(f"[WARN] {splits_dir.name}: no {val_name} found; proceeding with train.csv only")

    Xtr, ytr, subj_tr, feat_cols = get_Xy(train_df, args.subject_col, args.label_col)
    Xte, yte, subj_te, _         = get_Xy(test_df,  args.subject_col, args.label_col)

    print(f"Label mapping: {LABEL_MAP}  |  Classes: {CLASSES}")
    print(f"Train subjects: {len(np.unique(subj_tr))}  |  Test subjects: {len(np.unique(subj_te))}")
    print(f"Train class counts: " + ", ".join(f"{c}={np.sum(ytr==c)}" for c in CLASSES))
    print(f"Test  class counts: " + ", ".join(f"{c}={np.sum(yte==c)}" for c in CLASSES))

    all_rows              = []
    test_metrics_per_seed = {m: [] for m in METRIC_NAMES}
    last_trues, last_preds = None, None

    for seed in args.seeds:
        cv     = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=seed)
        splits = list(cv.split(Xtr, ytr, groups=subj_tr))

        cv_m, best_prior = cv_score(Xtr, ytr, subj_tr, feat_cols, splits)
        if cv_m is None:
            raise RuntimeError(f"Inner CV failed for seed={seed}.")

        # refit on full train, evaluate on test
        Xtr_c, Xte_c, _ = clean_train_apply(Xtr, Xte, feat_cols)
        model            = make_lda(ytr, best_prior)
        model.fit(Xtr_c, ytr)
        proba_te         = model.predict_proba(Xte_c)

        y_true_s, y_pred_s, proba_s = aggregate_subject_probs(
            subj_te, yte, proba_te, model.classes_
        )
        test_m = metrics_6(y_true_s, y_pred_s, proba_s)

        last_trues, last_preds = y_true_s, y_pred_s

        for m in METRIC_NAMES:
            test_metrics_per_seed[m].append(test_m[m])

        row = {"seed": seed, "best_prior": best_prior, "cv_F1": cv_m["F1 score"]}
        for m in METRIC_NAMES:
            row[f"cv_{m}"]   = cv_m[m]
            row[f"test_{m}"] = test_m[m]
        all_rows.append(row)

        print(f"[seed={seed}]  prior={best_prior}  inner-CV F1={cv_m['F1 score']:.4f}  "
              f"test F1={test_m['F1 score']:.4f}")

    print("\n--- TEST mean ± std across seeds ---")
    for m in METRIC_NAMES:
        print(f"  {m}: {mean_std_str(test_metrics_per_seed[m])}")

    # per-seed means (for correct cross-split aggregation)
    seed_means = {m: float(np.nanmean(test_metrics_per_seed[m])) for m in METRIC_NAMES}

    return all_rows, seed_means, last_trues, last_preds


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir",  required=True)
    ap.add_argument("--train_name",  default=None)
    ap.add_argument("--test_name",   default=None)
    ap.add_argument("--val_name",    default=None,
                    help="Validation CSV name (default: val.csv / val_scaled.csv). "
                         "Its subjects are merged into the training pool before CV.")
    ap.add_argument("--subject_col", default="subj_ID")
    ap.add_argument("--label_col",   default="Group")
    ap.add_argument("--seeds",       nargs="*", type=int, default=[41, 42, 43, 44, 45])
    ap.add_argument("--cv_folds",    type=int,  default=5)
    ap.add_argument("--out_csv",     default="lda_results.csv")
    ap.add_argument("--plot_cm",     action="store_true")
    ap.add_argument("--cm_dir",      default="confusion_matrices")
    ap.add_argument("--cm_cmap",     default="viridis")
    args = ap.parse_args()

    root       = Path(args.splits_dir)
    split_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("split_")],
        key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else p.name
    )

    targets            = split_dirs if split_dirs else [root]
    all_rows           = []
    # collect per-split means for correct grand mean/std
    per_split_means    = {m: [] for m in METRIC_NAMES}
    all_trues, all_preds = [], []

    for sd in targets:
        print(f"\n{'='*50}\n{sd.name}\n{'='*50}")
        rows, seed_means, last_trues, last_preds = run_one_split(sd, args)

        for r in rows:
            r["split"] = sd.name
        all_rows.extend(rows)

        # one mean per split (averaged over seeds) — correct hierarchy
        for m in METRIC_NAMES:
            per_split_means[m].append(seed_means[m])

        # For the confusion matrix, use each split's final-seed test predictions.
        if args.plot_cm and last_trues is not None:
            all_trues.append(last_trues)
            all_preds.append(last_preds)

    out_path = Path(args.out_csv)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)

    # grand mean ± std across split-level means
    print("\n=== OVERALL TEST (mean ± std across 5 split-level means) ===")
    for m in METRIC_NAMES:
        print(f"  {m}: {mean_std_str(per_split_means[m])}")

    if args.plot_cm and all_trues:
        class_names = [CLASS_NAMES[c] for c in CLASSES]
        save_cm(
            np.concatenate(all_trues), np.concatenate(all_preds),
            class_names,
            Path(args.cm_dir) / "cm_all_splits.png",
            cmap=args.cm_cmap,
        )

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()