#!/usr/bin/env python3
"""
Site-leakage control check: can the spectral features predict SITE rather
than diagnosis? If not (near-chance performance), that's evidence the
spectral features don't carry a site signature that could confound the
diagnosis classifier.

Reuses RDA_Test.py's exact classifier configuration:
  - RegularizedDiscriminantAnalysis(alpha, gamma, priors) -- the from-scratch
    two-knob Regularized Discriminant Analysis (Friedman, 1989) implementation
    from RDA_Test.py (sklearn has no built-in RDA):
        Sigma_c(alpha)        = (1-alpha)*S_c + alpha*S_pooled
        Sigma_c(alpha, gamma) = (1-gamma)*Sigma_c(alpha) + gamma*(tr(Sigma_c(alpha))/p)*I
    alpha=1 -> LDA (pooled covariance), alpha=0 -> QDA (per-class covariance);
    gamma shrinks toward a scaled identity for numerical stability.
  - prior_mode swept over ["uniform", "balanced"] (same PRIOR_MODES list),
    jointly grid-searched with alpha (ALPHA_GRID) and gamma (GAMMA_GRID),
    selected by inner-CV macro-F1 -- identical selection rule and identical
    default grids to RDA_Test.py:
        ALPHA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        GAMMA_GRID = [0.0, 0.05, 0.1, 0.2, 0.4]
  - same 5 seeds [41, 42, 43, 44, 45], same cv_folds default of 5 for AMP-SCZ
    (RDA_Test.py's comment: "AMP-SCZ has hundreds of subjects per split ...
    the default 5-fold CV ... is appropriate here")
  - same clean_train_apply pipeline (median imputation, near-zero-variance
    filtering, StandardScaler fit on train only)
  - same per-seed / per-split / ensemble aggregation and mean+-std reporting

Adapted for this task (unchanged from the LDA version of this script):
  - LABEL = site, parsed from the alphabetic prefix of subj_ID (e.g.
    "BI07844" -> site "BI"), not the Group (HC/CHR) column.
  - Metrics generalized from binary to multiclass (macro precision/recall/F1
    unchanged in form; AUROC now uses multi_class="ovr"; AUPRC's one-hot +
    average="macro" computation was already multiclass-general in
    RDA_Test.py's metrics_6, so it's unchanged in form here).
  - This input file is already ONE ROW PER SUBJECT (globally-averaged
    spectral features, no windows), so there is no window subsampling step
    and no subject-level probability aggregation step (RDA_Test.py's
    aggregate_subject_probs is not needed here), and no train/val/test split
    files to read -- this script generates its own stratified train/test
    splits directly from the single input CSV, with the same 80/20 split
    ratio the paper's methodology uses elsewhere.
  - Sites are no longer dropped by default (--min_site_n=1 keeps every site
    and every subject). --cv_folds defaults to 5 (matching RDA_Test.py's
    AMP-SCZ default), but the smallest possible site (n=2 subjects total)
    may still not survive every fold after the outer 80/20 split -- this is
    expected, handled gracefully (see the deliberate deviation from
    RDA_Test.py's fold-skip rule, noted at cv_score below), and is not a
    crash.
  - DEVIATION FROM RDA_Test.py's cv_score: RDA_Test.py requires ALL
    N_CLASSES (there, just HC/CHR) to be present in a training fold before
    fitting -- reasonable for a fixed 2-class problem. Here, n_classes is
    the number of AMP-SCZ *sites* (dozens), and many sites have very few
    subjects, so requiring every site to appear in every fold would discard
    nearly all folds. This script keeps the original site-leakage script's
    more tolerant rule (fit as long as >=2 classes are present in the
    training fold; metrics_multiclass already handles a class being absent
    from y_true by restricting AUROC/AUPRC to the classes actually present
    -- see metrics_multiclass below). The classifier itself
    (RegularizedDiscriminantAnalysis), its alpha/gamma grid, and the
    prior_mode sweep are otherwise identical to RDA_Test.py.

Usage:
    python predict_site_from_spectral_rda.py \
        --csv SCZ_FFT_DWT_with_Group.csv \
        --n_splits 5 \
        --out_csv site_leakage_results_rda.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler

import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=LinAlgWarning)

# Expected/harmless: with small-sample classes (sites with only 1-2 total
# subjects), per-class covariance estimation is degenerate before shrinkage
# is applied. RegularizedDiscriminantAnalysis.fit already guards this
# (zero covariance for a 1-sample class, ridge fallback if a shrunk
# covariance isn't positive-definite -- see the class below), so these
# warnings add no diagnostic value here and are suppressed rather than left
# to flood the console.
warnings.filterwarnings(
    "ignore",
    message="Only one sample available.*",
    category=UserWarning,
)

METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1 score", "AUROC", "AUPRC"]
PRIOR_MODES  = ["uniform", "balanced"]

# Grid of (alpha, gamma) candidates searched inside inner CV -- identical
# defaults to RDA_Test.py's ALPHA_GRID / GAMMA_GRID.
ALPHA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]   # 0 = QDA-like, 1 = LDA-like
GAMMA_GRID = [0.0, 0.05, 0.1, 0.2, 0.4]        # shrinkage toward scaled identity


# Regularized Discriminant Analysis (Friedman, 1989) -- transplanted
# verbatim from RDA_Test.py. sklearn has no built-in RDA (only LDA and QDA
# at the alpha=1 / alpha=0 endpoints, each with sklearn's own single-knob
# shrinkage), so this is a minimal from-scratch implementation of the
# general two-knob version.
class RegularizedDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.5, gamma=0.1, priors=None):
        self.alpha  = alpha   # 0 = QDA, 1 = LDA
        self.gamma  = gamma   # shrinkage toward scaled identity
        self.priors = priors  # None -> empirical class frequencies

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_features    = X.shape[1]

        means, class_covs, class_counts = [], [], []
        for c in self.classes_:
            Xc = X[y == c]
            class_counts.append(Xc.shape[0])
            means.append(Xc.mean(axis=0))
            S = np.cov(Xc, rowvar=False, bias=False) if Xc.shape[0] > 1 \
                else np.zeros((n_features, n_features))
            class_covs.append(np.atleast_2d(S))
        self.means_  = np.array(means)
        class_counts = np.array(class_counts, dtype=float)

        # Pooled covariance: weighted by each class's degrees of freedom
        # (n_c - 1), same convention LDA uses for its shared covariance.
        pooled, total_dof = np.zeros((n_features, n_features)), 0.0
        for cnt, S in zip(class_counts, class_covs):
            dof = max(cnt - 1, 0)
            pooled += dof * S
            total_dof += dof
        pooled = pooled / max(total_dof, 1.0)

        self.covariances_ = []
        for S in class_covs:
            Sigma = (1 - self.alpha) * S + self.alpha * pooled
            avg_var = np.trace(Sigma) / n_features
            Sigma = (1 - self.gamma) * Sigma + self.gamma * avg_var * np.eye(n_features)
            self.covariances_.append(Sigma)

        if self.priors is None:
            self.priors_ = class_counts / class_counts.sum()
        else:
            self.priors_ = np.asarray(self.priors, dtype=float)

        # Precompute inverse / log-determinant per class for the discriminant.
        self._inv_cov, self._logdet = [], []
        for Sigma in self.covariances_:
            sign, logdet = np.linalg.slogdet(Sigma)
            if sign <= 0:
                # Not PD (can happen at low gamma with small class n) --
                # fall back to a small ridge rather than fail the fold.
                Sigma = Sigma + 1e-6 * np.eye(n_features)
                sign, logdet = np.linalg.slogdet(Sigma)
            self._logdet.append(logdet)
            self._inv_cov.append(np.linalg.inv(Sigma))

        return self

    def _log_joint(self, X):
        X = np.asarray(X, dtype=float)
        log_joint = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(len(self.classes_)):
            diff = X - self.means_[i]
            maha = np.einsum('ij,jk,ik->i', diff, self._inv_cov[i], diff)
            log_joint[:, i] = -0.5 * maha - 0.5 * self._logdet[i] + np.log(self.priors_[i])
        return log_joint

    def predict_proba(self, X):
        log_joint = self._log_joint(X)
        log_joint -= log_joint.max(axis=1, keepdims=True)  # softmax, stabilized
        probs = np.exp(log_joint)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# I/O
def read_csv_safely(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, encoding="latin1")


def parse_site(subj_id: str) -> str:
    """AMP-SCZ subj_ID -> site code: the alphabetic prefix, e.g. 'BI07844' -> 'BI'."""
    m = re.match(r'^([A-Za-z]+)', str(subj_id))
    return m.group(1) if m else str(subj_id)


# Cleaning + scaling -- identical to RDA_Test.py's clean_train_apply
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


def mean_std_str(vals):
    v    = np.asarray(vals, dtype=float)
    m, s = np.nanmean(v), (np.nanstd(v, ddof=1) if np.sum(~np.isnan(v)) > 1 else 0.0)
    return f"{100*m:.2f}±{100*s:.2f}"


def make_rda(ytr, prior_mode, alpha, gamma, n_classes):
    """Same RegularizedDiscriminantAnalysis construction as RDA_Test.py's
    make_rda, generalized to n_classes (was hardcoded to N_CLASSES=2 for
    HC/CHR; here n_classes = n_sites). The balanced-prior divide-by-zero
    guard (absent class in this fold) is carried over from this script's
    original make_lda, since -- unlike RDA_Test.py's fixed 2-class problem
    -- a site class can legitimately be absent from a given fold here."""
    if prior_mode == "uniform":
        priors = np.ones(n_classes) / n_classes
    elif prior_mode == "balanced":
        counts = np.bincount(ytr, minlength=n_classes).astype(float)
        counts[counts == 0] = 1.0  # guard divide-by-zero for an absent class in this fold
        priors = (1.0 / counts) / (1.0 / counts).sum()
    else:
        raise ValueError(f"Unknown prior_mode: {prior_mode}")
    return RegularizedDiscriminantAnalysis(alpha=alpha, gamma=gamma, priors=priors)


def metrics_multiclass(y_true, y_pred, y_proba, n_classes):
    """Same metric set/formulas as RDA_Test.py's metrics_6, generalized
    from binary (HC/CHR) to macro-averaged multiclass (site)."""
    out = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro",
                                     labels=list(range(n_classes)), zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average="macro",
                                  labels=list(range(n_classes)), zero_division=0),
        "F1 score":  f1_score(y_true, y_pred, average="macro",
                              labels=list(range(n_classes)), zero_division=0),
    }

    try:
        # multiclass AUROC needs the full probability matrix + multi_class="ovr"
        # (RDA_Test.py's binary version just took y_proba[:, 1]).
        #
        # When a site class never appears in y_true for this fold/split (which
        # can happen once tiny sites aren't dropped -- see --min_site_n), its
        # one-vs-rest sub-score is undefined and silently poisons the whole
        # macro average to NaN. Fix: restrict the AUROC computation to only
        # the classes actually present in y_true, renormalizing their
        # probability columns to sum to 1 per row (required by sklearn for
        # multiclass roc_auc_score). This changes what "macro" is averaged
        # over on any fold where a class is absent -- fewer classes on those
        # folds -- which is why it's flagged in the returned dict via
        # "AUROC_n_classes_present" rather than applied silently.
        present = np.unique(y_true)
        if len(present) < 2:
            out["AUROC"] = np.nan
        else:
            proba_present = y_proba[:, present]
            row_sums = proba_present.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0  # guard divide-by-zero
            proba_present = proba_present / row_sums
            out["AUROC"] = roc_auc_score(
                y_true, proba_present, multi_class="ovr", average="macro",
                labels=present,
            )
        out["AUROC_n_classes_present"] = len(present)
    except Exception:
        out["AUROC"] = np.nan
        out["AUROC_n_classes_present"] = len(np.unique(y_true))

    try:
        # Same root cause as the AUROC fix above: a site class absent from
        # y_true for this fold/split has an all-zero one-hot column, which
        # average_precision_score scores as a trivial 1.0 ("no positives to
        # miss") rather than NaN -- silently inflating the macro average with
        # "free" perfect scores for classes that were never actually tested.
        # Fix: restrict to the classes actually present in y_true, same as
        # AUROC (no renormalization needed here -- average_precision_score
        # scores each one-hot column independently, unlike roc_auc_score's
        # multiclass mode which requires rows to sum to 1).
        present_prc = np.unique(y_true)
        y_true_onehot = np.eye(n_classes)[y_true][:, present_prc]
        proba_present_prc = y_proba[:, present_prc]
        out["AUPRC"] = average_precision_score(y_true_onehot, proba_present_prc, average="macro")
        out["AUPRC_n_classes_present"] = len(present_prc)
    except Exception:
        out["AUPRC"] = np.nan
        out["AUPRC_n_classes_present"] = len(np.unique(y_true))

    return out


# Inner CV -- joint grid search over prior_mode x alpha x gamma, identical
# grids/selection rule to RDA_Test.py's cv_score, but with this script's
# original (more tolerant) fold-skip rule -- see the DEVIATION note in the
# module docstring.
def cv_score(Xtr_full, ytr, subj_tr, feat_cols, splits, n_classes,
             alpha_grid, gamma_grid):
    best_metrics = None
    best_score   = -np.inf
    best_prior   = None
    best_alpha   = None
    best_gamma   = None

    for prior_mode in PRIOR_MODES:
        for alpha in alpha_grid:
            for gamma in gamma_grid:
                fold_metrics = []

                for tr_idx, va_idx in splits:
                    Xtr_f, Xva_f = Xtr_full[tr_idx], Xtr_full[va_idx]
                    ytr_f, yva_f = ytr[tr_idx],       ytr[va_idx]

                    if len(np.unique(ytr_f)) < 2:
                        continue

                    try:
                        Xtr_c, Xva_c, _ = clean_train_apply(Xtr_f, Xva_f, feat_cols)
                        model            = make_rda(ytr_f, prior_mode, alpha, gamma, n_classes)
                        model.fit(Xtr_c, ytr_f)
                    except Exception as e:
                        print(f"[FOLD FAIL] {type(e).__name__}: {e}")
                        continue

                    proba_va = model.predict_proba(Xva_c)
                    # model.classes_ may be a subset of range(n_classes) if a
                    # class was absent from this training fold -- expand
                    # proba back to the full n_classes width (0 prob for
                    # absent classes) so metrics_multiclass's fixed-width
                    # AUROC/AUPRC math stays valid.
                    proba_full = np.zeros((proba_va.shape[0], n_classes))
                    proba_full[:, model.classes_] = proba_va
                    y_pred_va = np.argmax(proba_full, axis=1)

                    if len(np.unique(yva_f)) < 2:
                        continue

                    fold_metrics.append(
                        metrics_multiclass(yva_f, y_pred_va, proba_full, n_classes)
                    )

                if not fold_metrics:
                    continue

                mean_f1 = float(np.nanmean([m["F1 score"] for m in fold_metrics]))
                if mean_f1 > best_score:
                    best_score   = mean_f1
                    best_prior   = prior_mode
                    best_alpha   = alpha
                    best_gamma   = gamma
                    best_metrics = {k: float(np.nanmean([m[k] for m in fold_metrics]))
                                    for k in METRIC_NAMES}

    return best_metrics, best_prior, best_alpha, best_gamma


def run_one_split(Xtr, ytr, subj_tr, Xte, yte, subj_te, feat_cols, n_classes, args):
    all_rows              = []
    test_metrics_per_seed  = {m: [] for m in METRIC_NAMES}
    ensemble_proba_accum   = None
    ensemble_trues         = None

    for seed in args.seeds:
        cv     = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=seed)
        splits = list(cv.split(Xtr, ytr, groups=subj_tr))

        cv_m, best_prior, best_alpha, best_gamma = cv_score(
            Xtr, ytr, subj_tr, feat_cols, splits, n_classes,
            args.rda_alpha_grid, args.rda_gamma_grid,
        )
        if cv_m is None:
            raise RuntimeError(f"Inner CV failed for seed={seed}.")

        Xtr_c, Xte_c, _ = clean_train_apply(Xtr, Xte, feat_cols)
        model            = make_rda(ytr, best_prior, best_alpha, best_gamma, n_classes)
        model.fit(Xtr_c, ytr)
        proba_te         = model.predict_proba(Xte_c)

        proba_full = np.zeros((proba_te.shape[0], n_classes))
        proba_full[:, model.classes_] = proba_te
        y_pred_te = np.argmax(proba_full, axis=1)

        test_m = metrics_multiclass(yte, y_pred_te, proba_full, n_classes)

        if ensemble_proba_accum is None:
            ensemble_proba_accum = proba_full.astype(np.float64).copy()
            ensemble_trues       = yte.copy()
        else:
            ensemble_proba_accum += proba_full.astype(np.float64)

        for m in METRIC_NAMES:
            test_metrics_per_seed[m].append(test_m[m])

        row = {"seed": seed, "best_prior": best_prior, "best_alpha": best_alpha,
               "best_gamma": best_gamma, "cv_F1": cv_m["F1 score"]}
        for m in METRIC_NAMES:
            row[f"cv_{m}"]   = cv_m[m]
            row[f"test_{m}"] = test_m[m]
        row["test_AUROC_n_classes_present"] = test_m.get("AUROC_n_classes_present", None)
        row["test_AUPRC_n_classes_present"] = test_m.get("AUPRC_n_classes_present", None)
        all_rows.append(row)

        print(f"[seed={seed}]  prior={best_prior}  alpha={best_alpha}  gamma={best_gamma}  "
              f"inner-CV F1={cv_m['F1 score']:.4f}  "
              f"test F1={test_m['F1 score']:.4f}  test Acc={test_m['Accuracy']:.4f}  "
              f"test AUROC={test_m['AUROC']:.4f} ({test_m['AUROC_n_classes_present']}/{n_classes} classes)  "
              f"test AUPRC={test_m['AUPRC']:.4f} ({test_m['AUPRC_n_classes_present']}/{n_classes} classes)")

    print("\n--- TEST mean ± std across seeds ---")
    for m in METRIC_NAMES:
        print(f"  {m}: {mean_std_str(test_metrics_per_seed[m])}")

    ensemble_metrics = None
    if ensemble_proba_accum is not None:
        proba_avg        = ensemble_proba_accum / len(args.seeds)
        ensemble_preds   = np.argmax(proba_avg, axis=1)
        ensemble_metrics = metrics_multiclass(ensemble_trues, ensemble_preds, proba_avg, n_classes)

        print("\n--- ENSEMBLE TEST ---")
        for m in METRIC_NAMES:
            print(f"  {m}: {100*ensemble_metrics[m]:.2f}")
        print(f"  (AUROC computed over {ensemble_metrics['AUROC_n_classes_present']}/{n_classes} "
              f"classes present; AUPRC over {ensemble_metrics['AUPRC_n_classes_present']}/{n_classes} "
              f"classes present in the pooled test set)")

        ens_row = {"seed": "ensemble", "best_prior": None, "best_alpha": None,
                   "best_gamma": None, "cv_F1": None}
        for m in METRIC_NAMES:
            ens_row[f"cv_{m}"]   = None
            ens_row[f"test_{m}"] = ensemble_metrics[m]
        all_rows.append(ens_row)

    seed_means = {m: float(np.nanmean(test_metrics_per_seed[m])) for m in METRIC_NAMES}
    return all_rows, seed_means, ensemble_metrics


def main():
    ap = argparse.ArgumentParser(
        description="Site-leakage control: predict AMP-SCZ site from spectral "
                    "features using RDA_Test.py's exact classifier config "
                    "(RegularizedDiscriminantAnalysis, alpha/gamma grid, "
                    "prior_mode sweep)."
    )
    ap.add_argument("--csv", required=True, help="Path to SCZ_FFT_DWT_with_Group.csv "
                    "(one row per subject, globally-averaged spectral features).")
    ap.add_argument("--subject_col", default="subj_ID")
    ap.add_argument("--n_splits",    type=int, default=5,
                    help="Number of independent stratified 80/20 train/test "
                         "splits (mirrors the paper's 5-split methodology).")
    ap.add_argument("--test_size",   type=float, default=0.20)
    ap.add_argument("--seeds",       nargs="*", type=int, default=[41, 42, 43, 44, 45])
    ap.add_argument("--cv_folds",    type=int,  default=5,
                    help="Inner CV fold count. Defaults to 5, matching "
                         "RDA_Test.py's AMP-SCZ default ('hundreds of "
                         "subjects per split ... the default 5-fold CV is "
                         "appropriate here'). Lower this if very small "
                         "sites cause too many folds to be skipped (see "
                         "--min_site_n).")
    ap.add_argument("--min_site_n",  type=int,  default=1,
                    help="Minimum subjects a site needs to be included. "
                         "Default is 1 -- i.e. no site is dropped. Note that "
                         "very small sites (e.g. n=2-3) will still trigger "
                         "sklearn UserWarnings and may not appear in every "
                         "CV fold's training data; this is expected and "
                         "handled gracefully (that fold/class combination is "
                         "simply skipped), not an error.")
    ap.add_argument("--rda_alpha_grid", type=float, nargs="*", default=ALPHA_GRID,
                    help="Candidate alpha values (0=QDA-like, 1=LDA-like) "
                         "searched inside inner CV, jointly with prior_mode "
                         "and gamma -- identical default grid to "
                         "RDA_Test.py.")
    ap.add_argument("--rda_gamma_grid", type=float, nargs="*", default=GAMMA_GRID,
                    help="Candidate gamma values (shrinkage toward scaled "
                         "identity) searched inside inner CV alongside "
                         "alpha and prior_mode -- identical default grid to "
                         "RDA_Test.py.")
    ap.add_argument("--out_csv",     default="site_leakage_results_rda.csv")
    args = ap.parse_args()

    min_site_n = args.min_site_n

    df = read_csv_safely(Path(args.csv))
    df.columns = df.columns.str.strip()
    subject_col = args.subject_col.strip()

    if subject_col not in df.columns:
        raise ValueError(f"Missing subject column '{subject_col}'")

    df["site"] = df[subject_col].apply(parse_site)

    site_counts = df["site"].value_counts()
    keep_sites  = site_counts[site_counts >= min_site_n].index
    dropped     = site_counts[site_counts < min_site_n]

    print(f"[INFO] {df['site'].nunique()} total sites found; "
          f"{len(keep_sites)} retained with >= {min_site_n} subjects each")
    if len(dropped):
        print(f"[INFO] Dropped {len(dropped)} sites with < {min_site_n} subjects "
              f"({int(dropped.sum())} subjects total): "
              f"{dict(dropped.sort_values(ascending=False))}")
    else:
        print(f"[INFO] No sites dropped -- all {df['site'].nunique()} sites and "
              f"{len(df)} subjects retained.")

    df = df[df["site"].isin(keep_sites)].reset_index(drop=True)

    site_to_idx = {s: i for i, s in enumerate(sorted(keep_sites))}
    n_classes   = len(site_to_idx)
    y_all       = df["site"].map(site_to_idx).to_numpy()
    subj_all    = df[subject_col].astype(str).to_numpy()

    # Exclude non-numeric / identifier columns (subj_ID, Group, site) from features.
    exclude_cols = {subject_col, "site", "Group"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols    = [c for c in numeric_cols if c not in exclude_cols]
    X_all        = df[feat_cols].to_numpy(dtype=float)

    print(f"[INFO] {len(df)} subjects | {n_classes} site classes | {len(feat_cols)} features")
    print(f"[INFO] Chance-level accuracy (uniform)   = {100.0/n_classes:.2f}%")
    majority_frac = df["site"].value_counts(normalize=True).max()
    print(f"[INFO] Majority-class baseline accuracy  = {100.0*majority_frac:.2f}%")

    all_rows            = []
    per_split_means      = {m: [] for m in METRIC_NAMES}
    per_split_ensemble   = []

    for split_i in range(args.n_splits):
        print(f"\n{'='*50}\nsplit_{split_i + 1}\n{'='*50}")

        tr_idx, te_idx = train_test_split(
            np.arange(len(df)), test_size=args.test_size,
            stratify=y_all, random_state=split_i,
        )

        Xtr, ytr, subj_tr = X_all[tr_idx], y_all[tr_idx], subj_all[tr_idx]
        Xte, yte, subj_te = X_all[te_idx], y_all[te_idx], subj_all[te_idx]

        print(f"Train subjects: {len(tr_idx)}  |  Test subjects: {len(te_idx)}")

        rows, seed_means, ens_m = run_one_split(
            Xtr, ytr, subj_tr, Xte, yte, subj_te, feat_cols, n_classes, args
        )
        for r in rows:
            r["split"] = f"split_{split_i + 1}"
        all_rows.extend(rows)

        for m in METRIC_NAMES:
            per_split_means[m].append(seed_means[m])
        if ens_m:
            per_split_ensemble.append({"split": f"split_{split_i + 1}", **ens_m})

    out_path = Path(args.out_csv)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)

    print("\n=== OVERALL TEST (mean ± std across splits) ===")
    for m in METRIC_NAMES:
        print(f"  {m}: {mean_std_str(per_split_means[m])}")

    if per_split_ensemble:
        print("\n=== ENSEMBLE TEST (mean ± std across splits) ===")
        for m in METRIC_NAMES:
            vals = [r[m] for r in per_split_ensemble if r.get(m) is not None]
            print(f"  {m}: {mean_std_str(vals)}")

    print(f"\n[INFO] Chance-level accuracy   = {100.0/n_classes:.2f}%")
    print(f"[INFO] Majority-class baseline = {100.0*majority_frac:.2f}%")
    print("[INFO] If overall test accuracy is close to these baselines, "
          "there is no meaningful site signature in the spectral features.")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()