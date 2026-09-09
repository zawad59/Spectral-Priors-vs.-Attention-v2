import argparse
import csv
import os
import sys
import time
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Confusion-matrix helpers
# -------------------------------------------------------

def _normalize_cm(raw_cm: np.ndarray, normalize: str) -> np.ndarray:
    """
    Normalize a raw (integer) confusion matrix.

    normalize : "true"  -> row-wise   (recall per class)
                "pred"  -> col-wise   (precision per class)
                "all"   -> global sum
                "none"  -> no normalization (return float copy)
    """
    cm = raw_cm.astype(float)
    if normalize == "true":
        denom = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, denom, out=np.zeros_like(cm), where=denom != 0)
    elif normalize == "pred":
        denom = cm.sum(axis=0, keepdims=True)
        cm = np.divide(cm, denom, out=np.zeros_like(cm), where=denom != 0)
    elif normalize == "all":
        denom = cm.sum()
        cm = cm / denom if denom != 0 else cm
    return cm


def _save_cm_png(
    cm_normalized: np.ndarray,
    out_png: str,
    class_names,
    cmap: str = "viridis",
    title: str = "",
):

    n = cm_normalized.shape[0]
    figsize = max(3.2, n * 1.1)
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)

    names = class_names if class_names else [str(i) for i in range(n)]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    thresh = (cm_normalized.max() + cm_normalized.min()) / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_normalized[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", fontsize=15,
                color="white" if val > thresh else "black",
            )

    os.makedirs(os.path.dirname(out_png) if os.path.dirname(out_png) else ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[CM] Saved: {out_png}")


def _save_timing_csv(out_csv: str, row: dict) -> None:
    """
    Writes a single-row runtime CSV for the timed (split, seed) run.
    Creates parent dirs as needed; overwrites any existing file at
    out_csv (this benchmark only ever times one (split, seed) per run,
    so there is exactly one row to write).
    """
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"[TIMING] Saved: {out_csv}")


# -------------------------------------------------------

if __name__ == "__main__":
    """fix_seed = 42
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)"""

    parser = argparse.ArgumentParser(description="TimesNet")

    # basic config
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        default="long_term_forecast",
        help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
    )
    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="status"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, default="test", help="model id"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="Autoformer",
        help="model name, options: [Autoformer, Transformer, TimesNet]",
    )

    parser.add_argument("--visualize_attn", action="store_true")
    parser.add_argument("--attn_out_dir", type=str, default="./attn_viz")
    parser.add_argument("--attn_per_class_examples", type=int, default=1)

    parser.add_argument("--plot_cm", action="store_true", default=False)
    parser.add_argument("--cm_out_dir", type=str, default="./cm_heatmaps")
    parser.add_argument("--cm_normalize", type=str, default="true", choices=["none","true","pred","all"])
    parser.add_argument("--cm_cmap", type=str, default="viridis")
    parser.add_argument("--class_names", nargs="*", default=None)  # e.g. A C F
    
    parser.add_argument(
        "--split_id",
        type=int,
        default=1,
        help="Which fixed stratified split to use: 1, 2, 3, 4, or 5.",
    )

    parser.add_argument(
    "--run_all_splits",
    action="store_true",
    default=False,
    help="If set, automatically run split_id from 1 to 5.",
)

    parser.add_argument(
        "--stratified_splits_dir",
        type=str,
        default="stratified_hold_outs",
        help="Folder inside dataset root containing split_1, ..., split_5.",
    )

    parser.add_argument(
        "--split_subject_col",
        type=str,
        default="subj_ID",
        help="Subject ID column name in train_subject_ids.csv / val_subject_ids.csv / test_subject_ids.csv.",
    )

    parser.add_argument(
    "--split_json",
    type=str,
    default=None,
    help="Path to JSON file with subject splits (train/val/test)."
)

    parser.add_argument(
        "--timing_out_csv",
        type=str,
        default="medformer_runtime.csv",
        help="Where to write the single-(split,seed) runtime CSV. Timing "
             "covers exactly one run -- split_id=1, seed=41 (the first "
             "split/seed combination processed) -- starting right before "
             "training begins for that run and stopping right after "
             "testing/result-writing finishes for it. Every other split "
             "and seed still runs as before, just untimed.",
    )

    # data loader
    parser.add_argument(
        "--data", type=str, required=True, default="ETTm1", help="dataset type"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )
    parser.add_argument(
        "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
    )
    parser.add_argument(
        "--inverse", action="store_true", help="inverse output data", default=False
    )

    # inputation task
    parser.add_argument("--mask_rate", type=float, default=0.25, help="mask ratio")

    # anomaly detection task
    parser.add_argument(
       "--anomaly_ratio", type=float, default=0.25, help="prior anomaly ratio (%%)"
    )

    # model define for baselines
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in encoder",
    )
    parser.add_argument(
        "--no_inter_attn",
        action="store_true",
        help="whether to use inter-attention in encoder, using this argument means not using inter-attention",
        default=False,
    )
    parser.add_argument(
        "--chunk_size", type=int, default=16, help="chunk_size used in LightTS"
    )
    parser.add_argument(
        "--patch_len", type=int, default=16, help="patch_len used in PatchTST"
    )
    parser.add_argument("--stride", type=int, default=8, help="stride used in PatchTST")
    parser.add_argument(
        "--sampling_rate", type=int, default=256, help="frequency sampling rate"
    )
    parser.add_argument(
        "--patch_len_list",
        type=str,
        default="2,4,8",
        help="a list of patch len used in Medformer",
    )
    parser.add_argument(
        "--single_channel",
        action="store_true",
        help="whether to use single channel patching for Medformer",
        default=False,
    )

    
    parser.add_argument(
        "--augmentations",
        type=str,
        default="flip,shuffle,frequency,jitter,mask,drop",
        help="A comma-seperated list of augmentation types (none, jitter or scale). "
             "Randomly applied to each granularity. "
             "Append numbers to specify the strength of the augmentation, e.g., jitter0.1",
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=0, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )
    parser.add_argument(
        "--swa",
        action="store_true",
        help="use stochastic weight averaging",
        default=False,
    )

    parser.add_argument("--finetune_from", type=str, default="",
                    help="Path to pretrained checkpoint (.pth)")

    parser.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze almost everything for finetuning")

    parser.add_argument("--unfreeze_patch_embed", action="store_true",
                    help="Unfreeze patch embedding stem (required if enc_in differs)")

    parser.add_argument("--unfreeze_last_k_blocks", type=int, default=0,
                    help="Unfreeze last K transformer blocks too")

    parser.add_argument("--reset_head", action="store_true",
                    help="Reinit classification head (recommended if num_class/seq_len differ)")

    parser.add_argument("--psd_n_fft", type=int, default=256)
    parser.add_argument("--psd_fmin", type=float, default=0.5)
    parser.add_argument("--psd_fmax", type=float, default=45.0)
    parser.add_argument("--psd_log_power", action="store_true", default=True)
    parser.add_argument("--psd_eps", type=float, default=1e-8)

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multiple gpus"
    )

    parser.add_argument("--weight_decay", type=float, default=1e-4,
                    help="AdamW weight decay (L2).")
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                    help="CrossEntropy label smoothing in [0,1).")

    # de-stationary projector params
    parser.add_argument(
        "--p_hidden_dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="hidden layer dimensions of projector (List)",
    )
    parser.add_argument(
        "--p_hidden_layers",
        type=int,
        default=2,
        help="number of hidden layers in projector",
    )

    parser.add_argument(
    "--subject_mil",
    action="store_true",
    default=False,
    help="Subject-level MIL: one subject = one sample (bag of windows).",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="",
        help="Optional explicit label file. If empty, auto-detect in root_path.",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="",
        help="Optional splits dir. If empty, auto-detect ids in root_path.",
    )

    # ---- covariance pooling model params ----
    parser.add_argument(
    "--cov_mode",
    type=str,
    default="feat13",
    choices=["feat13", "chan"],
    help="Covariance pooling mode for covariance models. "
         "'feat13': cov across 13 features (samples=N_win*C). "
         "'chan': cov across channels (samples=N_win*13).",
    )
    parser.add_argument(
        "--num_feat",
        type=int,
        default=13,
        help="Number of per-channel features (e.g., 13 FFT/DWT features).",
    )
    parser.add_argument(
        "--cov_eps",
        type=float,
        default=1e-3,
        help="Diagonal shrinkage added to covariance for numerical stability.",
    )

    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to checkpoint.pth for test")

    parser.add_argument(
        "--attn_mask_ranges",
        type=str,
        default="",
        help='Comma-separated time index ranges to mask at test-time, e.g. "0:8,23:31,248:256". Applied across all channels.',
    )
    parser.add_argument(
        "--attn_mask_value",
        type=float,
        default=0.0,
        help="Value to write into the masked time ranges (default 0.0).",
    )


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    if args.task_name == "long_term_forecast":
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == "short_term_forecast":
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == "imputation":
        Exp = Exp_Imputation
    elif args.task_name == "anomaly_detection":
        Exp = Exp_Anomaly_Detection
    elif args.task_name == "classification":
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    # -------------------------------------------------------
    # CM accumulator (only used for classification + plot_cm)
    # raw_cm_global : sum of raw counts across ALL splits × seeds
    # raw_cm_per_split[split_id] : sum of raw counts across seeds for that split
    # -------------------------------------------------------
    do_cm = (args.task_name == "classification") and args.plot_cm
    raw_cm_global = None          # np.ndarray, accumulated across everything
    raw_cm_per_split = {}         # split_id (int) -> np.ndarray

    # Choose whether to run one split or all 5 splits
    split_ids = range(1, 6) if args.run_all_splits else [args.split_id]

    # Runtime timing (Medformer): times exactly ONE (split, seed) run --
    # split_id=1, seed=41, i.e. the first split/seed combination processed
    # below (with default args this is also the only one that runs).
    # Clock starts right before training begins for that run and stops
    # right after testing/result-writing finishes for it. Every other
    # split/seed still runs exactly as before, just untimed.
    TIMED_SPLIT_ID = 1
    TIMED_SEED = 41
    _timed_run_done = False

    for split_id in split_ids:
        args.split_id = split_id

        print("\n" + "=" * 80)
        print(f"RUNNING STRATIFIED SPLIT {args.split_id}")
        print("=" * 80 + "\n")

        if args.is_training:
            for ii in range(args.itr):
                seed = 41 + ii

                random.seed(seed)
                os.environ["PYTHONHASHSEED"] = str(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                if args.model != "TCN":
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True

                args.seed = seed

                # IMPORTANT: include split_id in setting name to avoid overwriting checkpoints/logs
                setting = "{}_{}_{}_{}_split{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_seed{}".format(
                    args.task_name,
                    args.model_id,
                    args.model,
                    args.data,
                    args.split_id,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des,
                    args.seed,
                )

                exp = Exp(args)

                is_timed_run = (
                    not _timed_run_done
                    and split_id == TIMED_SPLIT_ID
                    and seed == TIMED_SEED
                )
                if is_timed_run:
                    _t0 = time.perf_counter()

                print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
                exp.train(setting)

                print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
                raw_cm = exp.test(setting)

                if is_timed_run:
                    _elapsed = time.perf_counter() - _t0
                    print(f"[TIMING] split_id={split_id} seed={seed} "
                          f"train+test runtime: {_elapsed:.2f} s")
                    _save_timing_csv(args.timing_out_csv, {
                        "model": args.model,
                        "split_id": split_id,
                        "seed": seed,
                        "runtime_seconds": _elapsed,
                    })
                    _timed_run_done = True

                if do_cm and raw_cm is not None:
                    if raw_cm_global is None:
                        raw_cm_global = raw_cm.copy()
                    else:
                        raw_cm_global += raw_cm

                    if split_id not in raw_cm_per_split:
                        raw_cm_per_split[split_id] = raw_cm.copy()
                    else:
                        raw_cm_per_split[split_id] += raw_cm

                torch.cuda.empty_cache()

                if is_timed_run:
                    print(f"[TIMING] Requested run (split_id={split_id}, "
                          f"seed={seed}) is done -- stopping now, not "
                          f"continuing to the remaining splits/seeds.")
                    sys.exit(0)

        else:
            for ii in range(args.itr):
                seed = 41 + ii

                random.seed(seed)
                os.environ["PYTHONHASHSEED"] = str(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                if args.model != "TCN":
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True

                args.seed = seed

                setting = "{}_{}_{}_{}_split{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_seed{}".format(
                    args.task_name,
                    args.model_id,
                    args.model,
                    args.data,
                    args.split_id,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des,
                    args.seed,
                )

                exp = Exp(args)

                is_timed_run = (
                    not _timed_run_done
                    and split_id == TIMED_SPLIT_ID
                    and seed == TIMED_SEED
                )
                if is_timed_run:
                    _t0 = time.perf_counter()

                print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
                raw_cm = exp.test(setting, test=1)

                if is_timed_run:
                    _elapsed = time.perf_counter() - _t0
                    print(f"[TIMING] split_id={split_id} seed={seed} "
                          f"test-only runtime: {_elapsed:.2f} s")
                    _save_timing_csv(args.timing_out_csv, {
                        "model": args.model,
                        "split_id": split_id,
                        "seed": seed,
                        "runtime_seconds": _elapsed,
                    })
                    _timed_run_done = True

                if do_cm and raw_cm is not None:
                    if raw_cm_global is None:
                        raw_cm_global = raw_cm.copy()
                    else:
                        raw_cm_global += raw_cm

                    if split_id not in raw_cm_per_split:
                        raw_cm_per_split[split_id] = raw_cm.copy()
                    else:
                        raw_cm_per_split[split_id] += raw_cm

                torch.cuda.empty_cache()

                if is_timed_run:
                    print(f"[TIMING] Requested run (split_id={split_id}, "
                          f"seed={seed}) is done -- stopping now, not "
                          f"continuing to the remaining splits/seeds.")
                    sys.exit(0)

        # -------------------------------------------------------
        # Save aggregated confusion matrices (classification only)
        #
        # Layout inside --cm_out_dir:
        #   <cm_out_dir>/
        #       split_<id>/
        #           cm_aggregated.png          <- seeds summed, then normalized
        #           cm_aggregated_raw.npy      <- raw integer counts (useful for debugging)
        #       overall/
        #           cm_aggregated.png          <- all splits × seeds summed, then normalized
        #           cm_aggregated_raw.npy
        # -------------------------------------------------------
    if do_cm:
        cm_root = args.cm_out_dir
        class_names = args.class_names  # may be None → numbered labels

        # Per-split aggregated CMs
        for split_id, raw_cm in raw_cm_per_split.items():
            split_dir = os.path.join(cm_root, f"split_{split_id}")
            os.makedirs(split_dir, exist_ok=True)

            # Save raw counts for reproducibility
            np.save(os.path.join(split_dir, "cm_aggregated_raw.npy"), raw_cm)

            # Normalize once over the summed counts and save PNG
            cm_norm = _normalize_cm(raw_cm, args.cm_normalize)
            _save_cm_png(
                cm_norm,
                out_png=os.path.join(split_dir, "cm_aggregated.png"),
                class_names=class_names,
                cmap=args.cm_cmap,
                title=f"Split {split_id} – seeds aggregated",
            )

        # Overall aggregated CM (all splits × all seeds)
        if raw_cm_global is not None:
            overall_dir = os.path.join(cm_root, "overall")
            os.makedirs(overall_dir, exist_ok=True)

            np.save(os.path.join(overall_dir, "cm_aggregated_raw.npy"), raw_cm_global)

            cm_norm_global = _normalize_cm(raw_cm_global, args.cm_normalize)
            _save_cm_png(
                cm_norm_global,
                out_png=os.path.join(overall_dir, "cm_aggregated.png"),
                class_names=class_names,
                cmap=args.cm_cmap,
                title="Overall – all splits & seeds aggregated",
            )