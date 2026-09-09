import argparse
import os
import time
import torch
import pandas as pd
from exp.exp_classification import Exp_Classification
import random
import numpy as np
import psutil


def use_cpus(gpus: list, cpus_per_gpu: int):
    cpus = []
    for gpu in gpus:
        cpus.extend(list(range(gpu * cpus_per_gpu, (gpu + 1) * cpus_per_gpu)))
    p = psutil.Process()
    p.cpu_affinity(cpus)
    print(
        "A total {} CPUs are used, making sure that num_worker is small than the number of CPUs".format(
            len(cpus)
        )
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TeCh")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="TeCh",
        help="model name, options: [Autoformer, Medformer, TeCh]",
    )

    parser.add_argument(
        "--split_json",
        type=str,
        default=None,
        help="Path to JSON file with subject splits (train/val/test).",
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
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    # model define for baselines
    parser.add_argument(
        "--patch_len", type=int, default=1, help="for cross_channel patching"
    )
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument(
        "--d_model", type=int, default=512, help="dimension of model"
    )
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument(
        "--t_layer", type=int, default=6, help="num of encoder layers"
    )
    parser.add_argument(
        "--v_layer", type=int, default=6, help="num of encoder layers"
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout")

    # Augmentation
    parser.add_argument(
        "--augmentations",
        type=str,
        default="flip0.8,frequency0.,jitter0.,mask0.0,channel0.4,drop0.0",
        help="A comma-separated list of augmentation types (none, jitter or scale). "
        "Randomly applied to each granularity. "
        "Append numbers to specify the strength of the augmentation, e.g., jitter0.1",
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=16, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument(
        "--train_epochs", type=int, default=10, help="train epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="optimizer learning rate",
    )
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="cosine", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # GPU
    parser.add_argument(
        "--gpu_idx",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7],
        help="List of GPU indices to use",
    )
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu",
        action="store_true",
        help="use multiple gpus",
        default=False,
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multiple gpus"
    )

    # ------------------------------------------------------------------
    # Runtime benchmarking (new). No confusion-matrix step exists anywhere
    # in Exp_Classification (only Accuracy/Precision/Recall/F1/AUROC/AUPRC
    # are computed), so there is nothing to exclude on that front here.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--runtime_csv",
        type=str,
        default="tech_pipeline_runtimes.csv",
        help="Where to save per-split wall-clock runtimes.",
    )

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # For server 1, set cpus_per_gpu to 12
    # For server 2, set cpus_per_gpu to 24
    # use_cpus(gpus=args.gpu_idx, cpus_per_gpu=12)
    
    Exp = Exp_Classification

    metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC"]

    # Number of stratified splits available per dataset. Most datasets have
    # 5 splits, but some (e.g. APAVA) only have 3 due to fewer subjects.
    DATASET_NUM_SPLITS = {
        "APAVA": 3,
    }
    num_splits = DATASET_NUM_SPLITS.get(args.data, 5)

    # Choose whether to run one split or all available splits for this dataset
    split_ids = range(1, num_splits + 1) if args.run_all_splits else [args.split_id]

    # Per-split wall-clock runtimes, in the same spirit as the LDA/spectral
    # pipeline benchmark: one number per split, covering the model starting
    # to run through that split being fully processed across all seeds.
    split_runtimes = []

    for split_id in split_ids:
        args.split_id = split_id

        print("\n" + "=" * 80)
        print(f"RUNNING STRATIFIED SPLIT {args.split_id}")
        print("=" * 80 + "\n")

        avg_metrics = []

        # ================= START TIMER =================
        # Model starts running (instantiation -> train -> test) for the
        # first seed of this split.
        split_start = time.perf_counter()

        for ii in range(args.itr):
            seed = 42 + ii

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
            setting = (
                "{}_{}_split_{}_seed_{}_dm_{}_dp_{}_tl_{}_vl_{}_bs_{}_lr{}_aug_{}_pl_{}".format(
                    args.model,
                    args.data,
                    args.split_id,
                    args.seed,
                    args.d_model,
                    args.dropout,
                    args.t_layer,
                    args.v_layer,
                    args.batch_size,
                    args.learning_rate,
                    args.augmentations,
                    args.patch_len,
                )
            )

            exp = Exp(args)

            print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
            exp.train(setting)

            print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
            avg_metrics.append(exp.test(setting))

            torch.cuda.empty_cache()

        # ================== END TIMER ===================
        # This split has now been completely processed across all
        # args.itr seeds.
        split_elapsed = time.perf_counter() - split_start
        print(f"\n[TIMER] Split {args.split_id} runtime "
              f"({args.itr} seed(s)): {split_elapsed:.2f} s")
        split_runtimes.append({"split": f"split_{args.split_id}", "runtime_seconds": split_elapsed})

        # ============================================================
        # Aggregate metrics across iterations for this split
        # ============================================================

        means = [
            np.mean([avg_metrics[i][metric] for i in range(args.itr)])
            for metric in metric_names
        ]

        stds = [
            np.std([avg_metrics[i][metric] for i in range(args.itr)])
            for metric in metric_names
        ]

        print(f"\n---------------- SPLIT {args.split_id} RESULTS ----------------")

        print(
            f"Mean accuracy: {means[0]:.4f}, "
            f"precision: {means[1]:.4f}, "
            f"recall: {means[2]:.4f}, "
            f"f1: {means[3]:.4f}, "
            f"AUROC: {means[4]:.4f}, "
            f"AUPRC: {means[5]:.4f}"
        )

        print(
            f"Std accuracy: {stds[0]:.4f}, "
            f"precision: {stds[1]:.4f}, "
            f"recall: {stds[2]:.4f}, "
            f"f1: {stds[3]:.4f}, "
            f"AUROC: {stds[4]:.4f}, "
            f"AUPRC: {stds[5]:.4f}"
        )

        # ============================================================
        # Save results to:
        # Results/<Dataset Name>/split_<split_id>_metrics.csv
        # ============================================================

        results_root = "Results/DS_TUNED_Weighted_LOSS/C*5_Random_BANDS/"
        dataset_results_dir = os.path.join(results_root, args.data)
        os.makedirs(dataset_results_dir, exist_ok=True)

        split_name = f"split_{args.split_id}"

        result_row = {
            "dataset": args.data,
            "split": split_name,
            "model": args.model,
            "seed_start": 42,
            "itr": args.itr,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "d_model": args.d_model,
            "dropout": args.dropout,
            "t_layer": args.t_layer,
            "v_layer": args.v_layer,
            "patch_len": args.patch_len,
            "augmentations": args.augmentations,
        }

        for metric, mean_value, std_value in zip(metric_names, means, stds):
            result_row[f"mean_{metric}"] = mean_value
            result_row[f"std_{metric}"] = std_value
            result_row[f"mean_{metric}_percent"] = mean_value * 100.0
            result_row[f"std_{metric}_percent"] = std_value * 100.0

        result_df = pd.DataFrame([result_row])

        split_result_path = os.path.join(
            dataset_results_dir, f"{split_name}_metrics.csv"
        )

        result_df.to_csv(split_result_path, index=False)

        print("\nSaved split result CSV:")
        print(split_result_path)

        # ============================================================
        # Also maintain one combined CSV for all splits of the dataset
        # Results/<Dataset Name>/all_splits_metrics.csv
        # ============================================================

        combined_result_path = os.path.join(
            dataset_results_dir, "all_splits_metrics.csv"
        )

        if os.path.exists(combined_result_path):
            old_df = pd.read_csv(combined_result_path)

            duplicate_mask = (
                (old_df["dataset"] == args.data)
                & (old_df["split"] == split_name)
                & (old_df["model"] == args.model)
                & (old_df["batch_size"] == args.batch_size)
                & (old_df["learning_rate"] == args.learning_rate)
                & (old_df["d_model"] == args.d_model)
                & (old_df["dropout"] == args.dropout)
                & (old_df["t_layer"] == args.t_layer)
                & (old_df["v_layer"] == args.v_layer)
                & (old_df["patch_len"] == args.patch_len)
                & (old_df["augmentations"] == args.augmentations)
            )

            old_df = old_df.loc[~duplicate_mask].copy()
            combined_df = pd.concat([old_df, result_df], axis=0, ignore_index=True)

        else:
            combined_df = result_df.copy()

        combined_df = combined_df.sort_values(
            by=["dataset", "model", "split"]
        ).reset_index(drop=True)

        combined_df.to_csv(combined_result_path, index=False)

        print("Updated combined result CSV:")
        print(combined_result_path)

    # ============================================================
    # Runtime summary across all processed splits (model-start ->
    # split-fully-processed-by-all-seeds, per split). No confusion-matrix
    # generation is included anywhere above.
    # ============================================================

    runtime_df = pd.DataFrame(split_runtimes)

    avg_runtime = runtime_df["runtime_seconds"].mean()
    std_runtime = runtime_df["runtime_seconds"].std(ddof=1) if len(runtime_df) > 1 else 0.0
    total_runtime = runtime_df["runtime_seconds"].sum()

    print(f"\n{'=' * 60}")
    print("Per-split runtime:")
    for _, r in runtime_df.iterrows():
        print(f"  {r['split']}: {r['runtime_seconds']:.2f} s")
    print(f"\nAverage runtime per split (across {len(runtime_df)} split(s)): "
          f"{avg_runtime:.2f} +/- {std_runtime:.2f} s")
    print(f"TOTAL runtime aggregated across all {len(runtime_df)} split(s) "
          f"x {args.itr} seed(s): {total_runtime:.2f} s")

    # Append a TOTAL row so the saved CSV carries the aggregate number too,
    # not just the per-split breakdown.
    runtime_df = pd.concat(
        [runtime_df, pd.DataFrame([{"split": "TOTAL", "runtime_seconds": total_runtime}])],
        ignore_index=True,
    )
    runtime_df.to_csv(args.runtime_csv, index=False)
    print(f"Saved: {args.runtime_csv}")