import pandas as pd

files = {
    "ADFTD": "ADFTD_Runtime.csv",
    "AMP-SCZ": "AmpScz_TeCh_runtimes.csv",
    "APAVA": "APAVA_Timing.csv",
    "TDBrain": "TDBrain_Timing_Formal.csv",
}

for dataset, file_path in files.items():
    df = pd.read_csv(file_path)

    # Keep only actual split rows, excluding TOTAL
    df = df[df["split"].str.startswith("split_", na=False)]

    mean_runtime = df["runtime_seconds"].mean()
    std_runtime = df["runtime_seconds"].std()

    print(f"{dataset}")
    print(f"  Mean runtime: {mean_runtime:.2f} seconds")
    print(f"  Std runtime : {std_runtime:.2f} seconds")
    print(f"  Mean runtime: {mean_runtime / 60:.2f} minutes")
    print()