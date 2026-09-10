# Spectral Priors vs. Attention — Code Base (v2)

Code base for **"Spectral Priors vs. Attention: Investigating the Utility of
Attention Mechanisms in EEG-Based Diagnosis."** This repo contains all
pre-processing scripts, ML baselines, Transformer model changes, the
KL-Divergence and CRT ablation analyses, and the tensor-decomposition
(SBD/RBD) sensitivity experiments reported in the paper.

## Repository map

| Folder | Contents |
|---|---|
| `ADFTD`, `APAVA`, `TDBrain`, `AMP-SCZ` | Per-dataset spectral feature extraction (FFT/DWT) scripts and ML baselines (LDA/RDA) |
| `Common` | Dataset-agnostic utilities: stratified split generation, epoch counting, SBD/RBD tensor-dimension scripts |
| `Transformer_Changes` | Modified data loaders, models (Medformer, Reformer, Conformer, Transformer), training/eval entry point (`run.py`), CRT ablation results, and a separate `TeCh/` subtree |
| `KL_Divergence` | Post-hoc KL-divergence analysis over extracted attention weights |
| `Timing_Analysis` | Runtime-analysis variants of the Medformer/TeCh training scripts, plus per-dataset spectral-pipeline timing |

## Setup

```bash
git clone <repo-url>
cd CODEBASE
pip install -r requirements.txt
```

Each dataset script expects the raw dataset downloaded into its own folder
(see links below) before running the pipeline.

**Original datasets:** [ADFTD](https://openneuro.org/datasets/ds004504/versions/1.0.6/), [APAVA](https://osf.io/jbysn/overview/), [TDBrain](https://brainclinics.com/resources/), [AMP-SCZ](https://nda.nih.gov/ampscz)

## Pipeline

Run in this order for any dataset:

1. **Spectral feature extraction.** In the dataset's folder (e.g.
   `ADFTD/adftd_ff_dwt_from_derivatives.py`, `APAVA/APAVA_fft_dwt.py`,
   `TDBrain/TDBrain_fft_dwt.py`, `AMP-SCZ/AMP_SCZ_fft_dwt_PreProcessing.py`),
   run the FFT/DWT script against the downloaded raw dataset, filtering
   subjects as described in the paper.
2. **Stratified split generation.** Run `Common/gen_Stratified_Sets.py` on
   the extracted features to generate the 5 train/val/test stratified
   splits (`stratified_holdout_splits/`).
3. **ML baselines (LDA/RDA).** Pass the `stratified_holdout_splits`
   directory as the model param to the per-dataset `LDA_Test.py` /
   `RDA_Test.py` scripts to train/test the ML baselines.
4. **Transformer models.** Use `Transformer_Changes/run.py` for
   Medformer, Reformer, Conformer, and Transformer. The data loader
   (`Transformer_Changes/data_provider/data_loader.py`) and data factory
   (`Transformer_Changes/data_provider/data_factory.py`) have been
   modified with AMP-SCZ and TDBrain classes. `run.py` supports running
   across N stratified splits with parallel processing and generating
   attention weights.

   Example — evaluating Conformer on AMP-SCZ:
   ```bash
   python -u run.py --task_name classification --is_training 1  --root_path ./dataset/AMP-SCZ --model_id AMP-SCZ_Conformer  --model Conformer --data AMP-SCZ  --e_layers 6 --batch_size 32 --d_model 128 --d_ff 256 --swa --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 10 --use_gpu True --dropout 0.15 --weight_decay 1e-2 --stratified_splits_dir stratified_holdout_splits --run_all_splits --num_workers 8
   ```
   `TeCh` has its own separate copy of the training/eval scripts at
   `Transformer_Changes/TeCh/run.py` and
   `Transformer_Changes/TeCh/exp_classification.py`, with per-dataset
   shell configs in `Transformer_Changes/TeCh/scripts/` (`ADFTD.sh`,
   `APAVA.sh`, `TDBRAIN.sh`).
5. **Evaluation / attention extraction.** Re-run `run.py` with
   `--is_training 0`, keeping all other params identical to the training
   run. Add `--output_attention` to generate attention maps. Attention-map
   computation lives in `Transformer_Changes/exp/exp_classification.py`
   and `Transformer_Changes/utils/vis_attn.py`.

## KL-Divergence analysis (Medformer)

We extract the window-averaged, last-layer, all-head query-key attention
matrix per subject and compute the KL divergence independently for every
(head, query) row between all correct/incorrect subject pairs within each
split. After removing the CLS key, the attention matrix is 64×64 for ADFTD
and APAVA, and 32×32 for TDBrain and AMP-SCZ — reflecting Medformer's
dataset-specific `seq_len`, `enc_in`, and `patch_len` configs.

Steps to reproduce Table 2:

1. Run Medformer with the dataset-specific configs in `Transformer_Changes/scripts/Medformer.sh` and `--output_attention` set. This generates a folder such as `Attn_Weights_Medformer` containing correct/incorrect
   attention weights (query and key vectors, 64 patches) as two CSVs.
2. Run `KL_Divergence/kl_raw_pairs.py` on the two CSVs to produce
   `kl_raw_by_head_query_pairs.csv`.
3. Run `KL_Divergence/Calc_Grand_Mean_KL_raw.py` to compute the mean and
   standard deviation across all correct/incorrect pairs.

For within-group KL results: keep the correct/incorrect attention-weight
CSVs and run `KL_Divergence/kl_within_group.py` and
`KL_Divergence/Calc_within_group_stat.py`. For the distribution vs. a
uniform KL baseline, run `KL_Divergence/Div_from_Uniform.py`.

## CRT ablation

To ablate cross-token interaction, changes are limited to
`Transformer_Changes/models/Transformer.py` — masking is applied to the
off-diagonal attention weights in the constructor and classification
method. Pre-computed results are already in the repo:
`Transformer_Changes/CRT Ablation/Transformer_With_Attention_Res_*.txt`,
`Transformer_No_Attention_Res_*.txt`, and the summary figure
`attention_ablation_comparison_combined.png`.

## Sensitivity of spectral features (Section 3.3)

To reproduce with Medformer:

- **Structured bands (SBD):** run `Common/increase_Tensor_Dim_SBD.py
  --flatten` against the downloaded `Feature` numpy directory. Given a
  dataset feature directory, it segregates the time signals into 5
  frequency bands, producing a modified tensor dimension.
- **Random bands (RBD):** run `Common/increase_Tensor_Dim_RBD.py`
  the same way.

Before running either SBD/RBD experiment with attention models, update
`Transformer_Changes/data_provider/data_loader.py` for the new tensor
shape. `Transformer_Changes/data_provider/uea.py`,
`Transformer_Changes/layers/Augmentation.py`, and
`Transformer_Changes/layers/Embed.py` have already been modified to
handle the updated tensor dimensions.

## Runtime analysis (Appendix)

- **Model runtime:** `Timing_Analysis/MedFormer/run.py` and
  `Timing_Analysis/TeCh/run.py` are runtime-instrumented copies of the
  respective training scripts; per-dataset timing CSVs are included
  alongside each.
- **Spectral pipeline runtime:** `Timing_Analysis/Spectral_Pipeline/{dataset}/timing.py`
  generates the full spectral pipeline along with distributional
  statistics on the number of windows per subject, for each dataset.


