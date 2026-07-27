# Leakage-aware two-stage classification of paediatric pneumonia

This repository accompanies a study of chest radiographs from the Kaggle *Chest X-Ray Images (Pneumonia)* dataset.  It contains the training and evaluation scripts, fixed filename-group assignments, aggregate metrics, figures, and de-identified per-image prediction tables used for reproducibility.

## Task

The evaluation is organised as a cascade:

1. **Stage 1:** normal versus pneumonia.
2. **Stage 2:** bacterial versus viral pneumonia, applied to images routed from Stage 1.
3. **End-to-end cascade:** normal, bacterial pneumonia, or viral pneumonia.

The project also reports a single-stage three-class reference model, group-level bootstrap confidence intervals, paired comparisons, source-directory analyses, and a rejection-option analysis (CER-Net).

## Important evaluation note

All original Kaggle folders were merged before analysis and a filename-derived group split was used to prevent a group appearing in more than one partition.  The repository preserves the evaluated split and predictions.  Because the fixed test partition was inspected during iterative model development, it should be interpreted as a **fixed exploratory evaluation set**, rather than as a freshly locked final test set.  Results are research outputs only and are not clinical performance claims.

## Reproduction

1. Obtain the dataset directly from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and comply with its terms.
2. Create an environment with Python 3.11 and install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Place the extracted Kaggle dataset under `data/raw/extracted/` (the directory is deliberately excluded from version control).
4. Run a script with `--project-root` set to this repository.  For example:

   ```powershell
   python -m src.stage1_compare_select_stable_v3 --project-root $PWD
   python -m src.stage2_compare_select_nextgen --project-root $PWD
   ```

Hardware and package versions affect training time and minor stochastic variation.  The published `results/` files are the exact tabular outputs used for the reported analyses; absolute local paths have been removed.

## Contents

- `src/` — training, stacking, calibration and cascade-evaluation programs.
- `scripts/` — a portable run helper and result-publication helper.
- `results/` — metrics, group assignments and sanitised OOF / end-to-end predictions.
- `figures/` — publication figures generated from the experiment outputs.
- `config.yaml` and `requirements.txt` — experimental defaults and package versions.

Raw radiographs, model checkpoints, and cached feature arrays are intentionally not uploaded.  This code is for research and education; it is not a medical device and must not be used for diagnosis.

## Authors

Luyao Tian (first author): tianluyao1024@gmail.com

Yuannong Ye (corresponding author): ynye.uestc@gmail.com
School of Biology and Engineering, Guizhou Medical University, Guiyang, Guizhou, China
