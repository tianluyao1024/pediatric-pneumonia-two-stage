# Leakage-aware two-stage classification of paediatric pneumonia

Reproducibility package for the accompanying study of the Kaggle *Chest X-Ray Images (Pneumonia)* collection. The project evaluates a sequential classifier:

1. **Stage 1 screening:** normal versus pneumonia.
2. **Stage 2 subtype-label classification:** bacterial versus viral among pneumonia images.
3. **End-to-end cascade:** normal, bacterial pneumonia, or viral pneumonia.

The package includes the training/evaluation code, fixed filename-derived group assignments, sanitised prediction tables, aggregate metrics and publication figures. It deliberately excludes raw radiographs, weights, cached embeddings and manuscript files.

> **Research-use warning.** Dataset labels are dataset-provided subtype labels, not microbiological diagnoses. This repository is not a medical device and must not be used for diagnosis or clinical decision-making.

## What is reproducible

The code implements the following analyses reported in the paper:

- filename-group-disjoint train/validation/test partition construction;
- conventional and deep Stage-1 baselines, including PneuNet;
- Stage-2 PneuNet-v2 training, grouped bootstrap evaluation and fixed-transform stacking;
- fold-specific paediatric SimSiam representation learning, ImageNet-initialised training and fixed-weight OOF fusion;
- fixed-transform GroupKFold stacking (the previous within-cohort rank stack is not used);
- complete end-to-end cascade and single-stage three-class comparison;
- CER-Net calibration/rejection audit, source-directory stress test and figure generation.

The committed `results/` directory contains the de-identified tables and predictions used to generate the reported estimates. Re-running neural-network training can vary slightly with GPU, CUDA/cuDNN and stochastic data augmentation; use the included predictions for exact reproduction of the published tables.

## Data access and layout

Download the images directly from the [Kaggle dataset page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and comply with its licence and terms. Do **not** commit downloaded images.

After extracting the Kaggle archive, place the `chest_xray/` directory below `data/raw/extracted/` so that this example exists:

```text
data/raw/extracted/chest_xray/train/NORMAL/IM-0115-0001.jpeg
```

The scripts pool the original Kaggle folders before building the filename-derived groups. The split is subtype-namespaced and filename-group-disjoint; it is not a verified patient-disjoint split. The test partition was examined during model development and is therefore a **fixed exploratory test set**, not a fresh confirmatory test set.

## Environment

The experiments were developed with Python 3.11 and CUDA-enabled PyTorch. Create a fresh environment and install the pinned packages:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm that PyTorch detects the intended GPU before long training:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Reproduction workflow

Set the repository root once, then run each analysis in order. The command below uses the local Python interpreter; replace it with an absolute path if necessary.

```powershell
$root = (Get-Location).Path
$python = "python"
$env:PNEUMONIA_PROJECT_ROOT = $root

# 1. Stage 1: normal versus pneumonia and fixed split outputs
& $python -m src.stage1_compare_select_stable_v3 --project-root $root

# 2. Stage 2 PneuNet-v2 and related whole-image baselines
& $python -m src.train_stage2_optimized --project-root $root --name pneunet_v2 --size 320 --epochs 14
& $python -m src.export_stage2_val_predictions --project-root $root

# 3. Optional next-generation whole-image models
& $python -m src.stage2_train_nextgen --project-root $root --model all --device cuda --resume

# 4. Audit available Stage-2 prediction files and calculate grouped statistics
& $python -m src.stage2_compare_select_nextgen --project-root $root

# 5. Fold-specific paediatric SimSiam pretraining
& $python -m src.stage2_ssl_pretrain_groupfold --project-root $root

# 6. Nested whole-image OOF models and the fixed 50:50 ImageNet-SSL fusion
& $python -m src.stage2_whole_nested_diagnostic --project-root $root --initialization imagenet --run-name whole_nested_imagenet_simple
& $python -m src.stage2_whole_nested_diagnostic --project-root $root --initialization ssl --run-name whole_nested_ssl_simple
& $python -m src.assemble_fixed_oof_fusion --project-root $root

# 7. Fixed-transform GroupKFold stack (a separate conditional benchmark)
& $python -m src.stage2_oof_fixed_stack --project-root $root

# 8. Route all images through the Stage-1 -> Stage-2 cascade
& $python -m src.evaluate_end_to_end_cascade --project-root $root

# 9. Regenerate the paper figures
& $python -m src.plot_end_to_end_cascade
& $python -m src.redraw_reference_style_architecture
```

Or use the provided helper for the core five training/evaluation stages:

```powershell
.\scripts\run_selected_experiments.ps1 -Python "C:\\Path\\To\\python.exe" -ProjectRoot $PWD
```

Training writes local checkpoints, intermediate arrays and large images below `models/`, `artifacts/` and `data/processed/`; these directories are intentionally ignored by Git. The complete run needs a CUDA-capable GPU, substantial disk space and several hours or longer depending on the hardware. The committed `results/` files are the canonical route for reproducing the reported tables without retraining.

## Exact reported results

For the paper tables without retraining, use the committed sanitised files under `results/`. They contain fixed-split predictions, group identifiers, bootstrap summaries and source-directory analyses with local absolute paths removed. Re-run the corresponding analysis scripts after training if you want regenerated outputs.

### Complete 95% confidence intervals

All reported uncertainty is a percentile 95% bootstrap interval calculated by resampling the subtype-namespaced filename group (2,000 resamples). The fixed test set remains exploratory; these intervals quantify sampling variability under this grouping rule and do not turn it into a confirmatory evaluation.

| File | Contents |
|---|---|
| `results/main_tables_complete_group_ci.csv` | Complete Stage-1 and conditional Stage-2 table metrics: ROC-AUC, PR-AUC, accuracy, balanced accuracy, sensitivity, specificity, and F1, each with lower/upper bounds. |
| `results/stage2_whole_nested_oof_group_ci.csv` | ImageNet, paediatric SimSiam, and fixed 50:50 whole-image development OOF ROC-AUC and PR-AUC intervals. |
| `results/end_to_end_cascade_metrics.json` | End-to-end cascade accuracy, macro-F1, and all class-recall intervals. |
| `results/single_stage_pneunet_3class_metrics.json` | Matched single-stage three-class comparator accuracy, macro-F1, and all class-recall intervals. |
| `results/stage1_all_models_locked_test_ci.csv`, `results/stage2_all_models_locked_test_ci.csv` | Model-level fixed-exploratory-test bootstrap results, including all recorded metrics. |

The compact manuscript notation `estimate (+upper/-lower)` expresses the same asymmetric interval as the CSV/JSON lower and upper bounds. The complete plotting script for the cascade comparison is `src/plot_end_to_end_cascade.py`; it consumes the two end-to-end JSON files and draws the reported error bars.

The post hoc Stage-2 ensemble is a **fixed 50:50 probability average** of ImageNet-initialised and fold-specific paediatric-SimSiam EfficientNet-B0 models. It uses no test-set information and no weight search, but it was assembled after component OOF predictions existed; its pooled OOF results are exploratory development evidence rather than a preregistered primary endpoint.

## Repository map

```text
src/       Training, SSL, stacking, calibration, cascade and plotting modules
scripts/   Portable PowerShell runner and result-sanitisation utility
results/   Sanitised metrics, split assignments and prediction tables
figures/   Submission-quality figures generated from experiment outputs
config.yaml  Default training and bootstrap settings
```

## Authors

Luyao Tian (first author), tianluyao1024@gmail.com  
Yuannong Ye (corresponding author), ynye.uestc@gmail.com  
School of Biology and Engineering, Guizhou Medical University, Guiyang, Guizhou, China
