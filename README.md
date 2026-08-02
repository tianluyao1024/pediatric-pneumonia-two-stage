# Pediatric pneumonia classification: auditable revision package

Reproducibility package for:

> **Leakage-Aware Evaluation and Source-Directory Technical Audit of Pediatric Pneumonia Classification on a Public Chest-Radiograph Benchmark**

The study evaluates a sequential classifier on the Kaggle *Chest X-Ray Images (Pneumonia)* collection:

1. Stage 1: normal versus pneumonia.
2. Stage 2: bacterial versus viral **dataset-label classification** among pneumonia images.
3. Routing-aware cascade: normal, bacterial pneumonia, or viral pneumonia.
4. FLOP-matched single-stage three-class comparator.

The principal evidence is five-fold outer out-of-fold (OOF) evaluation of all 5,281 development images. The historical test directory is a **fixed exploratory test set**, not a fresh confirmatory test set.

> **Research-use warning:** Stage 2 labels are inherited dataset labels, not microbiological diagnoses. This repository is not a medical device and must not be used for diagnosis or clinical decision-making.

## Revision v1.1.0

This release contains the frozen, auditable materials supporting the revised manuscript:

- principal namespaced filename-group + exact-content component analysis;
- conservative raw-identifier-linked + exact-content component sensitivity analysis;
- routing-aware cascade and budget-matched single-stage OOF comparisons;
- 5,000-draw paired constraint-component bootstrap confidence intervals;
- development-only source-directory technical audit and 200-permutation negative control;
- prespecified 339-image VinDr-PCXR/PediCXR Stage-1 external dataset-label feasibility result;
- post-result lung-field and intensity-scaling technical sensitivity summaries;
- complete fold-level runtime, optimizer-step, memory and test-read accounting;
- editable SVG/PDF figures and their source-data tables.

The main OOF comparison did not support a cascade advantage. Macro-F1 was 0.797 (95% CI 0.784–0.811) for the cascade and 0.825 (0.813–0.836) for the matched single-stage model; the paired difference was −0.028 (−0.040 to −0.015). Frozen VinDr-PCXR Stage-1 inference yielded AUROC 0.642 (0.583–0.701). The external analysis is restricted dataset-label feasibility evidence, not clinical validation.

## Repository map

```text
protocol/                         frozen configurations and protocol amendments
folds/                            the two immutable 5-fold assignment files and audits
audit/duplicate_review/           blinded near-duplicate adjudication outputs
environment/                      sanitized software/GPU environment
results/revision_v1_1/
  phase4b/main/                   principal OOF estimates, paired differences, errors
  phase4b/conservative_raw_identifier/
  phase5_source_directory_audit/ source classifier and permutation negative control
  phase6_external_stage1/         label map, approval hashes, frozen aggregate result
  figure_source_data/             exact plot source tables
  runtime_accounting_by_fold.csv  30 model-fold execution records
figures/final_submission/         editable SVG and PDF publication figures
src/                              analysis and training source code
scripts/                          release/reproduction helpers
release_v1_1_sha256.csv           SHA-256 and size for every published file
```

## Data access

Download the internal benchmark from the [Kaggle dataset page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and comply with its terms. Raw images are intentionally excluded.

The external resource is [VinDr-PCXR v1.0.0](https://physionet.org/content/vindr-pcxr/1.0.0/), described as PediCXR in the accompanying data article (version DOI: `10.13026/k8qc-na36`). It is access-controlled. No VinDr DICOM files or redistributable derivatives are included here.

The filename-derived grouping is **not** verified patient grouping. The conservative raw-identifier-linked analysis may over-merge unrelated records sharing a numeric token and is reported only as a leakage-sensitivity analysis.

## Environment

The frozen public environment is in `environment/public_environment.json`. Core versions were Python 3.11.6, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, CUDA 12.8, and an NVIDIA GeForce RTX 5080 Laptop GPU.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduction order

1. Verify `release_v1_1_sha256.csv`.
2. Review the frozen files under `protocol/`.
3. Place locally obtained images outside version control.
4. Reuse the committed outer-fold assignments under `folds/`; do not regenerate them.
5. Run the Stage-1, Stage-2 and matched single-stage OOF pipelines using the frozen configurations.
6. Build routing-aware cascade predictions and paired component-bootstrap statistics.
7. Run the source-directory audit only on the development cohort.
8. Treat the committed external result as one-time frozen inference; do not tune against it.
9. Rebuild figures from `results/revision_v1_1/figure_source_data/`.

The committed aggregate tables reproduce the manuscript values exactly without requiring redistribution of images or model weights. Neural-network retraining can vary slightly across CUDA/cuDNN hardware despite fixed seeds.

## Integrity and privacy boundaries

This public package excludes:

- all raw Kaggle and restricted VinDr images;
- trained checkpoints and cached embeddings;
- local absolute paths and executable locations;
- private blinded-review keys;
- usernames, passwords, cookies, tokens, and browser state;
- the manuscript file itself.

Before release, the repository is scanned for local paths and credentials, and every published file is covered by `release_v1_1_sha256.csv`.

## Authors

- **Lu-Yao Tian** — `1470496531@qq.com`; ORCID `0009-0006-7097-8604`
- **Yuan-Nong Ye** (corresponding author) — `yyngmc@gmail.com`

Affiliations:

1. Guizhou Key Laboratory of Microbiome and Infectious Disease Prevention and Control, School of Biology and Engineering, Guizhou Medical University, Guiyang, China.
2. Laboratory of Intelligent Analysis and Mining for Biomedical Big Data, Department of Medical Artificial Intelligence, School of Biology and Engineering (Modern Industrial College of Health Medicine), Guizhou Medical University, Guiyang, China.

## License and citation

Code licensing does not override the terms of the original image datasets. Cite the manuscript, Kaggle benchmark, and VinDr-PCXR/PediCXR data record as applicable. Release URL: <https://github.com/tianluyao1024/pediatric-pneumonia-two-stage/releases/tag/v1.1.0>.
