# Phase 6 external Stage-1 inference complete

One and only one approved VinDr-PCXR v1.0.0 external Stage-1 dataset-label feasibility inference was completed on 2026-08-02.

- Cohort: the approved 339-image source-level-PA subset (139 pneumonia-related dataset labels; 200 clean no-finding labels).
- Model SHA-256: `1b54cedec02923e8a21cbcc6956c882446be23c58c4b14c2090f2f3aa0e7ca24`.
- Threshold: `0.8201918601989746`, frozen from the inner filename-group-disjoint validation set using Youden J.
- Fixed exploratory test access: false.
- Stage 2 inference: false.
- Result files: `external_stage1_one_time_predictions.csv` and `external_stage1_one_time_results.json`.

The result is an external dataset-label feasibility evaluation only. It must not be described as patient-level validation, per-image view-verified validation, clinical diagnosis, pathology, or microbiological diagnosis. No changes to model, threshold, preprocessing, or cohort selection are permitted after this run.
