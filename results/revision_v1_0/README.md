# Final-revision aggregate results

This directory contains sanitized, aggregate-only outputs for manuscript revision v1.0.0. No VinDr-PCXR DICOM files, filenames, patient identifiers, image-level predictions, credentials, or model weights are redistributed.

- `external_stage1_complete_metrics_ci.csv`: complete available Stage-1 external-feasibility and post-result technical-sensitivity estimates with 95% image-bootstrap confidence intervals.
- `external_stage1_feasibility_summary.csv`: plotting source data for the four-panel external-feasibility figure.
- `external_stage1_threshold_freeze.json`: prespecified full-development threshold provenance, selection cohort, and checkpoint/preprocessing/code hashes.
- `external_inference_approval_v1_1.yaml`: hash-bound two-author approval record for the one permitted VinDr-PCXR inference run.
- `submission_artifact_sha256.csv`: SHA-256 manifest for the final submission figures and external-threshold freeze records.

The frozen whole-image run is the prespecified external result. Lung-field cropping and crop-plus-intensity runs were performed after that result was observed and are exploratory technical sensitivities. `vindr_internal_oof` is internal image-level OOF within the target subset and is not external validation.

The external threshold was 0.8201918601989746. It was selected on the prespecified internal group-disjoint validation partition before any VinDr-PCXR inference and was distinct from the fold-specific thresholds used for development OOF evaluation.
