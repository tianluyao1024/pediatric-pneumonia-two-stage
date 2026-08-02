# Final-revision aggregate results

This directory contains sanitized, aggregate-only outputs for manuscript revision v1.0.0. No VinDr-PCXR DICOM files, filenames, patient identifiers, image-level predictions, credentials, or model weights are redistributed.

- `external_stage1_complete_metrics_ci.csv`: complete available Stage-1 external-feasibility and post-result technical-sensitivity estimates with 95% image-bootstrap confidence intervals.
- `external_stage1_feasibility_summary.csv`: plotting source data for the four-panel external-feasibility figure.

The frozen whole-image run is the prespecified external result. Lung-field cropping and crop-plus-intensity runs were performed after that result was observed and are exploratory technical sensitivities. `vindr_internal_oof` is internal image-level OOF within the target subset and is not external validation.
