# Protocol amendment v1.4: exact-Pneumonia-label external sensitivity analysis

## Discovery time

After completion of the approved one-time 339-image VinDr-PCXR Stage-1 external dataset-label feasibility inference.

## Reason

The approved primary external mapping combined clean `Pneumonia` and clean `Brocho-pneumonia` labels as a pneumonia-related dataset-label group. The authors requested an additional stricter binary sensitivity analysis restricted to the exact `Pneumonia` label.

## Frozen rule

- Negative: `No finding == 1` and every disease label equals zero.
- Positive: `Pneumonia == 1` and every other image-level label, including `Brocho-pneumonia`, equals zero.
- Reuse only probabilities from `external_stage1_one_time_predictions.csv`.
- Reuse the frozen threshold `0.8201918601989746`.
- Do not reload the model or DICOM images.
- Use 5,000 image-level percentile-bootstrap replicates with seed `20260720`.

## Interpretation

This is a post-result exploratory label-definition sensitivity analysis. It cannot replace the approved 339-image external result and cannot be described as a new independent or confirmatory external test.

