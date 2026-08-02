# Protocol amendment: Phase 2 near-duplicate review conclusion

Status: FINAL FOR THE CALIBRATION REVIEW SAMPLE.

The 120 blinded calibration pairs sampled from the immutable 87,219-pair Tier-0 near-duplicate screening universe were manually reviewed and confirmed to be visually similar but different radiographs. All 120 pairs are recorded as `different_radiograph` with action `do_not_merge`.

Consequences:

1. Perceptual-hash candidate status is a high-recall screening signal only and is not evidence of image duplication.
2. No candidate from the near-duplicate universe is automatically merged, excluded, or used as a split constraint.
3. Exact SHA-256 duplicate components remain the only duplicate-content constraints for subsequent grouping and sensitivity analyses.
4. This conclusion does not make a prevalence claim for all 87,219 candidates; it freezes the operational rule that no near-duplicate threshold will be used to create additional duplicate components in this revision.
