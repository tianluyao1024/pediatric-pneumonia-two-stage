# Protocol amendment v1.3: VinDr-PCXR source-level PA verification

**Status:** USER-APPROVED; two-author inference approval still required  
**Discovery time:** 2026-08-02  
**Affected phase:** Phase 6 external Stage-1 feasibility evaluation

## Issue discovered

All 339 author-approved VinDr-PCXR DICOM files passed their publisher-provided
SHA-256 checksums, were parseable, and had decodable pixel arrays. However, the
deidentified files do not contain standard `Modality` or `ViewPosition` DICOM tags.
Consequently, the v1.2 rule requiring per-file DICOM confirmation of `CR` and `PA`
cannot be executed on this release.

## Old rule

```text
Every downloaded external image must have DICOM Modality == CR and
ViewPosition == PA before external inference.
```

## Revised rule

```text
Every selected external image must match its publisher SHA-256, be DICOM-parseable,
and have a decodable pixel array. PA status is verified at source level from the
VinDr-PCXR release documentation; the absence of per-file ViewPosition and Modality
tags is explicitly reported.
```

## Scientific justification and limits

VinDr-PCXR documentation describes the released pediatric cohort as PA chest
radiographs. This is source-level cohort evidence, not per-image metadata evidence.
The amendment does not change the external label mapping, image selection, model,
preprocessing, threshold, or analysis endpoints.

The external analysis remains limited to Stage 1 and may only be described as an
external feasibility evaluation of dataset-label mapping. It cannot support claims of
clinical diagnosis, pathogen classification, patient-level validation, or guaranteed
per-image view verification.

## Required artifacts and approval gate

Before any inference:

1. retain `external_dicom_audit.csv` and `external_dicom_audit_summary.json`;
2. freeze the exact checkpoint or predeclared ensemble, preprocessing hash, threshold,
   code commit, and environment hash;
3. complete `external_inference_approval.yaml` with both authors' approval;
4. execute one inference run only, with no post-result changes.

## Approval record

The user approved this amendment in the active Codex task on 2026-08-02. This record
does not substitute for the two-author external-inference approval file required above.
