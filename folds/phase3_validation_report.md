# Phase 3 validation report

- Development cohort: 5,281 images (normal 1,425; bacterial 2,524; viral 1,332).
- Fixed exploratory set: 575 images; it was not read for assignment, fitting, threshold selection, or model selection.
- Main and conservative assignment files each contain one and only one outer-fold assignment for every development image.
- Stage-2 future training must inherit the complete-cohort outer-fold assignment; no pneumonia-only re-splitting is permitted.
- No training, preprocessing, model selection, metric computation, or source-data modification occurred.
- Phase 4 is blocked pending explicit configuration freeze.
