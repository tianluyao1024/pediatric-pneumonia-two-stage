# Protocol amendment v1.6: square thoracic crop and robust intensity harmonization

## Status

Frozen after completion of amendment v1.5 and before generation of any v1.6 classification result.

## Motivation

The v1.5 model inputs retained a substantial brightness and padding difference between datasets. Median image mean intensity was 157.7 in the development cohort and 106.6 in the external cohort; median zero-pixel fraction was 7.1% and 24.7%, respectively.

## Frozen transformation

- Reuse the immutable v1.5 source-image hashes and lung-crop coordinates.
- Reconstruct the crop from the original source image; do not use labels or predictions.
- Expand the existing rectangular crop to the smallest centered square contained within the v1.5 center-square field, shifting at boundaries without changing side length.
- Compute robust intensity parameters from nonzero pixels inside the square thoracic crop.
- Center: pixel median.
- Scale: IQR / 1.349, with a fixed lower bound of 5 grayscale units.
- Clip robust z-scores to [-3, 3] and linearly map to [0, 255].
- Resize directly to 224 x 224; no letterbox or padding.
- Apply the identical transformation to all 5,281 development images and all 339 approved external images.

## Analysis role

This is a post-result technical sensitivity analysis. It does not replace the original external result or the v1.5 lung-crop result. The five frozen filename-group outer folds, inner assignments, Stage-1 architecture, optimization, augmentation, checkpoint selection, seed policy, and threshold policy remain unchanged.

## Prohibitions

- No parameter selection using external labels, probabilities, or performance.
- No external threshold optimization or calibration.
- No fixed exploratory test access.
- No deletion or overwrite of v1.5 artifacts.

