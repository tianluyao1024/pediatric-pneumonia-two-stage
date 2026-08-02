# Protocol amendment v1.5: unified lung-field crop technical sensitivity

## Status

Frozen before generation of any cropped classification result.

## Reason

Blinded technical review identified a field-of-view mismatch: development images generally emphasize the thorax, whereas part of the approved VinDr-PCXR subset retains substantial inferior abdominal anatomy. Direct fixed-size resizing can therefore reduce the effective thoracic resolution in the external images.

## Role

This amendment defines a post-result technical preprocessing sensitivity analysis. The original 339-image external result remains the approved primary external dataset-label feasibility result. Cropped-model results may not replace or retroactively redefine it.

## Frozen processing

- Anatomical segmenter: TorchXRayVision 1.5.2 ChestX-Det PSPNet.
- Weight SHA-256: `019b167eac6b729fc1bb92bbbc185fc1730aaa65819f4e3fe718186cadc044fc`.
- Segmentation targets: Left Lung and Right Lung.
- Segmenter input: deterministic grayscale display image; official center-square crop; normalization to [-1024, 1024]; 512 x 512 inference.
- Mask activation and threshold: sigmoid, 0.5.
- Component rule: largest connected component independently for each lung channel.
- Quality gate: both lungs present; each lung area >= 0.5% of the 512 x 512 field; combined area 2%-60%; combined bounding-box width and height each >= 25% of the field.
- Crop margin relative to the lung bounding box: left/right 12%, superior 15%, inferior 10%.
- Failure fallback: deterministic center-square crop; no image exclusion and no manual crop.
- Display standardization: MONOCHROME1 inversion when required, then fixed 0.5-99.5 percentile scaling.
- Classifier input: retain crop aspect ratio, black letterbox to 224 x 224, grayscale replicated to three channels, ImageNet normalization.

## Dataset handling

- Apply the identical code and parameters to all 5,281 development images and all 339 approved external images.
- Do not read the fixed exploratory test set.
- Do not use labels, probabilities, predictions, or performance metrics to determine a crop.
- Raw images are immutable. All derived PNGs, crop coordinates, quality flags, code/configuration hashes, and output hashes are retained separately.

## Model analysis

- Reuse the frozen five outer folds and inner group-disjoint assignments.
- Reuse the frozen Stage-1 architecture, optimizer, augmentation, checkpoint, seed, and threshold-selection policies.
- Complete five-fold development OOF before training the complete-development final model.
- The external cropped analysis reuses the same approved 339 image IDs and label mapping and is reported only as a post-result technical sensitivity analysis.

