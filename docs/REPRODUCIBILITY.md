# Reproducibility Protocol

## Main Operating Point

The primary DCA result uses:

- input size: `512 x 512`
- threshold: `0.5`
- test-time augmentation: disabled
- post-processing: identical binary connected-component handling for all models
- split: held-out DCA true-test frames, `n=134`
- metric code: identical implementation for every model

Validation-selected thresholds are acceptable as a secondary clean protocol when
the threshold is selected on validation data only and frozen before true-test
evaluation.

## What Is Not a Main Result

The following can be useful as sensitivity analysis but should not be used as the
headline claim:

- per-image oracle threshold chosen against each test ground-truth mask;
- per-metric threshold sweep on test labels;
- TTA-only comparisons when baselines do not receive the same TTA budget;
- legacy tables with disclosed uplift constants.

## Metric Set

Primary binary segmentation metrics:

- Dice
- IoU
- sensitivity
- precision
- specificity
- MCC

Vessel-structure metrics:

- clDice
- centerline continuity
- branch recall
- small-vessel recall
- break count
- HD95
- ASSD

## Recommended Statistical Tests

For a paper submission, compute per-image paired metrics under the fixed
operating point and report:

- paired bootstrap confidence intervals, `B=10000`;
- two-sided Wilcoxon signed-rank tests;
- Holm correction across baseline comparisons;
- matched-pairs rank-biserial effect size;
- explicit note that frame-level bootstrap is used unless patient/procedure IDs
  are available for cluster-aware aggregation.

## Checkpoint Compatibility

The cleaned package supports configurable depth-band policies. When releasing a
checkpoint, include:

- exact config YAML;
- git commit hash;
- training data split manifest;
- metric script version;
- whether expert encoders were frozen, partially unfrozen, or fully trainable.
