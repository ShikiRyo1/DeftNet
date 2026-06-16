# Reproducibility Protocol

This protocol is aligned to manuscript working version **v77, 2026-06-15** and
to the public code release shape in this repository.

## Main Operating Point

The primary public benchmark uses:

- input size: `512 x 512`;
- threshold: `0.5`;
- test-time augmentation: disabled;
- post-processing: identical binary-mask handling across methods;
- seeds: `0`, `42`, and `44`;
- benchmark: pooled public coronary angiography data from DCA/DCAE, XCAD, and
  ARCADE after binary vessel-mask harmonization;
- metric code: identical implementation for every model.

The public CSV reports three-seed mean `+/-` seed SD:
`experiments/pooled_public_fixed_notta_seed_summary_v77.csv`.

## What Is Not a Main Result

The following may be useful as sensitivity analysis but should not be used as
the headline claim:

- per-image oracle thresholds chosen against test ground truth;
- per-metric threshold sweeps on test labels;
- TTA-only comparisons when baselines do not receive the same TTA budget;
- older DCA-only or enhanced-setting numbers from intermediate drafts;
- validation-calibrated mechanism controls used only to test competing
  explanations.

## Metric Set

Primary binary segmentation metrics:

- Dice
- IoU
- sensitivity
- precision
- specificity
- MCC

Vessel-structure and guardrail metrics:

- clDice
- cbDice
- centerline continuity
- branch recall
- small-vessel recall
- beta0 error
- fragmentation
- break count
- HD95
- ASSD

HD95 and ASSD are reported as distance-metric guardrails in the v77 narrative.
They are not claimed as DEFT-Net wins when a competing baseline is lower.

## Recommended Statistical Tests

For a paper submission or a fully reproducible artifact release, compute
per-image paired metrics under the fixed operating point and report:

- paired bootstrap confidence intervals, `B=10000`;
- two-sided Wilcoxon signed-rank tests;
- Holm correction across baseline comparisons;
- matched-pairs rank-biserial effect size;
- source-aware sensitivity over reconstructed source strata;
- explicit note that image-level bootstrap is used unless patient/procedure IDs
  are available for cluster-aware aggregation.

The repository includes `scripts/per_image_metrics.py` and
`scripts/paired_stats.py` for this workflow. Final per-image CSVs are gated until
filename, metadata, license, and privacy checks are complete.

## Checkpoint Compatibility

The cleaned package supports configurable depth-band policies. When releasing a
checkpoint, include:

- exact config YAML;
- git commit hash;
- training data split manifest;
- metric script version;
- whether expert encoders were frozen, partially unfrozen, or fully trainable;
- hardware profile for latency, VRAM, and approximate MACs.

## Minimal Local Verification

Without datasets or checkpoints, users can still verify the package surface:

```bash
pip install -e ".[dev]"
pytest
python -m compileall src scripts tests examples
python examples/synthetic_demo.py --output-dir examples_output
```
