# Experiment artifacts

Committed artifacts are structured, aggregate, and non-sensitive.

## Current tables

- `pooled_public_fixed_notta_seed_summary.csv`: current 15-method
  released-image benchmark, reported as three-seed mean and seed SD;
- `mechanism_controls_three_seed_summary.csv`: fusion, routing, admission,
  output-level, and expert controls with explicit evidence level;
- `full_data_controls_three_seed_summary.csv`: same-architecture full-data,
  heterogeneous full-data, and fold-perspective heterogeneous expert-bank
  comparison;
- `fusion_hsaf_vs_mean.json`: compact machine-readable view of selected HSAF
  controls.
- `release_manifest_v0.3.0.json`: protocol summary, artifact boundary, and
  SHA-256 digests for the current public release files.

Version-suffixed files, including the source files from which the stable aliases
were copied, are retained for traceability. New analyses should use the stable
filenames above rather than internal manuscript revision identifiers.

## Not committed

Raw datasets, clinical images, prediction masks, restricted filenames,
checkpoints, and large per-image dumps are excluded. Release them through a
separately reviewed artifact store only after source-license and privacy checks.

Use `scripts/per_image_metrics.py`, `scripts/paired_stats.py`, and
`scripts/profile_model.py` to regenerate approved evaluation and hardware
artifacts.
