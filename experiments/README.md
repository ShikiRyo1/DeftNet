# Experiments

Committed experiment artifacts are intentionally small, structured, and
non-sensitive.

## Current Public Tables

- `pooled_public_fixed_notta_seed_summary_v77.csv`: v77 15-method pooled public
  coronary angiography benchmark. Values are three-seed mean and seed SD under
  fixed threshold `0.5` with no test-time augmentation.
- `fusion_hsaf_vs_mean.json`: v77 HSAF mechanism controls against feature-mean
  fusion, output-level ensembles, random routing, and shuffled routing.
- `release_manifest_v0.1.1.json`: current public release manifest and config
  checksums.

## Historical Release Records

- `release_manifest_v0.1.0.json`: pointer to the first public code-only release.
  The old aggregate table remains available through the `v0.1.0` Git tag, but it
  is not the current README headline result.

## Not Committed

Large raw dumps, per-image prediction masks, dataset archives, clinical images,
and checkpoints are excluded from git. Release them through GitHub Releases,
Zenodo, Hugging Face, or another artifact store only after license and privacy
checks.

Use `scripts/per_image_metrics.py`, `scripts/paired_stats.py`, and
`scripts/profile_model.py` to regenerate final per-image statistics and hardware
profiles after the final checkpoint and prediction masks are release-approved.
