# Experiments

Committed experiment artifacts are intentionally small and non-sensitive:

- `dca_fixed_notta_results.csv`: fixed-threshold/no-TTA DCA aggregate table.
- `fusion_hsaf_vs_mean.json`: audit result for HSAF vs uniform mean fusion.
- `release_manifest_v0.1.0.json`: code-release manifest and config checksums.

Large raw dumps, per-image prediction masks, dataset archives, and checkpoints
are excluded from git. Release them through GitHub Releases or an external
artifact store only after license and privacy checks.

Use `scripts/per_image_metrics.py`, `scripts/paired_stats.py`, and
`scripts/profile_model.py` to regenerate final per-image statistics and hardware
profiles after the final checkpoint and prediction masks are release-approved.
