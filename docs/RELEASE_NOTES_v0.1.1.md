# v0.1.1 v77 Experiment-Summary Release

This release updates the public DeftNet / DEFT-Net repository to the manuscript
working version **v77, 2026-06-15**.

## Changed

- Updated README headline results to the v77 pooled public coronary angiography
  benchmark.
- Replaced the earlier DCA-only aggregate CSV with
  `experiments/pooled_public_fixed_notta_seed_summary_v77.csv`.
- Updated HSAF mechanism-control JSON to v77 Table 8 values.
- Rewrote the experiment-design and reproducibility docs around the final v77
  evidence chain.
- Added `docs/PAPER_V77_EXPERIMENT_UPDATE.md` for a recruiter- and collaborator-
  friendly summary of what the paper experiments demonstrate.
- Added `experiments/release_manifest_v0.1.1.json`.

## Still Gated

- final trained checkpoints;
- exact split manifest from the final release-approved data root;
- per-image metric CSVs containing filenames or source metadata;
- paired statistical result JSONs derived from the final per-image table;
- final hardware profile under the selected deployment machine;
- final manuscript citation, arXiv URL, DOI, or journal citation.

## Validation

Run from the repository root:

```bash
python -m pytest -q
python -m compileall src scripts tests examples
```
