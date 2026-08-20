# Artifact roadmap

## Included in v0.3.0

- importable DEFT-Net model package and canonical config;
- Phase-I fold manifest and expert training entry points;
- Phase-II HSAF/shared-decoder training with checkpoint validation;
- fixed-threshold evaluation, per-image metrics, paired statistics, profiling,
  split-manifest, and qualitative-gallery utilities;
- stable benchmark, mechanism-control, and full-data-control aggregate CSVs;
- final-manuscript architecture image, model card, data scope, reproducibility protocol,
  release notes, citation metadata, and MIT license.

## Release-gated artifacts

- trained Phase-I and Phase-II checkpoints;
- exact manifests generated from the release-approved data root;
- per-image metric rows and paired-statistics JSON;
- hardware-matched latency/VRAM profile;
- clinical-image panels where redistribution terms permit;
- final paper DOI or accepted-manuscript citation when available.

## Regeneration commands

```bash
python scripts/make_split_manifest.py --data-root path/to/data_root \
  --output manifests/split_manifest.csv

python scripts/per_image_metrics.py \
  --pred-dir path/to/predictions --mask-dir path/to/data_root/test_masks \
  --model DEFT-Net --split test --threshold 0.5 \
  --output experiments/per_image_deftnet.csv

python scripts/paired_stats.py \
  --input experiments/per_image_all_models.csv --reference DEFT-Net \
  --holm-family per-metric --output experiments/paired_stats.json

python scripts/profile_model.py --config configs/deftnet_cmig.yaml \
  --device cuda --output experiments/profile_deftnet_cuda.json
```
