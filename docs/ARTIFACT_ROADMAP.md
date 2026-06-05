# Artifact Roadmap

This repository is currently a public code release for DeftNet / DEFT-Net. It
contains the cleaned implementation, configs, aggregate public result summaries,
and scripts for producing paper-grade artifacts when the final data, checkpoint,
and manuscript decisions are fixed.

## Included Now

- model package and configs;
- train/evaluate/infer scripts;
- fixed-threshold per-image metric script;
- paired statistical testing script;
- split-manifest checksum script;
- model profiling script;
- qualitative gallery script;
- synthetic API demo;
- model card, data policy, reproducibility protocol, citation metadata, and MIT
  license.

## Gated Until Final Paper Decisions

- final trained checkpoint, if release is approved;
- exact split manifest generated from the final data root;
- fixed-protocol per-image metric CSV for the final prediction masks;
- paired bootstrap and Wilcoxon/Holm result JSON from the final per-image CSV;
- final hardware-matched latency, VRAM, and compute profile;
- final paper citation, arXiv URL, DOI, or journal citation;
- clinical-image qualitative gallery, only if redistribution and privacy terms
  allow it.

## Recommended Post-Paper Commands

```bash
python scripts/make_split_manifest.py \
  --data-root path/to/data_root \
  --output manifests/dca_split_manifest.csv

python scripts/per_image_metrics.py \
  --pred-dir path/to/deftnet_predictions \
  --mask-dir path/to/data_root/true_test_masks \
  --model DeftNet \
  --split true_test \
  --threshold 0.5 \
  --output experiments/per_image_metrics_deftnet_fixed_0.5_no_tta.csv

python scripts/paired_stats.py \
  --input experiments/per_image_metrics_all_models_fixed_0.5_no_tta.csv \
  --reference DeftNet \
  --output experiments/paired_stats_fixed_0.5_no_tta.json

python scripts/profile_model.py \
  --config configs/dca_five_expert.yaml \
  --device cuda \
  --output experiments/profile_deftnet_cuda.json

python scripts/make_gallery.py \
  --image-dir path/to/data_root/true_test_images \
  --mask-dir path/to/data_root/true_test_masks \
  --pred-dir path/to/deftnet_predictions \
  --output assets/qualitative_gallery.png
```
