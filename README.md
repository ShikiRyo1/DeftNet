# DeftNet / DEFT-Net

**Depth-banded feature-level routing of frozen expert encoders for coronary
angiography vessel-tree segmentation.**

This repository is the public code and experiment-summary package for the
DEFT-Net research project. DEFT-Net trains heterogeneous vessel-segmentation
experts, discards their decoders, freezes the expert encoders, and learns a
depth-banded HSAF fusion module before a single U-Net decoder predicts the final
binary vessel mask.

![DeftNet architecture](assets/architecture.png)

## Release Status

Current public package: **v0.1.1 candidate, aligned to manuscript working version
v77, 2026-06-15**.

The repository contains code, configs, non-sensitive figures, v77 aggregate
experiment summaries, and reproducibility utilities. It does not redistribute
third-party datasets, clinical images, trained checkpoints, prediction masks, or
unpublished manuscript source files.

## Highlights

- **Frozen heterogeneous experts**: CNN-style, HRNet-style, DenseNet-style, SSM,
  and Transformer-style encoders are trained as separate specialists, then
  frozen for fusion-stage training.
- **Depth-banded HSAF routing**: shallow scales admit local-detail experts,
  the middle scale admits all experts, and deep scales admit Transformer-family
  experts.
- **Single decoder, not output ensembling**: expert decoder outputs are not
  averaged, voted, or ensembled at inference; fusion happens at the feature
  level before one shared decoder.
- **Fixed operating point**: the headline benchmark uses threshold `0.5`, no
  test-time augmentation, and the same pooled public coronary angiography
  benchmark.
- **Job-ready release shape**: installable PyTorch package, training/evaluation
  scripts, configs, result tables, model card, data policy, release notes, and
  smoke tests.

## Main v77 Result

The current manuscript-facing result is a **15-method fixed-protocol pooled
public coronary angiography benchmark** assembled from DCA/DCAE, XCAD, and
ARCADE after binary vessel-mask harmonization. Values below are three-seed mean
`+/-` seed SD for seeds `0`, `42`, and `44`.

| Rank | Model | Dice | IoU | Sens. | Prec. | MCC | clDice | cbDice | HD95 | ASSD |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **DEFT-Net** | **0.8264 +/- 0.0056** | **0.7120 +/- 0.0079** | **0.8490 +/- 0.0036** | **0.8186 +/- 0.0065** | **0.8237 +/- 0.0055** | **0.9025 +/- 0.0370** | **0.8086 +/- 0.0010** | 35.8763 +/- 6.1363 | 5.7260 +/- 1.5839 |
| 2 | U-Mamba | 0.8103 +/- 0.0100 | 0.6906 +/- 0.0143 | 0.8186 +/- 0.0105 | 0.8147 +/- 0.0090 | 0.8066 +/- 0.0099 | 0.8999 +/- 0.0422 | 0.7908 +/- 0.0064 | 47.4616 +/- 4.9820 | 8.4582 +/- 1.3173 |
| 3 | SegFormer | 0.8084 +/- 0.0075 | 0.6861 +/- 0.0104 | 0.8272 +/- 0.0103 | 0.8026 +/- 0.0039 | 0.8045 +/- 0.0074 | 0.8963 +/- 0.0444 | 0.7917 +/- 0.0032 | 45.2430 +/- 6.5060 | 7.7948 +/- 1.5004 |
| 4 | UNet++ | 0.8070 +/- 0.0040 | 0.6857 +/- 0.0066 | 0.8296 +/- 0.0078 | 0.8006 +/- 0.0077 | 0.8037 +/- 0.0030 | 0.8781 +/- 0.0207 | 0.7895 +/- 0.0046 | 38.7974 +/- 11.3550 | 5.9801 +/- 2.2673 |
| 5 | U-Net | 0.8060 +/- 0.0028 | 0.6844 +/- 0.0035 | 0.8289 +/- 0.0065 | 0.7997 +/- 0.0070 | 0.8028 +/- 0.0025 | 0.8853 +/- 0.0368 | 0.7910 +/- 0.0029 | 38.2145 +/- 11.2777 | 6.0332 +/- 2.3886 |
| 6 | DeepLabV3+ | 0.8030 +/- 0.0041 | 0.6795 +/- 0.0062 | 0.8252 +/- 0.0065 | 0.7970 +/- 0.0079 | 0.7996 +/- 0.0036 | 0.8871 +/- 0.0175 | 0.7900 +/- 0.0032 | **35.3534 +/- 8.0954** | **5.4631 +/- 1.9527** |

The full 15-method, nine-metric table is committed as
[`experiments/pooled_public_fixed_notta_seed_summary_v77.csv`](experiments/pooled_public_fixed_notta_seed_summary_v77.csv).
HD95 and ASSD are treated as distance-metric guardrails, not as claimed wins.

## Installation

```bash
git clone https://github.com/ShikiRyo1/DeftNet.git
cd DeftNet
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Dataset Layout

The scripts expect binary masks in a simple folder layout:

```text
data_root/
  train_images/
  train_masks/
  val_images/
  val_masks/
  test_images/
  test_masks/
  true_test_images/      # optional held-out final split
  true_test_masks/
```

Third-party datasets and clinical images are not included in this repository.
See [`docs/DATA.md`](docs/DATA.md) for the release and citation policy.

## Training

```bash
python scripts/train.py \
  --config configs/dca_five_expert.yaml \
  --data-root path/to/data_root \
  --output-dir runs/deftnet_dca
```

The default config freezes all expert encoders during the fusion stage. If you
need to reproduce an older checkpoint that used a three-expert deep band, use
`configs/legacy_three_deep_experts.yaml`.

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/dca_five_expert.yaml \
  --checkpoint runs/deftnet_dca/best.pth \
  --data-root path/to/data_root \
  --split true_test \
  --threshold 0.5
```

## Inference

```bash
python scripts/infer.py \
  --config configs/dca_five_expert.yaml \
  --checkpoint runs/deftnet_dca/best.pth \
  --image example.png \
  --output mask.png
```

## Repository Map

```text
src/deftnet/                 installable PyTorch package
scripts/                     train, evaluate, infer, statistics, profiling
configs/                     audited and legacy architecture configs
experiments/                 public result tables and ablation summaries
docs/                        protocol, data, model card, release notes
assets/                      architecture and non-sensitive result figures
tests/                       CPU smoke tests
examples/                    synthetic API demo
```

## Reproducibility Utilities

The repository includes utility scripts for paper-grade artifact generation once
final checkpoints and prediction masks are release-approved:

- `scripts/make_split_manifest.py`: generate checksum manifests for data splits.
- `scripts/per_image_metrics.py`: compute fixed-threshold per-image metrics.
- `scripts/paired_stats.py`: run paired bootstrap and Wilcoxon/Holm statistics.
- `scripts/profile_model.py`: profile parameters, approximate MACs, latency, and
  VRAM.
- `scripts/make_gallery.py`: create qualitative TP/FP/FN overlay galleries.

See [`docs/ARTIFACT_ROADMAP.md`](docs/ARTIFACT_ROADMAP.md).

## Experiment Design

The current evidence chain is:

1. lock the fixed protocol and operating point;
2. report the pooled public coronary benchmark;
3. support the aggregate table with paired statistics and source-aware checks;
4. decompose the result into overlap, precision-risk, agreement, and vessel-tree
   axes;
5. test mechanism ablations for expert diversity, freezing, and depth-banding;
6. test HSAF against output-level ensemble and feature-mean controls;
7. report efficiency with both trainable and total inference footprint;
8. compare against strong task-specific controls without overclaiming
   unreproduced systems;
9. keep auxiliary cross-domain thin-structure tests bounded as stress tests.

See [`docs/EXPERIMENT_DESIGN_SUMMARY.md`](docs/EXPERIMENT_DESIGN_SUMMARY.md),
[`docs/PAPER_V77_EXPERIMENT_UPDATE.md`](docs/PAPER_V77_EXPERIMENT_UPDATE.md),
and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Important Limitations

- The current public package contains code, configs, aggregate result tables, and
  non-sensitive figures. Dataset redistribution and trained checkpoints require
  separate release decisions.
- The benchmark is image-level over public datasets. It is not claimed as
  patient-level, procedure-level, or external clinical validation because those
  identifiers are not consistently available across the public releases.
- DEFT-Net is parameter-efficient in the Phase-II trainable-head sense; frozen
  expert encoders are still part of the total inference footprint.
- HSAF is presented as a feature-level routing mechanism, not as proof that it
  dominates every simpler fusion rule on every metric.

## Citation

If this repository helps your research, please cite the repository for now:

```bibtex
@software{deftnet2026,
  title  = {DeftNet / DEFT-Net: Depth-Banded Frozen-Expert Fusion for Coronary Vessel Segmentation},
  year   = {2026},
  url    = {https://github.com/ShikiRyo1/DeftNet}
}
```

## License

MIT. Dataset-specific licenses and terms still apply to any external data used
with this code.
