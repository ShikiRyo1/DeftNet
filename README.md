# DEFT-Net

**Depth-banded pre-decision routing of frozen heterogeneous experts for
coronary X-ray angiography vessel-tree segmentation.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-2E7D32.svg)](LICENSE)
[![Project version](https://img.shields.io/badge/version-0.3.0-146C94.svg)](docs/RELEASE_NOTES_v0.3.0.md)
[![CI](https://github.com/ShikiRyo1/DeftNet_coronary-vessel-semantic-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/ShikiRyo1/DeftNet_coronary-vessel-semantic-segmentation/actions/workflows/ci.yml)

This repository contains the mechanism-aligned reference implementation,
configuration, evaluation utilities, and public aggregate artifacts associated
with the final DEFT-Net manuscript. DEFT-Net preserves local and contextual
expert evidence until a single vessel-mask decision is formed. It is not an
output ensemble.

![DEFT-Net architecture](assets/architecture.png)

## Method

DEFT-Net is trained in two phases:

1. **Complementary-view specialization.** Five complete heterogeneous
   segmentors are trained independently. Expert `Ei` sees four fifths of the
   fixed training split and omits internal fold `i`. Validation selects the
   Phase-I checkpoint.
2. **Frozen feature routing.** The five Phase-I decoders are removed and the
   selected encoders are frozen. Per-scale adapters align their same-scale
   features. A fixed depth-band admits only the intended expert family at each
   scale, and HSAF computes pixel-wise weights within that admitted set. One
   shared U-Net decoder produces the final mask.

The expert bank is:

| Expert | Public architecture description | Family | Phase-I view |
|---|---|---|---|
| E1 | residual semantic CNN | CNN | training set except internal fold 1 |
| E2 | HRNet-lite CNN | CNN | training set except internal fold 2 |
| E3 | dense CNN | CNN | training set except internal fold 3 |
| E4 | pyramid Transformer | Transformer | training set except internal fold 4 |
| E5 | Swin-lite Transformer | Transformer | training set except internal fold 5 |

The fixed admission sets are:

| Decoder scale | Admitted experts | Intended evidence |
|---|---|---|
| `S1`, `S2` | E1, E2, E3 | high-resolution local structure |
| `S3` | E1, E2, E3, E4, E5 | cross-family competition |
| `S4`, `S5` | E4, E5 | wider contextual structure |

At scale `s`, all five aligned maps condition router `r_s`. The hard admission
mask is applied before the temperature-scaled softmax (`tau = 1.5`), so only
experts in the prespecified set `A_s` receive non-zero mixture weights. The
deepest fused tensor `F5` initializes the bottleneck path; `F1`-`F4` are fused
skip tensors for the shared decoder.

## Fixed-protocol evidence

The pooled public-image benchmark contains 1,760 released coronary XCA images:
1,500 ARCADE SYNTAX images, 134 supervised DCA/DCAE image-mask pairs, and 126
supervised XCAD image-mask pairs. After source-specific binary harmonization,
the fixed image-level split is 1,408 training, 176 validation, and 176 held-out
test images.

The headline benchmark uses seeds `0`, `42`, and `44`, a fixed threshold of
`0.5`, and no test-time augmentation.

| Model | Dice | IoU | Sens. | Prec. | MCC | clDice | cbDice | HD95 (px) | ASSD (px) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **DEFT-Net** | **0.8264 +/- 0.0056** | **0.7042 +/- 0.0081** | **0.8490 +/- 0.0036** | **0.8186 +/- 0.0065** | **0.8237 +/- 0.0055** | **0.9025 +/- 0.0240** | **0.8086 +/- 0.0010** | 42.46 +/- 1.25 | 7.21 +/- 0.58 |
| U-Mamba | 0.8103 +/- 0.0100 | 0.6906 +/- 0.0143 | 0.8186 +/- 0.0105 | 0.8147 +/- 0.0090 | 0.8066 +/- 0.0099 | 0.8999 +/- 0.0422 | 0.7908 +/- 0.0064 | 47.4616 +/- 4.9820 | 8.4582 +/- 1.3173 |
| SegFormer | 0.8084 +/- 0.0075 | 0.6861 +/- 0.0104 | 0.8272 +/- 0.0103 | 0.8026 +/- 0.0039 | 0.8045 +/- 0.0074 | 0.8963 +/- 0.0444 | 0.7917 +/- 0.0032 | 45.2430 +/- 6.5060 | 7.7948 +/- 1.5004 |

The public result files use stable names:

- [`pooled_public_fixed_notta_seed_summary.csv`](experiments/pooled_public_fixed_notta_seed_summary.csv): complete 15-method benchmark;
- [`mechanism_controls_three_seed_summary.csv`](experiments/mechanism_controls_three_seed_summary.csv): fusion, routing, family, admission, and output controls;
- [`full_data_controls_three_seed_summary.csv`](experiments/full_data_controls_three_seed_summary.csv): same-architecture full-data, heterogeneous full-data, and fold-perspective expert-bank ladder.

HD95 and ASSD are boundary-distance guardrails rather than universal-win
claims. Mechanism and full-data controls are reported separately from the
leaderboard so that benchmark ranking and causal interpretation are not
conflated.

## Installation

```bash
git clone https://github.com/ShikiRyo1/DeftNet_coronary-vessel-semantic-segmentation.git
cd DeftNet_coronary-vessel-semantic-segmentation
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Dataset layout

```text
data_root/
  train_images/    train_masks/
  val_images/      val_masks/
  test_images/     test_masks/
```

Images are loaded as single-channel inputs and resized to `512 x 512`; masks
use nearest-neighbour resizing. Third-party dataset files are not
redistributed. See [`docs/DATA.md`](docs/DATA.md) for source counts,
harmonization, provenance checks, and claim boundaries.

## Reproduce the two-stage protocol

### 1. Create the internal perspective-fold manifest

```bash
python scripts/make_perspective_folds.py \
  --data-root path/to/data_root \
  --output manifests/perspective_folds_seed2026.json \
  --folds 5 --seed 2026
```

The assignment is deterministic and near-balanced. Expert `Ei` omits internal
fold `i` during Phase I. These folds lie entirely within the global training
split and do not replace the validation or held-out test partitions.

### 2. Train the five complete Phase-I expert segmentors

```bash
python scripts/train_phase1.py --data-root path/to/data_root \
  --perspective-manifest manifests/perspective_folds_seed2026.json \
  --expert E1 --omit-fold 0 --seed 42 --output-dir runs/phase1
```

Repeat for `E2`-`E5` with omitted folds `1`-`4`. The selected checkpoint stores
the complete Phase-I segmentor and an `encoder_state_dict` for Phase II.

### 3. Train HSAF and the shared decoder

```bash
python scripts/train.py --data-root path/to/data_root --seed 42 \
  --expert-checkpoint E1=runs/phase1/E1_omit_fold0_seed42/best.pth \
  --expert-checkpoint E2=runs/phase1/E2_omit_fold1_seed42/best.pth \
  --expert-checkpoint E3=runs/phase1/E3_omit_fold2_seed42/best.pth \
  --expert-checkpoint E4=runs/phase1/E4_omit_fold3_seed42/best.pth \
  --expert-checkpoint E5=runs/phase1/E5_omit_fold4_seed42/best.pth
```

Run the complete two-stage procedure for seeds `0`, `42`, and `44`. Phase-II
training refuses to proceed when an expert checkpoint is missing.
`--allow-random-experts` is limited to synthetic smoke tests and must not be
used for manuscript results.

## Evaluation and inference

```bash
python scripts/evaluate.py \
  --checkpoint runs/deftnet_phase2/seed42/best.pth \
  --data-root path/to/data_root --split test --threshold 0.5

python scripts/infer.py \
  --checkpoint runs/deftnet_phase2/seed42/best.pth \
  --image example.png --output mask.png --threshold 0.5
```

Additional utilities:

- `scripts/per_image_metrics.py`: fixed-threshold overlap, centerline, branch,
  component, topology, and boundary metrics;
- `scripts/paired_stats.py`: paired bootstrap, Wilcoxon, rank-biserial effect
  size, and explicitly scoped Holm correction;
- `scripts/make_split_manifest.py`: image/mask checksums and split provenance;
- `scripts/profile_model.py`: trainable and total parameters, MACs, latency,
  and VRAM;
- `scripts/make_gallery.py`: deterministic TP/FP/FN qualitative panels.

## Parameter-accounting boundary

Two configurations must not be conflated:

| Configuration | Phase-II trainable | Total inference graph | Purpose |
|---|---:|---:|---|
| Manuscript experimental configuration | 4.48 M | 14.99 M | final reported efficiency experiment |
| Bundled lightweight reference configuration | 1,862,074 | 12,364,598 | executable mechanism reference and CI |

The final manuscript additionally reports 17.82 ms no-TTA latency and 0.52 GB
peak VRAM for its experimental configuration. The public reference graph uses
the same two-phase mechanism and admission policy but a lighter channel
configuration. Its regression-tested parameter counts are not substitutes for
the manuscript efficiency row. Frozen expert encoders remain active at
inference in both cases, so trainable and total parameters must always be
reported together.

## Evidence and release boundary

- The repository contains code, configs, aggregate tables, and non-sensitive
  figures. It does not redistribute third-party angiograms, masks, private
  metadata, trained weights, or manuscript per-image exports.
- The main evidence is a released-image, fixed-protocol benchmark. It is not
  patient-, procedure-, sequence-, site-, or external clinical validation
  where the public resources do not expose the necessary identifiers.
- The final evidence supports the tested constrained combination. It does not
  establish that any isolated component, `K = 5`, or the expert roster is
  universally optimal.
- Historical versioned artifacts are retained for traceability; stable aliases
  identify the current public aggregate tables.

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md),
[`docs/DATA.md`](docs/DATA.md),
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), and
[`docs/MANUSCRIPT_ALIGNMENT.md`](docs/MANUSCRIPT_ALIGNMENT.md).

## Citation

Until a DOI-bearing manuscript record is available, cite the software release:

```bibtex
@software{wu2026deftnet,
  author  = {Yuhui Wu and Zhe Chen and Kai Li and Rob M. Ewing and Zehor Belkhatir and Yihua Wang},
  title   = {DEFT-Net: Depth-banded pre-decision routing of frozen heterogeneous experts for coronary X-ray angiography vessel-tree segmentation},
  year    = {2026},
  version = {0.3.0},
  url     = {https://github.com/ShikiRyo1/DeftNet_coronary-vessel-semantic-segmentation}
}
```

## License

The code is released under the MIT License. Dataset-specific licenses and
terms remain in force for all external data used with this repository.
