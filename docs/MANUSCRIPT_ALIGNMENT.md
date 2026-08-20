# Manuscript-to-repository alignment

This document maps the final DEFT-Net manuscript to the public repository
without implying that unavailable third-party data or restricted run artifacts
are redistributed here.

## Method mapping

| Manuscript element | Public implementation or artifact |
|---|---|
| Five complementary four-fifths Phase-I views | `scripts/make_perspective_folds.py`, `scripts/train_phase1.py` |
| Complete Phase-I segmentor, decoder removal, frozen encoder reuse | `scripts/train_phase1.py`, `scripts/train.py` |
| Five heterogeneous experts E1-E5 | `src/deftnet/models/experts.py`, `src/deftnet/models/deftnet.py` |
| Depth-banded admission | `configs/deftnet_cmig.yaml`, `src/deftnet/models/deftnet.py` |
| Pixel-wise HSAF routing within the admitted set | `src/deftnet/models/hsaf.py` |
| F5 bottleneck input and F1-F4 fused skips | `src/deftnet/models/deftnet.py`, `src/deftnet/models/decoder.py` |
| One shared decoder and one final vessel mask | `src/deftnet/models/decoder.py` |
| Fixed 0.5/no-TTA evaluation | `scripts/evaluate.py`, `scripts/per_image_metrics.py` |
| Paired uncertainty and effect-size utilities | `scripts/paired_stats.py` |

## Public aggregate evidence

The stable aggregate files are:

- `experiments/pooled_public_fixed_notta_seed_summary.csv`;
- `experiments/mechanism_controls_three_seed_summary.csv`;
- `experiments/full_data_controls_three_seed_summary.csv`.

They preserve the finalized values from the corresponding version-suffixed
source tables. The source files remain in the repository for traceability.
Stable aliases remove internal manuscript revision identifiers from the public
interface; they do not recalculate or modify any metric.

## Configuration and parameter boundary

The manuscript experimental configuration reports:

- `4.48 M` trainable Phase-II parameters;
- `14.99 M` parameters in the complete inference graph;
- `17.82 ms` no-TTA latency at batch size 1;
- `0.52 GB` peak VRAM.

The bundled lightweight reference graph is regression-tested at `1,862,074`
trainable Phase-II parameters and `12,364,598` total inference parameters. It
implements the same mechanism but uses a lighter channel profile for public
execution and CI. The repository does not claim that profiling the lightweight
configuration reproduces the manuscript efficiency row.

## Evidence boundary

The main benchmark is a fixed released-image comparison over 1,760 harmonized
coronary XCA images split into 1,408 training, 176 validation, and 176 held-out
test images. It is suitable for comparing fixed segmentation mechanisms under
the declared target and operating point. Where patient, procedure, sequence,
or site identifiers are unavailable, it is not evidence of independence at
those levels and is not described as external clinical validation.

Third-party images, masks, private metadata, manuscript per-image exports, and
trained checkpoints are not committed unless their licenses and privacy review
permit redistribution.
