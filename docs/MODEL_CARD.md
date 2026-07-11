# Model card

## Model

DEFT-Net is a two-stage binary semantic-segmentation architecture for coronary
X-ray angiography vessel trees. It combines independently specialized frozen
CNN and Transformer encoders through fixed depth-banded admission and
pixel-wise HSAF routing before one shared U-Net decoder.

## Intended use

- research on coronary vessel-tree segmentation;
- controlled study of fold-perspective expert specialization;
- controlled study of heterogeneous feature routing before mask prediction;
- extension to other thin structures after task-specific training and
  validation.

## Not intended for

- autonomous diagnosis or treatment;
- patient-specific deployment without clinical validation;
- use on new modalities without retraining and validation;
- claims of patient-, procedure-, site-, or external clinical validation from
  released-image benchmark evidence alone.

## Architecture

Phase I trains five full segmentors on complementary four-fifths views of the
training set. Phase II retains only their encoders:

- E1: residual semantic CNN;
- E2: HRNet-lite CNN;
- E3: dense CNN;
- E4: pyramid Transformer;
- E5: Swin-lite Transformer.

E1-E3 are admitted at shallow scales, all experts at the middle scale, and
E4-E5 at deep scales. HSAF produces one fused tensor per scale. The shared
decoder uses the deepest fused tensor as its bottleneck input and the remaining
fused tensors as skip features.

## Current public aggregate result

Three-seed mean +/- seed SD at threshold `0.5`, no TTA:

- Dice: `0.8264 +/- 0.0056`
- IoU: `0.7042 +/- 0.0081`
- sensitivity: `0.8490 +/- 0.0036`
- precision: `0.8186 +/- 0.0065`
- MCC: `0.8237 +/- 0.0055`
- clDice: `0.9025 +/- 0.0240`
- cbDice: `0.8086 +/- 0.0010`
- HD95: `42.46 +/- 1.25 px`
- ASSD: `7.21 +/- 0.58 px`

HD95 and ASSD are boundary-distance guardrails and are not presented as
universal wins. Full results and mechanism controls are in `experiments/`.

## Efficiency

The current architecture has 1,862,074 trainable Phase-II parameters and
12,364,598 total inference parameters. Frozen experts remain active at
inference, so the trainable and total counts must be reported together. These
values are measured directly from the v0.2.0 model graph.

## Limitations

- Raw third-party datasets and trained weights are not included.
- Results are tied to the stated released-image protocol and evaluator.
- The current evidence supports the tested constrained combination; it does not
  establish that any component is uniquely optimal for every dataset or metric.
