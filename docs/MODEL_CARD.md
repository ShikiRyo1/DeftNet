# Model Card

## Model

DeftNet / DEFT-Net is a binary semantic-segmentation architecture for thin
vessel structures, with the current research focus on coronary angiography
vessel-tree segmentation.

## Intended Use

- research on vessel segmentation;
- benchmarking heterogeneous frozen-expert feature fusion;
- educational reference for reproducible medical-image segmentation releases;
- extension to other thin-structure segmentation tasks after re-training and
  validation.

## Not Intended For

- clinical diagnosis;
- autonomous treatment decisions;
- patient-specific deployment without external validation;
- use on unrelated modalities without re-training and re-validation;
- claims of patient-level or procedure-level validation from image-level public
  benchmark evidence.

## Architecture

The model uses a two-stage workflow:

1. pretrain multiple expert encoder-decoder models;
2. discard expert decoders;
3. freeze expert encoders;
4. learn a depth-banded HSAF feature router plus a single shared decoder.

The default public config uses:

- shallow band: CNN-family experts;
- mid band: all five experts;
- deep band: Transformer-family experts.

## Current Public Evidence

The v77 public summary reports a 15-method fixed-protocol pooled public coronary
angiography benchmark. The DEFT-Net row is:

- Dice: `0.8264 +/- 0.0056`
- IoU: `0.7120 +/- 0.0079`
- sensitivity: `0.8490 +/- 0.0036`
- precision: `0.8186 +/- 0.0065`
- MCC: `0.8237 +/- 0.0055`
- clDice: `0.9025 +/- 0.0370`
- cbDice: `0.8086 +/- 0.0010`
- HD95: `35.8763 +/- 6.1363`
- ASSD: `5.7260 +/- 1.5839`

Values are three-seed mean `+/-` seed SD under fixed threshold `0.5` and no
test-time augmentation. HD95 and ASSD are guardrails, not claimed wins.

## Known Limitations

- The public release currently has code and aggregate result summaries, not raw
  data or final release checkpoints.
- The benchmark is image-level over public datasets unless future metadata
  verifies patient/procedure independence.
- HSAF is not claimed to dominate every simpler fusion rule on every metric.
- Phase-II trainable parameters and total inference footprint must be reported
  separately because frozen expert encoders remain active at inference.
