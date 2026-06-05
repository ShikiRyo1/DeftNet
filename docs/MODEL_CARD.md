# Model Card

## Model

DeftNet / DEFT-Net is a binary semantic-segmentation architecture for thin vessel
structures, with the main project focus on coronary DSA vessel segmentation.

## Intended Use

- research on vessel segmentation;
- benchmarking heterogeneous expert fusion;
- educational reference for reproducible medical-image segmentation releases.

## Not Intended For

- clinical diagnosis;
- autonomous treatment decisions;
- patient-specific deployment without external validation;
- use on unrelated modalities without re-training and re-validation.

## Architecture

The model uses a two-stage workflow:

1. pretrain multiple expert U-Net-style models;
2. keep only their encoders, freeze them, and train a fusion head plus a single
   decoder.

The default public config uses:

- shallow band: CNN-family experts;
- mid band: all five experts;
- deep band: Transformer-family experts.

## Known Limitations

- The public release currently has code and result summaries, not data or final
  release checkpoints.
- The audited DCA evidence is frame-level unless metadata later confirms a
  stronger patient/procedure split.
- HSAF is not claimed to dominate mean fusion on every metric.
- External clinical validation remains future work.
