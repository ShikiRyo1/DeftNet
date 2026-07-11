# Reproducibility protocol

This document defines the public v0.2.0 reference protocol.

## Fixed operating point

- released-image split: `1408 / 176 / 176`;
- input: one grayscale `512 x 512` image;
- seeds: `0`, `42`, and `44`;
- threshold: `0.5`;
- test-time augmentation: disabled;
- Phase-I and Phase-II epochs: `60`;
- batch size: `4`;
- optimizer: AdamW, learning rate `1e-4`, weight decay `1e-5`;
- scheduler: cosine annealing, `T_max=60`, `eta_min=1e-6`;
- training: mixed precision when CUDA is available and gradient clipping at
  norm `5.0`;
- objective: `0.50 BCE + 0.50 Dice`.

clDice, cbDice, component, centerline, and boundary-distance metrics are
evaluation readouts, not optimized losses in the current protocol.

## Phase I: fold-perspective specialization

Create one deterministic five-fold assignment within the training split. E1
through E5 omit folds 1 through 5, respectively, and each expert is trained as
a complete encoder-decoder segmentor on the complementary four-fifths view.
The validation split selects the best checkpoint. Save the complete run
manifest and the encoder state dictionary.

## Phase II: frozen bank, HSAF, shared decoder

Load all five Phase-I encoder checkpoints, remove the Phase-I decoders, and
freeze the encoders. Train only feature adapters, HSAF routers, and the shared
decoder. The training script fails when a Phase-I checkpoint is missing; this
prevents the earlier failure mode in which randomly initialized experts could
be frozen silently.

## Evaluation

Aggregate each run separately, then report the mean and sample standard
deviation across seeds. Use one evaluator and one threshold for every model.
The public metric script provides:

- Dice, IoU, sensitivity, precision, specificity, and MCC;
- clDice and hard cbDice;
- branch-skeleton recovery, thin-structure recall, centerline continuity, and
  centerline break count;
- Betti-0 error, component-count ratio, and distance from the ideal ratio;
- HD95 and ASSD.

For paired comparisons, `scripts/paired_stats.py` reports a paired bootstrap
confidence interval, two-sided Wilcoxon signed-rank test, and matched-pairs
rank-biserial effect size. The Holm family is explicit and defaults to one
family per endpoint across baseline comparisons. The pairing unit is the CSV
identifier supplied with `--id-column`; use patient or sequence identifiers
instead of image identifiers whenever those grouping variables are available.

## Required run artifacts

Each release checkpoint should be accompanied by:

- exact YAML configuration and git commit;
- seed and best validation epoch;
- split and perspective-fold manifests;
- paths or hashes of the five Phase-I checkpoints;
- epoch history and final fixed-threshold per-image metrics;
- software and hardware profile.

## Legacy compatibility

v0.1.x configurations used expert identifiers `E5/E7/E9/E11/E12`. The v0.2.0
loader migrates the complete legacy roster and matching checkpoint keys to
`E1/E2/E3/E4/E5`. New artifacts should use only the canonical identifiers.

## Local verification without research data

```bash
pip install -e ".[dev]"
pytest -q
python -m compileall src scripts tests examples
python examples/synthetic_demo.py --output-dir examples_output
```
