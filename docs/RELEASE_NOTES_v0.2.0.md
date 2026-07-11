# v0.2.0 paper-aligned reproducibility update

Released: 2026-07-11

## Added

- deterministic five-fold Phase-I perspective manifest generation;
- Phase-I expert training with complementary four-fifths views;
- validated Phase-I checkpoint loading for Phase-II training;
- extended structural per-image evaluator;
- current benchmark, mechanism-control, and full-data-control CSVs;
- run manifests containing seed, phase, checkpoint provenance, and git commit.

## Corrected

- public expert names now match the manuscript (`E1`-`E5`);
- current objective is `0.50 BCE + 0.50 Dice`; topology metrics are
  evaluation-only;
- Phase-II no longer silently freezes random expert initialization;
- optimizer, scheduler, AMP, gradient clipping, and augmentation match the
  current protocol;
- README/model card values now use IoU `0.7042`, clDice SD `0.0240`, HD95
  `42.46 +/- 1.25 px`, and ASSD `7.21 +/- 0.58 px`;
- the architecture image no longer shows the retired KAN decoder, legacy expert
  identifiers, deep supervision, or topology-loss objective.
- parameter accounting is derived from the released plain U-Net graph:
  1,862,074 Phase-II trainable and 12,364,598 total inference parameters.

## Compatibility

The loader automatically migrates complete v0.1.x expert rosters and checkpoint
keys from `E5/E7/E9/E11/E12` to `E1/E2/E3/E4/E5`. New artifacts should use the
canonical identifiers.

## Validation

```bash
python -m pytest -q
python -m compileall src scripts tests examples
```

Research data and final trained checkpoints remain release-gated by source
licenses and privacy review.
