# Experiment Design Summary

This file condenses the companion project-memory audit from the `Deft_Net`
experiment threads and the local manuscript planning folder. It is aligned to
manuscript working version **v77, 2026-06-15**.

## Final Scientific Story

DEFT-Net should not be framed as a universal segmenter, a clinical validation
system, or a method that wins because of one isolated gate. The strongest story
is narrower and more defensible:

> Frozen heterogeneous expert encoders provide complementary vessel evidence.
> A depth-banded HSAF router admits those experts at appropriate scales and fuses
> features before a single shared U-Net decoder makes the final segmentation.
> Under a fixed 0.5/no-TTA image-level public benchmark, this improves overlap,
> agreement, precision-risk, and vessel-tree consistency without claiming
> patient-level clinical validation.

## Current Evidence Chain

| Step | Purpose | Main evidence | Public repo status |
|---:|---|---|---|
| 0 | Protocol lock | fixed threshold 0.5, no TTA, seeds 0/42/44, fixed expert roster and K=5 | documented |
| 1 | Pooled public benchmark | 15-method leaderboard from DCA/DCAE, XCAD, ARCADE after binary mask harmonization | v77 CSV included |
| 2 | Statistical support | paired image-level tests, paired bootstrap, Wilcoxon/Holm, source-aware sensitivity | scripts included; per-image dump gated |
| 3 | Evidence-axis decomposition | Dice/IoU, MCC, precision-risk, clDice/cbDice, beta0 error, fragmentation | narrative and summary included |
| 4 | Vessel-tree preservation | branch recall, small-vessel recall, fragmentation, topology-error axes | aggregate values included where non-sensitive |
| 5 | Mechanism ablations | expert-family controls, depth-band admission, freezing, K-sweep | documented; selected summaries included |
| 6 | HSAF controls | output ensembles, feature mean, random routing, shuffled routing | v77 JSON included |
| 7 | Efficiency | trainable Phase-II params, total inference footprint, latency, VRAM | current values documented; final hardware profile gated |
| 8 | Strong controls | nnU-Net and XCA-specific DeepLabV3+ as reproduced controls; FR/SE/AngioNet as protocol-bound related systems | documented without overclaiming |
| 9 | Auxiliary stress tests | thin-structure robustness outside the primary coronary benchmark | bounded as stress tests, not clinical validation |

## Current Headline Numbers

The current main table is a pooled public coronary angiography benchmark with
three-seed mean `+/-` seed SD. The DEFT-Net row is:

| Model | Dice | IoU | Sens. | Prec. | MCC | clDice | cbDice | HD95 | ASSD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DEFT-Net | 0.8264 +/- 0.0056 | 0.7120 +/- 0.0079 | 0.8490 +/- 0.0036 | 0.8186 +/- 0.0065 | 0.8237 +/- 0.0055 | 0.9025 +/- 0.0370 | 0.8086 +/- 0.0010 | 35.8763 +/- 6.1363 | 5.7260 +/- 1.5839 |

Boundary-distance metrics are reported as guardrails. DeepLabV3+ remains
slightly better on HD95 and ASSD in the compact table, so those are not claimed
as DEFT-Net wins.

## Retired Earlier Protocols

Earlier project drafts used DCA-only or enhanced/sensitivity settings. Those
values are no longer the public headline:

- the earlier fixed-threshold DCA-only aggregate is superseded by the v77 pooled
  public benchmark;
- enhanced/sensitivity settings must not be mixed into the v77 primary table.

The v77 audit explicitly checked that retired protocol values do not appear in
the current manuscript text.

## HSAF vs Simpler Fusion

HSAF is described as a feature-level, depth-banded routing mechanism. The public
repo should not claim that HSAF dominates every simpler fusion rule on every
metric. The v77 mechanism control instead makes a more precise claim:

- output-level mask/logit/vote ensembles do not recover the same vessel-tree
  preservation profile;
- feature-mean fusion is a meaningful control but remains lower than HSAF in the
  v77 mechanism table for Dice, MCC, cbDice, branch recall, beta0 error, and
  fragmentation;
- random routing and shuffled HSAF weights test whether the routing weights
  themselves matter.

The structured control values are stored in
`experiments/fusion_hsaf_vs_mean.json`.

## Efficiency Wording

Efficiency must be written transparently:

- favorable: only the Phase-II HSAF plus shared decoder are trainable during
  fusion training, about 4.48M trainable parameters in the current manuscript
  table;
- transparent: frozen expert encoders are still part of inference, about 14.99M
  total inference parameters in the current manuscript table;
- not claimed: fastest model, smallest total compute, or universal edge
  deployment.

## Claim Boundary

The public benchmark is image-level over public datasets. Because
patient/procedure identifiers are not consistently available across the public
releases, the paper and this repo do not claim patient-level, procedure-level, or
external clinical validation.

## What Still Needs Completion Before a Final Paper Artifact

- Decide whether final trained checkpoints can be released.
- Generate the final split manifest from the exact release-approved data root.
- Release fixed-protocol per-image metrics only if filenames and metadata pass
  privacy/license review.
- Release paired statistical results once the per-image table is approved.
- Run final latency, VRAM, and compute profiling on the chosen hardware.
- Add the final manuscript citation, arXiv URL, DOI, or journal citation.
