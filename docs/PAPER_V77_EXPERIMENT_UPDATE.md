# Paper v77 Experiment Update

This note summarizes the current public-facing experiment story after the
manuscript audit chain that ended at **v77, 2026-06-15**.

## What Changed Since the First GitHub Release

- The headline result moved from an earlier DCA-only aggregate to a pooled public
  coronary angiography benchmark assembled from DCA/DCAE, XCAD, and ARCADE.
- The main number is now a three-seed fixed 0.5/no-TTA summary:
  Dice `0.8264 +/- 0.0056`.
- Earlier DCA-only and enhanced-setting values are retired from the current
  public headline to avoid mixing protocols.
- The manuscript now explicitly separates trainable Phase-II parameters from
  total inference footprint.
- The paper narrative treats HD95/ASSD as guardrails rather than claimed wins.
- FR-UNet, SE-RegUNet 4GF, and AngioNet are discussed as protocol-bound related
  systems unless fully reproduced under the locked protocol.
- nnU-Net and an XCA-specific DeepLabV3+ coronary recipe serve as stronger
  reproduced controls in the evidence chain.

## Current Experiment Logic

1. **Protocol lock**: define threshold `0.5`, no TTA, seeds `0/42/44`, frozen
   expert roster, K=5, and checkpoint rule before interpreting results.
2. **Primary benchmark**: compare DEFT-Net with fourteen reproduced baselines
   under the same fixed protocol.
3. **Paired statistics**: ask whether the same held-out images improve under
   DEFT-Net rather than only comparing aggregate means.
4. **Evidence-axis decomposition**: interpret the result through overlap,
   agreement, precision-risk, class imbalance, and vessel-tree consistency.
5. **Vessel-tree metrics**: use clDice/cbDice, branch recall, beta0 error, and
   fragmentation to show that the gain is not only a Dice-margin story.
6. **Mechanism controls**: test whether the method is just an output ensemble,
   feature averaging, random routing, or shuffled routing.
7. **Efficiency accounting**: report trainable and total footprint separately.
8. **Strong-control comparison**: check whether nnU-Net or a task-specific
   DeepLabV3+ recipe closes the gap.
9. **Auxiliary stress tests**: present cross-domain thin-structure results as
   robustness stress tests, not as clinical validation.

## Key Values Now Exposed in the Repo

- Full v77 seed-summary table:
  `experiments/pooled_public_fixed_notta_seed_summary_v77.csv`
- HSAF mechanism-control table:
  `experiments/fusion_hsaf_vs_mean.json`
- v77 release manifest:
  `experiments/release_manifest_v0.1.1.json`

## Boundaries to Keep in Interviews

Use this wording:

> DEFT-Net is a fixed-protocol public image-level benchmark method for coronary
> angiography vessel-tree segmentation. It is not yet a patient-level clinical
> validation study.

Avoid these claims:

- "clinical validation";
- "patient-level validated";
- "universal SOTA";
- "best on every metric";
- "lightweight fastest model";
- "just plug-and-play for clinical deployment".

## Resume-Friendly One-Liner

DEFT-Net: built and released a PyTorch research package for coronary angiography
vessel segmentation using frozen heterogeneous expert encoders, depth-banded
feature routing, fixed-protocol evaluation, reproducibility scripts, and
public-facing experiment documentation.
