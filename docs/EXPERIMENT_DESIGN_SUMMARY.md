# Experiment Design Summary

This file condenses the project-memory audit from the companion Codex thread
`审计论文实验设计` plus the local planning folder `_paper_planning_v15`.

## Final Scientific Story

DeftNet/DEFT-Net should not be framed as a universal segmenter or as a model that
wins because of one magical gate. The strongest story is narrower and more
defensible:

> A frozen bank of heterogeneous vessel-segmentation expert encoders, organized
> by depth-specific inductive bias, improves coronary DSA vessel segmentation at
> a fixed operating point while preserving vessel-tree structure and keeping the
> trainable fusion head small.

## Main Evidence Chain

| Exp | Purpose | Main Evidence | Status |
|---:|---|---|---|
| 0 | Protocol hygiene | split units, expert-test isolation, fixed operating point, no-TTA main protocol | documented |
| 1 | DCA primary benchmark | fixed 0.5 / no-TTA table, held-out `ttest_*`, `n=134` | complete aggregate table |
| 2 | Statistical support | paired bootstrap, Wilcoxon/Holm, rank-biserial effect size | scripts included; final per-image dump pending |
| 3 | Efficiency/cost | trainable params, total params, GMac, latency/VRAM/TTA cost | profiling script included; final hardware run pending |
| 4 | Mechanism ablations | K sweep, family LOO, depth-band, freezing, fusion | several measured; partial-unfreeze lower priority |
| 5 | FP mechanism | expert false-positive overlap, disagreement, FP maps | gallery script included; release-approved images pending |
| 6 | Coronary structure | clDice/cbDice, continuity, branch recall, break count | definitions and partial table ready |
| 7 | Cross-dataset | auxiliary robustness and precision-rank stability | auxiliary only, not external clinical validation |
| 8 | Failure modes | low contrast, overlap, tiny branches, modality mismatch | narrative ready; release-approved gallery pending |

## Key Audit Decisions

### 0.8196 vs 0.8410

The audit locks **0.8196 Dice** as the main DCA claim because it is fixed
threshold 0.5 + no-TTA. The **0.8410** number is a real project number but comes
from an enhanced/sensitivity protocol and must not be mixed into the primary
fixed-protocol table.

### HSAF vs Mean Fusion

HSAF should be described as adaptive, interpretable, and theoretically motivated.
The project should not claim HSAF universally beats mean fusion. In the current
fixed-protocol head-to-head, mean fusion is marginally ahead on some aggregate
metrics while HSAF is marginally ahead on several vessel-structure surrogates.

### Efficiency

Efficiency is useful but must be written transparently:

- favorable: small trainable head, lower parameter profile than HRNet/TransUNet;
- not claimed: fastest model, smallest total compute, universal edge deployment.

### Split Hygiene

DCA is treated as a strict held-out frame-level split. Unless patient/procedure
metadata later proves otherwise, the paper should not claim patient-level or
procedure-level validation.

### Architecture Naming

The local history includes HMES-Net gamma6 and DEFT-Net. The public repository
uses **DeftNet / DEFT-Net** and explains that DEFT-Net is the manuscript-facing
name.

### Depth-Band Consistency

The audited manuscript narrative uses Transformer-family admission at deep
levels `e4/e5`. Older notebooks show a legacy `E5,E11,E12` deep admission set.
Both configs are preserved:

- `configs/dca_five_expert.yaml`: audited manuscript default.
- `configs/legacy_three_deep_experts.yaml`: legacy checkpoint compatibility.

## What Still Needs Completion Before a Paper Submission

- Run the included fixed-protocol per-image metrics script on the final
  prediction masks.
- Run the included paired statistics script on the final all-model per-image
  table.
- Run the included latency/VRAM/compute profiling script under identical
  hardware and batch settings.
- Decide public release status for weights and any prediction masks.
- Add the final model checkpoint's exact config hash to the release notes.
- If possible, verify patient/procedure independence or report the frame-level
  limitation exactly as a limitation.
