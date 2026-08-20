# Experiment design summary

The experiment package separates benchmark performance, mechanism evidence,
and evidence boundaries.

## 1. Fixed benchmark

The primary table compares 15 shared-recipe models on the same released-image
split, threshold, no-TTA rule, evaluator, and seeds. It establishes the overall
overlap, agreement, centerline, and centerline-boundary profile before the
method is decomposed.

Current DEFT-Net aggregate:

| Dice | IoU | Sensitivity | Precision | MCC | clDice | cbDice |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8264 +/- 0.0056 | 0.7042 +/- 0.0081 | 0.8490 +/- 0.0036 | 0.8186 +/- 0.0065 | 0.8237 +/- 0.0055 | 0.9025 +/- 0.0240 | 0.8086 +/- 0.0010 |

## 2. Strong pipeline and task-specific comparators

nnU-Net and a within-resource DeepLabV3+ recipe are analyzed separately from
the shared-recipe leaderboard. This prevents a pipeline-level comparator from
being mislabeled as another architecture under the shared recipe. Dice is read
together with MCC and cbDice; the manuscript does not rely on a Dice-only
separation from nnU-Net.

## 3. Vessel-tree structure

Branch recovery, thin-structure recall, continuity, break count, Betti-0 error,
component-count ratio, HD95, and ASSD test whether overlap gains correspond to
the intended vessel-tree behavior. Isolated proxies are not treated as
interchangeable: for example, a model may reduce component count while losing
reference branches.

## 4. Fusion and admission controls

The same-bank controls answer distinct alternatives:

- mask average, logit average, and majority vote: output-level combination;
- feature mean: pre-decision fusion without learned routing;
- random and shuffled routing: learned/spatial routing specificity;
- CNN-only and Transformer-only: family restriction;
- flat and reversed bands: depth-policy controls;
- best single expert: single-expert bracket.

Depth-banded HSAF provides the strongest tested joint Dice/MCC/cbDice and
branch/component profile. Flat admission can remain better on an isolated
component-count proxy, so the allowed claim is a stronger joint profile rather
than universal dominance on every topology readout.

## 5. Full-data expert controls

Three rows separate two alternative explanations:

1. five same-architecture experts trained on all training images;
2. five heterogeneous experts trained on all training images;
3. the heterogeneous fold-perspective DEFT-Net bank.

The ordered aggregate profile is reported in
`experiments/full_data_controls_three_seed_summary.csv`. It supports the
coupled design point
tested here: architecture heterogeneity helps, and the fold-perspective bank
adds a second improvement when the roster and Phase-II mechanism are retained.
It does not claim that fold diversity alone or `K=5` is globally optimal.

## 6. Freezing and specialization

Frozen, unfrozen, and from-scratch variants test whether the Phase-I expert
states are useful and whether preserving those states during Phase II matters.
Trainable Phase-II and total inference parameters are reported separately.

The manuscript efficiency experiment uses `4.48 M` trainable Phase-II
parameters and a `14.99 M`-parameter complete inference graph. The bundled
lightweight reference implementation has a smaller channel profile and must
not be used to replace those manuscript values.

## 7. Statistical support

Each training result is repeated for seeds `0`, `42`, and `44`. Dataset-level
mean +/- seed SD describes training variability. When permitted per-image
exports are available, paired bootstrap intervals, Wilcoxon signed-rank tests,
and rank-biserial effects support matched comparisons. Multiplicity families
must be declared explicitly.

## 8. Robustness and qualitative analysis

Same-modality and auxiliary thin-structure evaluations are supportive stress
tests. Qualitative cases use a fixed TP/FP/FN rule after quantitative ranking to
show missed distal branches, false continuations, and broken geometry. These
analyses are not described as patient-level external clinical validation.
