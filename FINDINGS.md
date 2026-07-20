# Measured Findings

All numbers measured 2026-07-19 on the full 199-sample training set unless noted.
Reproduce with `scripts/analyze_dataset.py`.

---

## Dataset structure

| Fact | Value | Why it matters |
|---|---|---|
| Embryos in training | **2** — `6bba` (128), `44b6` (71) | Hidden test is embryo-disjoint. Random sample splits **leak**. |
| Volume shape | `(100, 64, 256, 256)` uint16, all 199 | Uniform -> batching is safe |
| Voxel scale | z=1.625, y=x=0.40625 µm | 4:1 anisotropy; the crop is a 104 µm **cube** |
| Annotated nodes | 133,318 total, **~2.8 per timepoint** | vs ~30-130 real nuclei -> a few % labelled |
| **GT divisions** | **151 total**, 0.76/video, 112/199 videos have zero | Extremely rare event |
| `test/` folder | 4 samples, all **copies from train** | Real hidden test is swapped in at rerun, ~= training-set size |

The sparse ground truth makes this a **positive-unlabelled** problem: unlabelled
cells are real cells, not negatives. The baseline handles this with
`--det-neg-weight 1e-2`.

---

## Baseline performance (held-out embryo `44b6`, 20 videos, det-threshold 0.96875)

> **CORRECTION (2026-07-20).** The public dataset
> `pilkwang/biohub-tracking-support-pack-**50ep**-v1` does **not** contain a
> 50-epoch checkpoint. Its `checkpoint_last.pth` carries `epoch = 402`, and its
> `ARTIFACT_MANIFEST.json` is named `biohub-tracking-support-pack-**400ep**-snapshot-v1`.
> The "50ep" in the dataset title is wrong. An earlier version of this file took
> the name at face value and concluded "longer training is worse" -- **that
> conclusion was backwards.** Corrected below.

| checkpoint | actual epochs | score | edge_jaccard | division_jaccard | div TP/FP/FN | node_recall |
|---|---|---|---|---|---|---|
| **support pack** (`50ep-v1`) | **402** | **0.8596** | 0.8567 | 0.0127 | 2 / **153** / 2 | 0.9987 |
| 350ep pin | 350 | 0.8395 | 0.8412 | 0.0119 | 2 / 164 / 2 | 0.9979 |
| 300ep pin | 300 | 0.8392 | 0.8409 | 0.0102 | 2 / 193 / 2 | 0.9972 |

**More training is BETTER, and monotonically so** over the range we can observe
(300 -> 350 -> 402). The +0.0204 gap from 350ep to 402ep is significant under the
paired bootstrap (CI [+0.0069, +0.0346], p=0.0024). Whether it has plateaued by
402 is **unknown** -- we have no checkpoint beyond it.

Checkpoint SHA256 (all three genuinely distinct):

    12f6881e...  support pack (402ep)
    dfb848aa...  350ep pin
    12b5d32a...  300ep pin

**Caveat:** these weights were probably trained on both embryos, so even the
held-out run is contaminated. A clean number needs retraining on one embryo only.

---

## The one big opportunity: division false positives

We emit ~155 forks to catch 2 of 4 real divisions. **Division precision ~1.3%.**

| scenario | division_jaccard | score contribution | delta |
|---|---|---|---|
| current | 0.0127 | 0.0013 | — |
| suppress all FPs, keep 2 TP | 0.50 | 0.050 | **+0.049** |
| catch all 4 cleanly | 1.00 | 0.100 | **+0.099** |

We over-predict divisions ~10x (7.75/video vs 0.76 in GT). The GT rate is
measurable per embryo; public notebooks reverse-engineer the same constant from
leaderboard feedback (`SAFE_DIV_GLOBAL_FRAC_CAP=0.00375`), which is exactly the
kind of constant that will not survive an embryo shift.

### A division TP needs a 5-generation window

From `src/tracking_cellmot/division_metrics.py`:

```
parent -> divider -> child1 -> grandchild1
                  -> child2 -> grandchild2
```

`_is_strongly_connected_division` requires the predicted graph to reproduce that
whole structure. **You get no credit for detecting a fork** -- you get credit for
having the lineage intact ~2 frames either side. Track fragmentation near a
division destroys the TP even when the fork is correct.

This explains why the strongest knob found by public notebooks is
`ILP_DISAPPEARANCE_WEIGHT=1.4` vs `ILP_APPEARANCE_WEIGHT=0.0` -- making track
termination 14x costlier than initiation suppresses fragmentation, which feeds
**both** metric terms.

---

## Negative results (things that do NOT work -- save yourself the time)

| Tried | Result |
|---|---|
| ~~**Longer training** (300ep, 350ep)~~ | **RETRACTED 2026-07-20.** This entry claimed longer training hurt. It was based on mis-reading the support pack as 50 epochs when it is **402**. The true ordering is 402ep > 350ep > 300ep -- more training **helps**. See the correction above. |
| **Anisotropic U-Net pooling** | Redundant. `--downsample 1,4,4` already makes the input isotropic (64x64x64 from a 104 µm cube). |
| **Better detection** | node_recall is already 0.9987. Nothing left here. |
| **Faster graph/assignment code** | Hungarian at real scale (65-194 nodes/frame) costs **<15 s across the entire test set**. Not a bottleneck. |
| **Embryo-wide division synchrony** | KS vs uniform D=0.131 p=0.010 -- weak. Divisions occupy 74/100 timepoints. Zebrahub images post-MBT where Kane & Kimmel synchrony (cycles 1-9) is lost by design. |
| **Optimising `node_recall`** | Actively misleading. Raising det-threshold 0.99 -> 0.9995 **drops** recall 0.97 -> 0.82 while **raising** score 0.5022 -> 0.5265. |

---

## Metric properties worth knowing

From `metrics.md` and `src/tracking_cellmot/metrics.py`:

- **Unmatched predicted nodes are NOT false positives.** They still count toward
  `N_pred` though.
- `adj = max(0, J * (1 - 0.1 * (N_pred - N_true) / N_true))`, and the multiplier
  is **not clamped at 1** -- under-predicting node count yields a *bonus*. The
  competition page confirms "it is possible for scores to exceed 1.0."
- `N_true` comes from `estimated_number_of_nodes` in the `.geff` metadata, which
  is **training-only**. At test time the metric knows it and you don't.
- Node matching: bipartite assignment, max 7 µm centroid distance.
- Scoring is micro-averaged (counts summed across videos before the Jaccard).

### The metric-hack situation

Public notebooks scoring 0.95+ (mid-July 2026) used **synthetic out-of-bounds
graph surgery**: a hub node at `t=-1000, z=y=x=-10000` adopting every track root,
plus fake division forks. Free because unmatched nodes aren't FPs.

**This is patched.** The current `metrics.py` contains a consecutive-frame filter
(`_target_t - _source_t == 1`) that deletes hub edges, and `cross_component_forks`
classifies the pattern as a division **FP**. On current code the hub is a net
negative. Organizers posted a "Division Metric exploit and patch" thread.

Do not build on metric exploitation. It is commodity, actively patched, and
maximally fragile across an embryo shift.

---

## What the ~0.90 public recipe actually is

Six top notebooks surveyed. Findings:

1. **Nobody trains.** All load the prepackaged support-pack weights unchanged
   (the `50ep-v1` dataset, which actually contains a **402-epoch** checkpoint).
2. Everything above ~0.89 comes from **post-processing geometry**, not modelling.
3. Pipeline: baseline weights -> ILP for node selection -> `motion_relink_edges`
   **discards the ILP's edges** and re-solves with a velocity-predicted Hungarian
   assignment -> gap closing -> division rate-capping -> short-track filtering.
4. Four "different" 0.902-0.909 notebooks are the **same codebase** with ~20 env
   vars changed.
5. **Zero of six do any cross-validation.** Constants like `0.96875` (=31/32),
   `4.66`, `0.00375` are leaderboard-probe artifacts. The splits file is written
   with an empty train list.

Their architecture is worth copying. Their **constants are not** -- re-derive
them locally against an embryo-held-out split.

---

## Measurement harness (A0.0) — validated 2026-07-19

`scripts/eval_per_sample.py` + `scripts/compare_configs.py`. Three-way self-test on
held-out embryo `44b6` (20 videos), metric = `score`:

| test | delta | 95% CI | p | verdict |
|---|---|---|---|---|
| null (402ep vs itself) | +0.0000 | [0, 0] | 1.000 | not significant |
| known-different (402ep vs 300ep) | **-0.0204** | [-0.0346, -0.0069] | 0.0024 | **SIGNIFICANT** |
| known-similar (300ep vs 350ep) | +0.0003 | [-0.0186, +0.0142] | 0.981 | not significant |

Calibrated in both directions: catches a 0.020 difference, does not call 0.0003 real.

**Resolution floor: ~+/-0.015 on 20 videos.** Anything smaller is unmeasurable at that
sample size -- use the full 71-video fold for finer comparisons.

Why a bootstrap and not a t-test: `adj_edge_jaccard` is weighted by
`w_i = TP+FP+FN` and `division_jaccard` is micro-averaged, so a paired t-test on
unweighted per-video means tests a DIFFERENT quantity than the leaderboard reports.
The bootstrap resamples videos as pairs and recomputes the full weighted statistic.

`eval_per_sample.py` hard-errors when `n_adj != n` -- a GT geff missing
`estimated_number_of_nodes` silently drops that sample from `score` while it still
contributes to the micro-averaged terms.

Division metrics with frame tolerance (BC(i)) need `traccuracy` (added as the
`metrics` extra; API is `DivisionMetrics(max_frame_buffer=N)`). The repo's own
`division_metrics.py` has no frame-tolerance parameter.

---

## A0.2 — `--use-ilp` is worth +0.042 (measured 2026-07-20)

Held-out embryo `44b6`, 20 videos, 402ep weights, `--det-threshold 0.96875`,
paired bootstrap over the same videos.

| metric | greedy | **ILP** | delta | 95% CI | p |
|---|---|---|---|---|---|
| **score** | 0.8596 | **0.9012** | **+0.0416** | [+0.0317, +0.0527] | 0.0000 |
| adj_edge_jaccard | 0.8583 | 0.9012 | +0.0429 | [+0.0335, +0.0539] | 0.0000 |
| division_jaccard | 0.0127 | **0.0000** | -0.0127 | [-0.0339, +0.0000] | 0.22 (n.s.) |

Counts:

| | div TP/FP/FN | edge TP/FP/FN | n_pred |
|---|---|---|---|
| greedy | 2 / **153** / 2 | 4358 / 398 / 331 | 690,956 |
| ILP | **0 / 0 / 4** | 4392 / **224** / 297 | 636,528 |

**The ILP predicts exactly ZERO divisions** at the default `--ilp-division-weight 1.0`.
A division is a pure cost the solver never finds worth paying. That single change:

- removed all 153 false forks, cutting edge FP 398 -> 224 (most of the +0.043)
- cut `n_pred` by 54k, which also helps the adjustment multiplier
- **but removed the 2 true divisions too**, zeroing the division term

One flag reaches the clean public recipe's level (~0.90-0.909), with embryo-holdout
validation rather than leaderboard probing. And the entire 0.1 division block now
sits behind a **single scalar knob** (`--ilp-division-weight`), currently untuned.

### Runtime cost

| | s/video | projected for ~199-sample hidden test |
|---|---|---|
| greedy | 21.6 | 1.2 h |
| **ILP** | **81.6** | **4.5 h** of a 12 h cap |

ILP is 3.8x slower. Still comfortable alone, but it constrains how much TTA can be
stacked on top — 8x TTA on the detection branch would not fit alongside it.
