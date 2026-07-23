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

### The ILP division weight is a dead end (swept 2026-07-20)

Held-out `44b6`, 20 videos, 402ep weights, `--use-ilp`, sweeping `--ilp-division-weight`:

| division_weight | score | division_jaccard | div TP/FP/FN | behaviour |
|---|---|---|---|---|
| **1.0** (default) | **0.9012** | 0.0000 | 0 / 0 / 4 | full suppression |
| 0.0 | 0.8736 | 0.0127 | 2 / 154 / 2 | divisions leak in |
| -0.5 | 0.8596 | 0.0127 | 2 / 153 / 2 | **collapses to greedy** |

Monotone and saturated in both directions. `>= 1.0` already emits zero divisions so
nothing larger can help; `-0.5` reproduces the greedy baseline exactly (same score
to four decimals, same counts). The `-1.0` and `-2.0` runs were cancelled once
saturation was established.

**Making divisions cheaper does not buy good divisions -- it buys the same garbage
greedy produced** (154 FP for 2 TP, precision ~1.3%). The trade is lopsided:
recovering `division_jaccard = 0.0127` is worth **+0.0013** on score but costs
**-0.028** on the edge term, because false forks fragment tracks and fragmentation
hurts `adj_edge_jaccard` far more than two real divisions help.

**At current division precision, suppressing divisions entirely is optimal**, and
the shipped default is already right. The 0.1 division block is reachable only with
better division *evidence* -- a classifier or geometric features -- not by
re-weighting the solver. Turning this knob before precision improves is strictly
harmful.

Same structural lesson as the `node_recall` trap: the metric punishes
indiscriminate prediction.

---

## Phase 1 — Post-processing (fold 0, full 71-video held-out `44b6`)

Null baseline recomputed on the **full 71 videos** (earlier numbers used a
20-video subset): `score = 0.8981`, `adj_edge_jaccard = 0.8943`,
`division_jaccard = 0.0385`, `n_adj == n == 71` (clean, no sample dropout).

### A1.1 — Minimum track-length filter — SHIPS (+0.0126)

Drop every weakly-connected lineage component whose temporal span is < N frames,
**exempting any component that contains a division** (a division TP needs an intact
5-generation window). Sweep on fold 0:

| N | score | Δ vs null |
|---|---|---|
| null (off) | 0.8981 | — |
| 3 | 0.9067 | +0.0086 |
| **4** | **0.9107** | **+0.0126** |
| 6 (public value) | 0.9037 | +0.0056 |
| 8 | 0.8908 | −0.0073 |
| 10 | 0.8703 | −0.0278 |
| 12 | 0.8497 | −0.0484 |

Smooth unimodal curve peaking at **N=4**. Paired bootstrap N=4 vs null:
**+0.0126, 95% CI [+0.0091, +0.0162], p=0.0000 (10k resamples) — significant.**
`division_jaccard` is unchanged at 0.0385 across all N (keep-division-components
works — no divisions lost to the filter).

**The local optimum is 4, not the public 6.** The public constant costs −0.0070
relative to the re-derived value — a concrete instance of why LB-probed constants
do not transfer to an embryo-held-out split.

Caveats: (1) N was chosen as the argmax of a sweep and its p-value reported on the
same fold, so +0.0126 is mildly optimistic (winner's curse); the curve's smoothness
argues it is not noise. (2) Needs confirmation on **fold 1** (train `44b6` →
test `6bba`), which requires a GPU prediction run — pending. Above N≈8 the filter
turns harmful by deleting real short lineages.

### A1.4 — Motion relink (constant-velocity, gated Hungarian) — SHIPS (+0.0074)

Reconnect prematurely-terminated tracks (out-degree-0 nodes not in the last frame)
to next-frame orphan starts (in-degree-0 nodes not in the first frame), matching a
constant-velocity prediction to candidates by Hungarian assignment within a µm gate.
Because it only ever links a loose end to an orphan, it can create **neither a
division nor a merge** — it repairs single tracks only.

Runs *before* the length filter (rescue fragments into longer tracks, then filter).
Gate sweep, stacked on min-track-len=4:

| gate (µm) | score | Δ vs mtl4 |
|---|---|---|
| off (mtl4 only) | 0.9107 | — |
| 6 | 0.9175 | +0.0068 |
| **8** | **0.9181** | **+0.0074** |
| 10 | 0.9132 | +0.0025 |

Optimum at **8 µm** — again between the public tight/relaxed constants (6/10).
Paired bootstrap relink8+mtl4 vs mtl4: **+0.0074, 95% CI [+0.0037, +0.0114],
p=0.0000 — significant.** Relink alone (no length filter) vs null: 0.9049 (+0.0068).

`division_jaccard` unchanged at 0.0385 — relink rescues single tracks, not the
5-generation windows a division TP needs (that is A1.3 gap-closing's job).

**Phase 1 chain to date:** null 0.8981 → min-track-len 4 → motion-relink 8 µm →
**0.9181** (+0.0199 over null, p=0.0000). Same fold-1 confirmation caveat applies.

### Fold-1 confirmation — the Phase 1 chain generalizes across embryos

The chain (motion-relink 8 µm + min-track-len 4) was tuned entirely on fold 0
(held-out `44b6`). Applied **unchanged** to fold 1 (train `44b6` → test `6bba`,
128 held-out videos, run on a GCP L4):

| fold | test embryo | null | + chain | Δ | p |
|---|---|---|---|---|---|
| 0 | `44b6` (71) | 0.8981 | 0.9181 | +0.0199 | <0.0001 |
| **1** | **`6bba` (128)** | **0.9092** | **0.9195** | **+0.0103** | **<0.0001** |

Significant on **both** embryos. The smaller fold-1 Δ is expected — `6bba` starts
higher (less fragmentation to recover) — but the direction and significance hold on
an embryo never seen during tuning. **This rules out winner's-curse overfitting to
fold 0.** Caveat: N=4 and gate=8 were still *chosen* on fold 0; a formal
pick-on-1-test-on-0 re-derivation would be strictly stronger, but the effect is
robust enough (significant on both) to trust.

### A1.2 — ILP birth/death asymmetry — DEAD END on top of the chain

The public "strongest knob" is `appearance 0.0 / disappearance 1.4` (make track
termination 14× costlier than initiation, suppressing fragmentation). Re-derived
on held-out fold 0 (71 videos, L4), *with the Phase 1 chain applied*, against the
default-weight chain (0.9181):

| appearance / disappearance | score | Δ vs default | p |
|---|---|---|---|
| **0.1 / 0.1** (default) | **0.9181** | — | — |
| 0.0 / 1.4 (public) | 0.9146 | −0.0035 | 0.45 |
| 0.0 / 1.0 | 0.9148 | −0.0033 | 0.48 |

Both asymmetric variants are **worse** (not significantly) and drive
`division_jaccard` to **0.0000** — the heavy disappearance cost suppresses every
division. The reason: our `min_track_len` + `motion_relink` chain already removes
the short fragments the birth/death asymmetry targets, *while preserving divisions*
(keep-division-components). Doing both is redundant, and the asymmetry additionally
destroys the small division credit. **Solver-weight tuning for Phase 1 is now
exhausted** — the remaining headroom is division *evidence* (Phase 2), not linking
costs.

## Phase 2 — Divisions

### A2.0 diagnostic (blocking) — missed divisions are a LINKING problem, not segmentation

Ultrack's central claim is that missed/false divisions are segmentation merges
upstream. We tested it on held-out fold 0 (`44b6`, 71 videos, 26 GT divisions, 145
predicted forks) with `scripts/diagnose_divisions.py`:

**Q1 — for each GT division, are the daughter cells detected?** (predicted node
within 7 µm of each GT daughter at its frame)

| | count | |
|---|---|---|
| **both daughters detected** | **25 / 26 (96.2%)** | → linking miss: cells found, fork not linked |
| one daughter | 1 (3.8%) | partial |
| neither | 0 (0.0%) | detection/segmentation miss |

**Q2 — are false forks merges or links?** 99.3% have ≤1 GT nucleus at the parent
(link signature); 0% have ≥2 (merge signature).

**Verdict: the recall ceiling is LINKING, not detection.** In 96% of missed
divisions we already detect both daughters — the linker simply fails to connect
them into a fork. Ultrack's segmentation-merge thesis does not hold here (consistent
with detection recall 0.9987). **Phase 2 targets the linker** — division-aware edge
evidence (Linajea-style backward-offset regression: daughters regress a vector to
the parent; mutual agreement = division), not segmentation.

Caveat: Q2's merge signature is weakened by sparse GT (~2.8 annotated nuclei/frame,
so two GT nodes rarely fall within 7 µm of a fork parent by chance). Q1 is not
affected — it checks known GT daughter positions against our own dense detections —
and Q1 alone settles the direction.

### A2.1/A2.2 — geometric division recovery — DEAD END (points to learned evidence)

A2.0 established the daughters are detected but not linked into forks. The cheapest
fix is to add a fork edge to an orphan "sister" by geometry alone (symmetric split:
parent-daughter, sister-sister, and split-symmetry gates). Tested on fold 0, in the
chain (relink 8 -> recover-div -> min-track-len 4):

| recovery gates (div/sister/sym um) | score | division_jaccard |
|---|---|---|
| off (chain) | **0.9181** | **0.0385** |
| 6 / 8 / 3 | 0.9046 | 0.0168 |
| 4 / 5 / 1.5 | 0.9141 | 0.0145 |

**Both worse -- and division_jaccard goes DOWN, not up.** The proximity+symmetry
signature matches coincidental orphan pairs far more often than real sisters, so it
adds false forks (precision falls) and corrupts the 5-generation windows of the few
real divisions we had. Only 15 of ~51 detected daughters are even free orphans; 36
are already mid-track, unreachable without risky re-parenting.

**Geometry alone is insufficient for division precision.** This is the empirical
case for LEARNED division evidence -- Linajea-style backward-offset regression
(train a head so each daughter predicts a vector to its parent; mutual agreement =
division). That is a training investment, not a post-processing pass, and it is the
next real Phase 2 step -- it needs the GPU.

### A2.3 — learned candidate edge-prob is NOT division-discriminative

Before training a division head, we tested whether the transformer's *existing*
candidate scores already separate divisions. Patched prediction to dump the pre-ILP
candidate graph (up to 2 scored children per parent) and compared, on fold 0
(44b6), the weaker of the two daughter-edge probs for real divisions vs the top-2
min-prob of every false candidate fork:

| | n | min-edge-prob median |
|---|---|---|
| real divisions | 12 | 0.764 |
| false candidate forks | **58,181** | 0.623 |

The signal exists but is swamped by base rate. Precision at every threshold:

| thr | recall(real) | #false ≥ thr | precision |
|---|---|---|---|
| 0.5 | 1.00 | 58181 | 0.000 |
| 0.7 | 0.58 | 16129 | 0.000 |
| 0.9 | 0.33 | 478 | 0.008 |

**No usable operating point.** ~0.02% of candidate forks are real divisions; a
single learned-linking threshold cannot overcome that. The edge scorer was trained
for one-parent-per-child linking and carries no division-specific signal. This is
the quantitative statement of why divisions are the unsolved block: rare event
(base rate ~1/4800 candidate forks) + sparse GT (~125 training divisions) + a model
with no division head. A trained offset-regression head faces the same base-rate
wall; whether a multi-feature classifier can beat it is the open question.

### A2.4 — multi-feature division classifier — FAILS the kill criterion

The sanctioned single shot: pre-filter candidate forks to plausible ones, extract
9 features (learned probs min/max, commitment gap, parent-daughter distances, split
symmetry, sister distance, local density, daughter displacement divergence), train a
gradient-boosted classifier on 6bba, test on 44b6. Pre-registered kill criterion:
precision >= 0.05 at recall >= 0.30.

| recall | TP | FP | precision |
|---|---|---|---|
| 0.67 | 6 | 4339 | 0.001 |
| 0.33 | 3 | 1261 | 0.002 |
| 0.11 | 1 | 107 | 0.009 |

Train: 45 positives in 63,892 plausible forks. Test: 9 in 57,847. **Best precision
at recall >= 0.30 = 0.004 (needed 0.05) -- FAIL by >12x.** Combining the learned
score with geometry/density/divergence does not beat the base rate.

**Conclusion: divisions are not reachable with this data + model.** The chain of
evidence is complete: daughters are detected (A2.0), but geometry (A2.2), the
learned edge-prob (A2.3), and a multi-feature classifier (A2.4) all fail on the same
wall -- ~1 real division per ~1,300 plausible candidate forks, and only ~125 GT
divisions to learn from. The 0.1 division block stays at its incidental ~0.0385
(the ILP's few forks). Rational strategy: defend the 0.9181 edge-term lead
(Phase 4), not chase an unreachable +0.10. Kill criterion honoured.

## Track B (research) — unbalanced OT for division detection

### B3.1 — OT signal does NOT survive detector output — KILL criterion met

The GT-node probe separated dividers from non-dividers by unbalanced-OT row-sum at
Cohen's d ~ 2.9 (a dividing mother transports mass to two daughters; the tau term
lets its row exceed 1). B3.1 re-runs the probe over real DETECTIONS, labelling a
source detection a divider iff it matches a GT divider (<=7 um). Fold 0, sweep over
epsilon in {0.03,0.1,0.3} x tau in {1,2,5}:

| | divider row-sum | non-divider row-sum | Cohen's d |
|---|---|---|---|
| GT nodes (prior) | 1.333 | 1.029 | 2.94 |
| **detector output** | 1.029 | 1.023 | **0.21 (best)** |

**Pre-registered kill: d<1.0 -> stop. Best d = 0.21 -> STOP.** On sparse GT a
divider is geometrically isolated with its two daughters; on dense detector output
every cell has many neighbours, so the transport mass is uniform and the divider is
not special. The mass-splitting signal is real *in principle* (GT) but washed out by
real detection density -- the same base-rate/dense-field wall that defeated the
A2.4 classifier, reached independently. The GT-only d~2.9 stands as a mechanistic
characterisation, not a usable detector. B3.4 (learn the cost via optimistix
implicit-diff) is gated on B3.1 and is therefore not pursued.

## THE METRIC REFRAME — sparsification is the real lever (+0.07 ceiling)

Reading the scorer exactly (`metrics.py`):
```
adj_edge_jaccard = max(0, edge_jaccard * (1 - 0.1 * (N_pred - N_true)/N_true))   # NO upper clamp
score            = adj_edge_jaccard + 0.1 * division_jaccard
```
`N_true` is the **dense** `estimated_number_of_nodes` (~47,714/video); the annotated
GT is **sparse** (~700 nodes/video, ~1.3%). Two consequences we had been ignoring:

1. **Under-predicting nodes multiplies the score, up to x1.1** (as N_pred -> 0). Our
   dense chain predicts 25k-72k nodes/video -> multiplier ~1 or a *penalty*.
2. **The metric rewards a SPARSE graph.** Unmatched predicted edges are largely not
   penalised as FP, but every predicted node inflates N_pred and shrinks the bonus.

**Oracle test (fold 0):** keep only the 1.3% of predicted nodes that match a GT node
(<=7 um), drop the rest:

| | nodes | score | adj_edge_jaccard |
|---|---|---|---|
| dense chain | 2,127,482 | 0.9181 | 0.9143 |
| **oracle-sparse (GT-matched only)** | **27,035 (1.3%)** | **0.9882** | 0.9843 |

**+0.07 from node count alone**, landing above the current public leaderboard leader
(0.976). The oracle uses GT to select, so it is a ceiling, not a method -- but it
proves the leaders are almost certainly **sparsifying**, not out-tracking us. Our
entire dense detect-everything pipeline is metric-misaligned.

**The new problem:** select the ~380 GT-relevant nodes/video WITHOUT GT. This is
the winnable frontier the division work is not -- ceiling 0.988, and even partial
selection closes most of the 0.879->0.976 public gap. (Also reframes the private-LB
picture: sparsification transfers if it keys on a real annotation bias; it does not
if GT is a random subset. That is the next thing to measure.)

### Selector signal — GT cells are in LONGER tracks (the one honest handle)

Comparing GT-matched vs unmatched detections on fold 0 (per-node, ~8.7k matched vs
760k unmatched):

| feature | GT-matched | unmatched |
|---|---|---|
| depth z | 48.8 | 48.8 (no signal) |
| local density (15 um) | 8.0 | 8.0 (no signal) |
| **track length** | **median 34** | **median 16** |

Track length is the only GT-free signal, and it is real (2x). But a *hard* length
cutoff already saturated at +0.0126 (the min-track-len sweep) because GT also
contains short tracks. The open test: a LEARNED node-selector (length + mean edge
prob + track smoothness + ...) trained to predict GT-membership. Unlike the division
classifier (45 positives, failed), node selection has **~8,700 positives** -- a
trainable base rate. If it approaches the oracle (0.988) it is the competition's
real lever; if it plateaus near length-alone, honest sparsification is ~exhausted
and the public 0.97 cluster is likely metric-exploited (collapses at the patched
private rerun) -- in which case our honest 0.92 is well-positioned privately.
