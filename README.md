# Learning Cell Tracking in Development — a Starter Kit

A hands-on way into **computational developmental biology**, using a real, open
problem as the vehicle: the Kaggle competition
[**Biohub - Cell Tracking During Development**](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
(CZ Biohub SF / Royer Group).

The problem, in one sentence: given a 3D movie of a living zebrafish embryo, find
every cell nucleus, follow each one through time, and catch the moments a cell
divides into two. Do that well and you have reconstructed a **lineage tree** —
the record of which cell came from which, the thing developmental biologists have
wanted to read directly since the 19th century. Light-sheet microscopy now
produces these movies faster than anyone can annotate them by hand, so the
bottleneck is computational, and it is genuinely unsolved.

If you are a student, this repo is meant to get you *doing* the science in an
afternoon instead of spending three weeks on setup and dead ends. It is **not** a
solution and it will not hand you a leaderboard rank. It gives you a
correctly-measured baseline to start experimenting from, the reading that grounds
the field, and — just as important — a record of the paths that look promising
but lead nowhere, so you can spend your time where the real questions are.

### What you'll actually learn by doing this

- **Working with 3D+time microscopy data** — OME-Zarr volumes, anisotropic
  voxels, the GEFF graph format tracking data is stored in
- **The detect → link → divide pipeline** that underlies every modern cell
  tracker: a 3D U-Net for detection, learned edge scoring for linking, and
  integer-linear-programming for globally consistent tracks
- **Honest evaluation** — why the obvious validation split silently lies here,
  and how to build one that doesn't. This is the transferable skill; it outlasts
  the competition.
- **Where machine learning meets biology** — divisions are the open problem, and
  the best ideas come from knowing how cells actually divide, not from tuning

---

## Three things the data will teach you first

Every strong decision in this problem flows from these. We learned them the slow
way so you don't have to:

1. **You don't need to train a model to start.** Every top public notebook loads
   the same prepackaged weights untouched — everything above ~0.89 is
   *post-processing geometry*, not deep learning. That's liberating: you can make
   real progress with graph algorithms and biology, no GPU training required.

2. **There are only two embryos** (`6bba`, `44b6`), and the hidden test embryo is
   one neither of them. The baseline's default validation splits *by sample*, so
   both embryos leak onto both sides and every score it reports is inflated. Of
   six top public notebooks surveyed, **none built a split that measures what the
   competition actually tests.** Fixing this is [the repo's main
   contribution](#build-honest-validation-this-repos-main-contribution), and it's
   a lesson that applies far beyond cell tracking.

3. **Divisions are the frontier.** The baseline finds 2 real divisions against 153
   false ones (`division_jaccard = 0.0127` out of a possible 1.0). That block is
   worth up to **+0.10** of the score, and the field currently papers over it with
   hand-tuned constants rather than modelling the biology. This is the open
   question a newcomer can actually contribute to.

Full numbers behind each claim: [`FINDINGS.md`](FINDINGS.md).
Papers and background: [`notes/READING_LIST.md`](notes/READING_LIST.md).
Our running leaderboard record: [`CV_LB_LOG.md`](CV_LB_LOG.md).

---

## Quickstart

```bash
# 1. Baseline repo + environment  (aarch64 and x86 both fine; uv resolves CUDA)
git clone https://github.com/royerlab/kaggle-cell-tracking-competition.git
cd kaggle-cell-tracking-competition && uv sync --extra dev
./.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2. Data (~81 GB zip -> ~87 GB extracted; needs ~170 GB free)
#    Requires accepting the competition rules in-browser first.
kaggle competitions download -c biohub-cell-tracking-during-development -p DATA
cd DATA && unzip -q biohub-cell-tracking-during-development.zip

# 3. The weights everyone actually uses (dataset says "50ep"; it is really 402ep)
kaggle datasets download pilkwang/biohub-tracking-support-pack-50ep-v1 --unzip -p support_pack

# 4. Point the repo at the data
export CELLMOT_DATA_DIR=$PWD/DATA/train        # NOTE: the train/ dir itself, not its parent

# 5. Sanity check — should be 122 passed
./.venv/bin/python -m pytest -q -m "not slow"
```

### Build honest validation (this repo's main contribution)

```bash
python scripts/make_embryo_splits.py --data-dir $CELLMOT_DATA_DIR
# -> embryo_splits.json : fold 0 holds out 44b6, fold 1 holds out 6bba
```

### Get a real score

```bash
cd kaggle-cell-tracking-competition
./.venv/bin/python scripts/predict_unet_transformer.py \
    --splits $CELLMOT_DATA_DIR/embryo_splits.json --split 0 \
    --evaluate --det-threshold 0.96875 \
    --weights ../support_pack/weights/unet_transformer/split_0/edge_predictor_best.pth
```

Expected, on held-out `44b6`:

```
score=0.8596  edge_jaccard=0.8567  adj_edge_jaccard=0.8583
division_jaccard=0.0127 (TP=2 FP=153 FN=2)  node_recall=0.9987
```

**Now add `--use-ilp` and re-run.** That one flag is worth **+0.042** (0.8596 ->
0.9012, paired bootstrap CI [+0.0317, +0.0527], p<0.0001) — it replaces greedy
linking with a global flow-consistent solve:

```
score=0.9012  adj_edge_jaccard=0.9012  division_jaccard=0.0000
```

Note the catch: at the default `--ilp-division-weight 1.0` the solver emits
**zero divisions**. It removes all 153 false forks (edge FP 398 -> 224) and the 2
true ones with them. The whole 0.1 division term is behind that single scalar.
Costs 3.8x runtime (81.6 vs 21.6 s/video; ~4.5 h projected for the real ~199-sample
hidden test, of a 12 h cap).

### Understand the data

```bash
python scripts/analyze_dataset.py --data-dir $CELLMOT_DATA_DIR
```

Prints embryo composition, annotation sparsity, division statistics, and the
synchrony test. Reproduces every number in `FINDINGS.md`.

---

## Gotchas that cost us time

- **`CELLMOT_DATA_DIR` points at `train/` itself**, not its parent. The error
  message is ambiguous.
- **The predictions directory accumulates across runs.** Different threshold
  settings write into the same folder, silently producing a Frankenstein
  submission. `rm -rf` it between runs.
- **`node_recall` is a trap.** It moves *opposite* to score when you tune the
  detection threshold. Always evaluate against the real metric.
- **The `test/` folder is copies of training data.** The real hidden test
  (~199 samples, embryo-disjoint) is swapped in at notebook rerun. Budget
  inference for ~199 samples against the 12 h cap, not for 4.
- **This is a code competition**: notebooks only, ≤12 h, **internet disabled**.
  Dependencies and weights must be pre-uploaded as Kaggle Datasets.

---

## Where the open questions are

If you want to make a real dent — or just learn the most — these are the live
problems, ranked by measured headroom. Each is a place a newcomer can contribute:

1. **Division false-positive suppression** — ~+0.05 from precision alone. Start
   with Linajea-style backward-offset regression (daughters regress a vector at
   the parent; mutual agreement = division), plus free geometric features
   (volume halving, displacement symmetry after subtracting local tissue flow),
   plus the GT-derived rate of 0.76 divisions/video.
2. **Fragmentation suppression** — a division TP requires an intact 5-generation
   lineage window, so ILP birth/death asymmetry feeds *both* metric terms.
3. **Honest recalibration** — re-derive the public constants against an
   embryo-held-out split. They were leaderboard-probed and will not transfer.

Do **not** spend effort on: better detection (recall already 0.9987), architecture
changes, or faster graph code (Hungarian costs <15 s across the whole test set).
All measured — see `FINDINGS.md`.

**Longer training is NOT on that list.** An earlier version of this file said it
was; that was based on mis-reading the support pack as 50 epochs when it is
**402**. More training measurably helps (+0.020 from 350ep to 402ep, significant),
and whether it plateaus past 402 is unknown.

**Before any of it**, check Ultrack's central claim: are the missed divisions
actually *segmentation merges upstream* rather than linking failures? If so, all
three items above are aimed at the wrong stage.

---

## How this was made (and how to read it)

Assembled from measurement, not opinion. Every number is reproducible with the
scripts here — run them, don't trust us. **Negative results are stated as
prominently as positive ones**; in research the dead end you don't have to walk
is often the most valuable thing someone can hand you, and this repo tries to
hand you several.

Where a claim is uncertain it is flagged inline — a habit worth copying. The
largest known caveat: the 402-epoch weights were probably trained on both
embryos, so even our held-out numbers are mildly optimistic. A fully clean
measurement needs retraining on one embryo (~3 days on a single GPU) — itself a
good first project if you want one.

First measured leaderboard result: held-out-embryo **CV 0.9012 → public LB
0.867** — the two track, so the validation is trustworthy. The running record,
plus the offline-packaging recipe for submitting to a no-internet code
competition, lives in [`CV_LB_LOG.md`](CV_LB_LOG.md).
