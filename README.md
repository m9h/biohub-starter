# Biohub Cell Tracking — Starter Kit

Onboarding for the Kaggle competition
[**Biohub - Cell Tracking During Development**](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
(CZ Biohub SF / Royer Group). 3D+time light-sheet zebrafish embryos: detect
nuclei, link them across time, recover divisions and lineages.

This repo is **not** a solution. It is the shortest path from zero to a
correctly-measured baseline, plus the findings that save you from spending weeks
where there is nothing to gain.

---

## Read this first

Three things determine almost every strategic decision:

1. **Nobody trains.** Every top public notebook loads the same prepackaged
   50-epoch weights untouched. Everything above ~0.89 is post-processing
   geometry. If you start by training a model, you are burning time.

2. **There are only two embryos** (`6bba`, `44b6`) and the hidden test set is
   embryo-disjoint. The baseline's default split is random *by sample*, so both
   embryos land on both sides — every score measured that way is inflated. Of six
   top public notebooks surveyed, **none did any cross-validation.**

3. **Divisions are the open problem.** Held-out baseline gets 2 true divisions
   against 153 false ones — `division_jaccard = 0.0127` out of a possible 1.0.
   That block is worth up to **+0.10** and the field addresses it with
   leaderboard-tuned constants rather than modelling.

Full detail with numbers: [`FINDINGS.md`](FINDINGS.md).
Papers: [`notes/READING_LIST.md`](notes/READING_LIST.md).

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

# 3. The 50-epoch weights everyone actually uses
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

## Where to spend your effort

Ranked by measured headroom:

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

## Provenance

Assembled 2026-07-19 from a single day of measurement on a DGX Spark (GB10).
Every number is reproducible with the scripts here. Negative results are stated
as prominently as positive ones — several are more valuable.

Where a claim is uncertain it is flagged inline. The largest known caveat: the
50-epoch weights were probably trained on both embryos, so even held-out numbers
are contaminated. A clean measurement needs retraining on one embryo (~3 days on
a single GB10).
