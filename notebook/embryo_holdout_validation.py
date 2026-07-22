# %% [markdown]
# # Embryo-Held-Out Validation for This Competition
#
# *Morgan Hough — DevoWorm Group · Biopunk Lab.
# Full write-up + code: [github.com/m9h/biohub-starter](https://github.com/m9h/biohub-starter)*
#
# **TL;DR** — The 199 training samples come from only **two embryos**, and the hidden
# test set is *embryo-disjoint*. The baseline's default split is random **by sample**,
# so both embryos land on both sides. Every score tuned that way is measured under
# conditions that cannot occur at test time.
#
# I surveyed six of the top public notebooks. **None of them do any cross-validation** —
# no `GroupKFold`, no held-out split, no OOF. Constants like `0.96875` (= 31/32),
# `4.66` and `0.00375` are leaderboard-probe artifacts.
#
# Building an honest embryo-held-out split and re-deriving two standard post-processing
# steps on it moves the score **0.8981 → 0.9181** (+0.0199, *p*<1e-4, 71 held-out
# videos), the same chain **confirms on the second embryo** (fold 1, 128 videos:
# 0.9092 → 0.9195, *p*<1e-4), and the gain **transfers to the leaderboard**
# (CV +0.020 → LB +0.012). In every case the locally optimal constant differs from
# the public one. This notebook shows how, with:
#
# 1. A drop-in **embryo-disjoint split generator**
# 2. The **Phase-1 post-processing chain**, each step gated on a paired bootstrap,
#    and its **cross-embryo confirmation**
# 3. A measured **checkpoint comparison** (402ep vs 350ep vs 300ep — the "50ep" myth)
# 4. **Division statistics** from ground truth — the open +0.10 problem
# 5. Two traps: **`node_recall` moves opposite to score**, and runtime at test scale
#
# Everything is reproducible. Negative results included — several are the useful part.
#
# *Base pipeline credit: the ILP + motion-relink recipe from the public
# `biohub-exp110-ilp-birth-death-cost` / `clean-approach` family. This notebook adds
# validation methodology, not a new tracker.*

# %%
import collections
import json
from pathlib import Path

import numpy as np

# Works on Kaggle or locally
CANDIDATES = [
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
    Path("/kaggle/input/biohub-cell-tracking-during-development/train"),
    Path("data/train"),
]
DATA_DIR = next((p for p in CANDIDATES if p.exists()), None)
print("data dir:", DATA_DIR)

# %% [markdown]
# ## 1. There are only two embryos
#
# Folder names are `{embryo_id}_{field_of_view}`. The competition data description
# states: *"Train and test sets are embryo-disjoint — no embryo appears in both."*

# %%
stems = sorted(
    p.name[:-5] for p in DATA_DIR.glob("*.zarr")
    if (DATA_DIR / f"{p.name[:-5]}.geff").exists()
)
embryos = collections.Counter(s.split("_")[0] for s in stems)
print(f"{len(stems)} samples, {len(embryos)} embryos")
for e, n in embryos.most_common():
    print(f"  {e}: {n} samples")

# %% [markdown]
# So the *effective* sample size for generalization is **2**, not 199. The 199 samples
# are fields of view from two embryos, and are heavily correlated within each.
#
# Generalizing to an unseen embryo is the actual task. A random 90/10 split cannot
# measure it.

# %% [markdown]
# ## 2. Embryo-disjoint splits (drop-in)
#
# The baseline scripts accept `--splits <file>`, so this is a one-line change to your
# pipeline.

# %%
def build_embryo_splits(stems):
    """One fold per embryo: hold that embryo out entirely."""
    by_embryo = {}
    for s in stems:
        by_embryo.setdefault(s.split("_")[0], []).append(s)
    folds = []
    for i, held in enumerate(sorted(by_embryo)):
        train = [s for e, ss in by_embryo.items() if e != held for s in ss]
        folds.append({"split": i, "held_out_embryo": held,
                      "train": train, "test": by_embryo[held]})
    return folds


folds = build_embryo_splits(stems)
for f in folds:
    assert not set(f["train"]) & set(f["test"]), "leakage!"
    print(f"fold {f['split']}: hold out {f['held_out_embryo']} -> "
          f"{len(f['train'])} train / {len(f['test'])} test")

Path("embryo_splits.json").write_text(json.dumps(folds, indent=1))
print("\nUse with:  predict_unet_transformer.py --splits embryo_splits.json --split 0")

# %% [markdown]
# ## 3. The result: a post-processing chain that generalizes across embryos
#
# With the honest split in hand, I re-derived two standard post-processing steps
# **on the held-out embryo** (never on the public LB), each gated on a paired
# bootstrap over the size-weighted aggregate (10,000 resamples). Baseline = 402ep
# weights + tracksdata ILP, `--det-threshold 0.96875`.
#
# **Step A — minimum track-length filter.** Drop weakly-connected lineage
# components spanning `< N` frames, *exempting any component containing a division*
# (a division TP needs an intact 5-generation window). Sweep on fold 0 (held-out
# `44b6`, 71 videos):
#
# | N | score | Δ vs baseline |
# |---|---|---|
# | off | 0.8981 | — |
# | 3 | 0.9067 | +0.0086 |
# | **4** | **0.9107** | **+0.0126** |
# | 6 *(public constant)* | 0.9037 | +0.0056 |
# | 8 | 0.8908 | −0.0073 |
#
# The local optimum is **N=4**, *not* the public N=6 (which leaves +0.0070 on the
# table). +0.0126, 95% CI [+0.0091, +0.0162], **p<1e-4**.
#
# **Step B — motion relink.** Reconnect prematurely-terminated tracks to next-frame
# orphans by a constant-velocity, gated Hungarian assignment. By construction it can
# create neither a division nor a merge — it repairs single tracks only. Stacked on
# min-track-len 4, the optimal gate is **8 µm** (between the public tight/relaxed
# 6/10): +0.0074, 95% CI [+0.0037, +0.0114], **p<1e-4**.
#
# **Full chain: 0.8981 → 0.9181** (+0.0199, p<1e-4).
#
# ### It is not fold-0 overfitting
#
# Both constants were chosen on fold 0. Applied **unchanged** to fold 1 (train
# `44b6` → test `6bba`, 128 held-out videos) the chain still helps significantly:
#
# | fold | test embryo | baseline | + chain | Δ | p |
# |---|---|---|---|---|---|
# | 0 | `44b6` (71) | 0.8981 | 0.9181 | +0.0199 | <1e-4 |
# | **1** | **`6bba` (128)** | **0.9092** | **0.9195** | **+0.0103** | **<1e-4** |
#
# Significant on **both** embryos. The smaller fold-1 Δ is expected (`6bba` starts
# higher, less fragmentation to recover), but direction and significance hold on an
# embryo never seen during tuning. This is the property the embryo-disjoint hidden
# test actually rewards.
#
# ### CV tracks the leaderboard
#
# Two submissions bracket the result. Baseline (no LB-tuned constants): CV 0.8981 →
# **LB 0.867**. Phase-1 chain: CV 0.9181 → **LB 0.879**. The local +0.020 carries to
# +0.012 on the board — attenuated (the public set mixes both embryos) but same
# direction. **The held-out CV predicts the leaderboard**, which is the whole point.
#
# *The filter + relink are `scripts/postprocess.py` in the repo; the paired
# bootstrap is `scripts/compare_configs.py`. Both run on the saved `.geff`
# predictions — no GPU needed.*

# %% [markdown]
# ## 4. Checkpoint comparison on a held-out embryo
#
# ### First, a correction that may be useful to everyone
#
# The public dataset `pilkwang/biohub-tracking-support-pack-**50ep**-v1` — which most
# public notebooks load — does **not** contain a 50-epoch checkpoint. Its
# `checkpoint_last.pth` carries `epoch = 402`, and its `ARTIFACT_MANIFEST.json` is
# named `biohub-tracking-support-pack-**400ep**-snapshot-v1`.
#
# ```python
# torch.load("checkpoint_last.pth")["epoch"]   # -> 402
# ```
#
# I initially took the dataset name at face value and concluded "longer training is
# worse." That was backwards. Corrected comparison, on **held-out embryo `44b6`**
# (20 videos), `--det-threshold 0.96875`:
#
# | checkpoint | actual epochs | score | edge_jaccard | division_jaccard | div TP/FP/FN | node_recall |
# |---|---|---|---|---|---|---|
# | **support pack** (`50ep-v1`) | **402** | **0.8596** | 0.8567 | 0.0127 | 2 / 153 / 2 | 0.9987 |
# | 350ep pin | 350 | 0.8395 | 0.8412 | 0.0119 | 2 / 164 / 2 | 0.9979 |
# | 300ep pin | 300 | 0.8392 | 0.8409 | 0.0102 | 2 / 193 / 2 | 0.9972 |
#
# **More training is better, monotonically** across the range we can observe. The
# +0.0204 gap from 350ep to 402ep is significant under a paired bootstrap over the
# 20 videos (95% CI [+0.0069, +0.0346], p=0.0024).
#
# Whether it has plateaued by 402 is **unknown** — there is no public checkpoint
# beyond it. If anyone has one, that comparison is worth running.
#
# *Caveat: these weights were likely trained on both embryos, so even this held-out
# run is contaminated. It is a valid **relative** comparison; the absolute number is
# optimistic.*

# %% [markdown]
# ## 5. Divisions: we all over-predict by ~10x

# %%
# `tracksdata` reads the GEFF graphs. Installs on-demand (enable internet), and the
# section degrades gracefully if the data mount or the package isn't present, so the
# notebook always runs top-to-bottom.
div_times = []
try:
    import subprocess, sys
    try:
        import tracksdata as td
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tracksdata"], check=True)
        import tracksdata as td

    assert DATA_DIR is not None, "competition data not mounted"
    divs_per_video, tot_nodes = [], 0
    for p in sorted(DATA_DIR.glob("*.geff")):
        g, _ = td.graph.IndexedRXGraph.from_geff(str(p))
        N, E = g.node_attrs(), g.edge_attrs()
        t = np.asarray(N["t"])
        tot_nodes += len(t)
        node_t = dict(zip(np.asarray(N["node_id"]).tolist(), t.tolist()))
        outdeg = collections.Counter(np.asarray(E["source_id"]).tolist())
        d = [n for n, k in outdeg.items() if k >= 2]
        divs_per_video.append(len(d))
        div_times += [node_t[n] for n in d]

    print(f"total annotated nodes : {tot_nodes}")
    print(f"total GT divisions    : {sum(divs_per_video)}")
    print(f"divisions per video   : mean {np.mean(divs_per_video):.2f}")
    print(f"videos with ZERO      : {sum(1 for x in divs_per_video if x == 0)}/{len(divs_per_video)}")
except Exception as e:
    print(f"[skipped live division scan: {e}]")
    print("Reported from ground truth: 151 divisions total, mean 0.76/video, 112/199 zero.")

# %% [markdown]
# **151 divisions in the entire training set.** 112 of 199 videos have none.
#
# The held-out run above emitted ~155 forks across 20 videos (**7.75/video**) to catch
# 2 of 4 real ones. Division precision ≈ **1.3%**.
#
# The arithmetic on that block:
#
# | scenario | division_jaccard | score contribution | Δ |
# |---|---|---|---|
# | current | 0.0127 | 0.0013 | — |
# | suppress all FPs, keep 2 TP | 0.50 | 0.050 | **+0.049** |
# | catch all 4 cleanly | 1.00 | 0.100 | **+0.099** |
#
# Public notebooks handle this with rate caps (`SAFE_DIV_GLOBAL_FRAC_CAP = 0.00375`)
# reverse-engineered from LB feedback. **The correct rate is measurable from ground
# truth** — 0.76 divisions/video, derivable per embryo with confidence intervals.
# LB-probed constants are exactly the ones that won't survive an embryo shift.

# %% [markdown]
# ### A division TP needs an intact 5-generation window
#
# From `division_metrics.py`:
#
# ```
# parent → divider → child1 → grandchild1
#                  → child2 → grandchild2
# ```
#
# `_is_strongly_connected_division` requires the prediction to reproduce that whole
# structure. **You get no credit for detecting a fork** — you get credit for keeping
# the lineage intact ~2 frames either side.
#
# That explains why `ILP_DISAPPEARANCE_WEIGHT=1.4` vs `ILP_APPEARANCE_WEIGHT=0.0` is
# the strongest knob found publicly: making termination 14× costlier than initiation
# suppresses fragmentation, which feeds **both** metric terms.

# %% [markdown]
# ## 6. A biological prior that does NOT work (tested)
#
# Kane & Kimmel (1993) show zebrafish cleavage cycles 1–9 run on a ~15-min oscillator
# with near-perfect embryo-wide synchrony. If that held here, off-phase division calls
# would be easy false positives to reject.

# %%
ts = np.array(div_times)
if ts.size:
    occupied = len(np.unique(ts))
    print(f"divisions span {occupied} distinct timepoints; mean t={ts.mean():.1f} sd={ts.std():.1f}")
    try:
        from scipy import stats
        d, pv = stats.kstest(ts / max(1, ts.max()), "uniform")
        print(f"KS vs uniform: D={d:.3f}  p={pv:.4f}")
    except ImportError:
        pass
else:
    print("Reported: KS vs uniform D=0.131, p=0.010 (weak) -- see markdown below.")

# %% [markdown]
# **Negative result.** Weak deviation from uniform, divisions spread across most
# timepoints. The reason is staging: Zebrahub images from ~10 hpf, well past the
# midblastula transition, where synchrony is lost *by design*. Right biology, wrong
# developmental window.
#
# *Caveat: this pools across videos assuming `t` is comparable between fields of view,
# which I have not verified.*
#
# The more promising untested variant is **sister-phase inheritance** — Kane & Kimmel
# show cycle timing is cell-autonomous and inherited from lineal ancestors, so sisters
# stay in phase with each other longer than with neighbours. That is a per-lineage
# prior which may survive post-MBT where the global one does not.

# %% [markdown]
# ## 7. Two traps
#
# ### `node_recall` moves opposite to score
#
# Sweeping detection threshold on a 19-video split:
#
# | threshold | node_recall | score |
# |---|---|---|
# | 0.99 | 0.9715 | 0.5022 |
# | 0.9995 | **0.8161** ↓ | **0.5265** ↑ |
#
# Recall drops 15 points while score *rises*. Because unmatched predicted nodes aren't
# false positives but still count toward `N_pred`, and the adjustment multiplier
# `1 − 0.1·(N_pred − N_true)/N_true` is **not clamped at 1**, under-predicting earns a
# bonus. (The competition page confirms scores can exceed 1.0.)
#
# The training loop selects checkpoints on `acc × recall` — which optimizes the wrong
# objective. **Always evaluate against the real metric.**
#
# ### Runtime: budget for ~199 samples, not 4
#
# `test/` contains 4 samples that are *copies from train*. The real hidden test is
# "approximately the same size as the training dataset" — so ~199.
#
# **Measured on the actual Kaggle backend (2× T4), with ILP:** ~96.6 s/sample →
# **5.34 h for 199**, ~55% under the 12 h cap. That headroom is real but finite:
# it is *with* ILP and *without* TTA, so heavy test-time augmentation on top would
# eat it. Measure before submission day rather than trusting an extrapolation.

# %% [markdown]
# ## Summary
#
# | Finding | Status |
# |---|---|
# | Only 2 embryos; test is embryo-disjoint | Random splits leak — build an embryo-held-out split |
# | **Post-processing chain, re-derived on the split** | **0.8981 → 0.9181** (+0.0199, p<1e-4) |
# | **Cross-embryo confirmation** (fold 1, `6bba`, 128 vid) | **0.9092 → 0.9195** (+0.0103, p<1e-4) — not overfitting |
# | **CV → LB** | 0.867 → 0.879; local +0.020 transfers to +0.012 |
# | Every tuned constant | local optimum ≠ public value (N=4 vs 6, 8 µm vs 6/10, …) |
# | Support pack is 402ep, not "50ep" | dataset name is wrong; more training **helps** (+0.020) |
# | Divisions | 151 in training; we over-predict ~10×; open **+0.10** block, currently ~0.001 |
# | Embryo-wide synchrony prior | **negative** — wrong developmental window |
# | `node_recall` | inversely related to score |
# | Runtime (2× T4, ILP, measured) | **5.34 h** for 199 of a 12 h cap |
#
# Happy to be corrected on any of it — particularly the synchrony test, where my
# pooling assumption is the weak link. Full write-up, the post-processing code, and
# the paired-bootstrap harness: **github.com/m9h/biohub-starter**.
