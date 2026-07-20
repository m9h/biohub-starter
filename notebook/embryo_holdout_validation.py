# %% [markdown]
# # Your CV Doesn't Exist: Embryo-Held-Out Validation for This Competition
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
# With a public LB on 29% of the data and a private LB on the other 71% of an
# embryo-disjoint test set, that seems worth fixing. This notebook provides:
#
# 1. A drop-in **embryo-disjoint split generator**
# 2. A measured **checkpoint comparison** on a held-out embryo (402ep vs 350ep vs 300ep)
# 3. **Division statistics** from ground truth — and why we all over-predict ~10x
# 4. A **runtime projection** for the real hidden test size
# 5. A trap: **`node_recall` moves opposite to score**
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
# ## 3. Checkpoint comparison on a held-out embryo
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
# ## 4. Divisions: we all over-predict by ~10x

# %%
import tracksdata as td  # noqa: E402

divs_per_video, div_times, tot_nodes = [], [], 0
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
# ## 5. A biological prior that does NOT work (tested)
#
# Kane & Kimmel (1993) show zebrafish cleavage cycles 1–9 run on a ~15-min oscillator
# with near-perfect embryo-wide synchrony. If that held here, off-phase division calls
# would be easy false positives to reject.

# %%
ts = np.array(div_times)
occupied = len(np.unique(ts))
print(f"divisions span {occupied} distinct timepoints; mean t={ts.mean():.1f} sd={ts.std():.1f}")
try:
    from scipy import stats
    d, pv = stats.kstest(ts / max(1, ts.max()), "uniform")
    print(f"KS vs uniform: D={d:.3f}  p={pv:.4f}")
except ImportError:
    pass

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
# ## 6. Two traps
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
# Measured on one GPU, no TTA, no ILP: **~21.6 s/sample** → ~72 min for 199.
# With the 8× D4 TTA the public recipe uses, that is roughly **8–10 h against the
# 12 h cap**. Tight. Worth measuring before submission day.

# %% [markdown]
# ## Summary
#
# | Finding | Status |
# |---|---|
# | Only 2 embryos; test is embryo-disjoint | Random splits leak |
# | Support pack is 402ep, not "50ep" | dataset name is wrong; more training **helps** (+0.020) |
# | Divisions | 151 in training; we over-predict ~10× |
# | Division headroom | up to **+0.10**, currently ~0.001 |
# | Embryo-wide synchrony prior | **negative** — wrong developmental window |
# | `node_recall` | inversely related to score |
# | 8× TTA on 199 samples | ~8–10 h of a 12 h budget |
#
# Happy to be corrected on any of it — particularly the synchrony test, where my
# pooling assumption is the weak link. If anyone has run a genuine embryo-held-out
# comparison, I would like to see how it lines up.
