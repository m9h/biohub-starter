# Biohub Cell Tracking — Essential Reading & Actionable Priors

## THE FIVE MUST-READS

1. **HOCT — Higher-Order Cell Tracking Transformer** — Bragantini, Theodoro, Royer (2026).
   arXiv:2607.11754 · https://arxiv.org/abs/2607.11754
   By the competition organizer, posted 13 Jul 2026 (mid-competition). **"Higher-order" = edge-centric,
   NOT hypergraph** — candidate links attend to each other under a 3D geometric prior.
   Names our exact two failure modes: "divisions entangle lineage paths in node embedding space" and
   "edges sharing a node have near-random label agreement." 59% error reduction with 400 annotations.

2. **Ultrack** — Bragantini et al., Nature Methods 22:2423 (2025). doi:10.1038/s41592-025-02778-0
   https://github.com/royerlab/ultrack
   Reference method on this exact data. Key claim: do NOT commit to a segmentation before linking —
   keep a hierarchy of candidates and select by temporal consistency via ILP.
   **Check first: are our missed divisions actually segmentation merges upstream?**

3. **Linajea** — Malin-Mayor et al., Nature Biotechnology 41:44 (2023). doi:10.1038/s41587-022-01427-7
   https://github.com/funkelab/linajea
   Closest published analogue: whole-embryo light-sheet lineaging FROM SPARSE ANNOTATIONS.
   **Best division mechanism in the literature**: daughters regress a displacement vector back at the
   parent; mutual agreement between two children = division signal. No separate mitosis classifier.
   75.8% vs 31.8% complete 1-hour lineages. Released zebrafish data on Janelia figshare.

4. **Trackastra** — Gallusser & Weigert, ECCV 2024. arXiv:2405.15700
   https://github.com/weigertlab/trackastra
   Best score-per-hour: pretrained, division-aware, trainable on the leaderboard's own objective.
   **Key transferable trick: parental softmax normalization** — makes division-consistent
   assignments learnable rather than incidental. Kaggle artifact already packaged by user `jirkaborovec`.

5. **Kane & Kimmel 1993** — The zebrafish midblastula transition. Development 119:447.
   doi:10.1242/dev.119.2.447
   The item no ML competitor will have read. See caveat below.

## SECOND TIER

- **Maška et al. 2023**, Nat Methods 20:1010 — Cell Tracking Challenge, 10 years. Defines LNK
  (linking-only) and BC(i) (branching correctness w/ frame tolerance) — the published analogues of
  our two metric terms.
- **Ulman et al. 2017**, Nat Methods 14:1141 — division detection is the WEAKEST sub-task field-wide.
  Use to calibrate how much of the 0.1 is realistically recoverable.
- **btrack** — Ulicna et al. 2021, Front Comp Sci 3:734559. Builds tracklets containing NO splits,
  then a separate stage assigns {init, terminate, link, BRANCH, apoptose}. Architecture mirrors our
  two-term metric: optimize edges first, tune branch threshold independently.
- **TGMM** — Amat et al. 2014, Nat Methods 11:951. Division as Gaussian mixture-component split.
  Free signal: a nucleus elongates (anisotropic covariance) BEFORE splitting; each daughter's
  covariance determinant ~ half the parent's after.
- **ELEPHANT** — Sugawara et al. 2022, eLife 11:e69380. 23,829 nuclei from ~2% manual annotation.
  Incremental/active learning — our regime exactly (151 divisions total).
- **Kozawa et al. 2016**, Sci Rep 6:32962. Nuclear SHAPE predicts imminent division in live zebrafish,
  up to ~35 min ahead. Sequential inference over the shape TRAJECTORY beat single-timepoint ML
  (p=1.4e-21). Anticipatory signal — fires before the split is visible.
- **Betjes et al. 2025**, Nat Methods 22:2400. Calibrated per-edge error probability — principled
  thresholding instead of tuned constants.
- **Huh et al. 2011**, IEEE TMI 30:586. EDCRF: division as a labeled state transition along a
  candidate sequence ("before mitosis" -> "after mitosis"). Fits our +/-1 frame tolerance.
- **Schiegg et al. 2013**, ICCV. Conservation Tracking — models undersegmentation explicitly,
  the dominant error mode in dense nuclei.

## DETECTION (deprioritize — metric only needs centroids within 7um)

- StarDist-3D — Weigert et al. WACV 2020, arXiv:1908.03636. Native anisotropy support.
- nnU-Net — Isensee et al. 2021. Steal its anisotropy heuristics for 1.625/0.40625um.
- Cellpose-SAM — Pachitariu et al. 2025, bioRxiv 2025.04.28.651001. Robust to anisotropic blur.

## MEASURED FACTS FROM OUR DATA (2026-07-19)

- 2 embryos only: 6bba (128 samples), 44b6 (71). Test is embryo-DISJOINT.
- 151 GT divisions total across 199 videos. 112 videos have ZERO. Mean 0.76/video.
- 133,318 annotated nodes; ~2.8 per timepoint of ~30-130 real cells.
- Held-out 44b6, 50ep weights: score 0.8596, div_jaccard 0.0127 (TP=2 FP=153 FN=2), recall 0.9987.
- We over-predict divisions ~10x (7.75/video predicted vs 0.76 GT).
- Division FP suppression alone is worth ~ +0.05 score.

## BIOLOGICAL PRIORS — TESTED

**Embryo-wide division synchrony: NEGATIVE (tested).**
KS vs uniform D=0.131 p=0.010 — weak. Divisions occupy 74/100 timepoints in 6bba (mean t=46,
sd=25.8). Kane & Kimmel synchrony applies to cycles 1-9 PRE-MBT; Zebrahub imaging is ~10hpf
onward, post-MBT, where synchrony is lost by design. Caveat: test pooled across videos assuming
comparable `t`, which is unverified.

**Still untested and more promising:**
- **Sister-phase inheritance** — Kane & Kimmel show cycle timing is CELL-AUTONOMOUS and inherited
  from lineal ancestors, so sisters stay in phase with each other longer than with neighbours.
  This is a per-lineage prior that may survive post-MBT where embryo-wide synchrony does not.
- **Refractory period** — no cell divides twice within a stage-appropriate cycle length. Pure
  precision gain, zero cost.
- **Nuclear volume drop** — daughters ~half parent volume. A "division" where both children match
  the parent's volume is an undersegmentation split.
- **Daughter displacement symmetry** — daughters appear symmetrically about the parent centroid.
  Subtract local tissue flow first (Wan et al. 2019, Cell 179:355) or bulk migration masks it.
- **Rate calibration from GT** — 0.76 divisions/video, derivable per embryo with CIs. The public
  notebooks use LB-probed constants (SAFE_DIV_GLOBAL_FRAC_CAP=0.00375) for the same purpose.

## RANKED DIVISION FEATURES (from literature)

1. Backward-pointing offset regression (Linajea)
2. Nuclear volume halving, radially normalised
3. Shape-trajectory anticipation (Kozawa)
4. Displacement symmetry after flow subtraction
5. Refractory period
6. Deferred k-frame commitment (never finalise a division at the frame it's proposed)

## NOT VERIFIED / DEAD LEADS

- No 2026 dataset paper exists (searched arXiv + bioRxiv). metrics.md + baseline repo are authoritative.
- No hypergraph cell-tracking paper by Bragantini. "Higher-order" = edge-centric.
- "adjusted edge Jaccard" does not exist in the literature — competition-specific.
- Goldsborough's published work (InstanSeg) is 2D histology segmentation, not tracking.
- MOT transformers (TrackFormer/MOTR): poor fit — assume distinguishable appearance and no object
  splitting. Both fail for identical dividing nuclei.

---

# TOOL/METHOD SURVEY (added 2026-07-19)

## What actually wins on 3D+t embryos (Cell Tracking Challenge)

Two families of CLASSICAL combinatorial optimization, not end-to-end deep tracking.
Both treat mitosis as a first-class variable INSIDE the optimization:

1. **Iterative DP / Viterbi with explicit motion model** — KTH-SE.
   Magnusson et al. 2015, IEEE TMI 34(4):911-929, doi:10.1109/TMI.2014.2370951.
   Adds one track at a time; each iteration builds a trellis of every way to insert a
   track, solved by Viterbi shortest-path, with SWAP operations letting a new track edit
   earlier ones. Mitosis is a first-class arc with probability p_S.

2. **ILP over multi-hypothesis segmentation** — ultrack, linajea.
   Segmentation and linking solved JOINTLY (overlap constraints force a disjoint
   selection from a hierarchy of nested candidates).

CTC 3D embryo standings (approximate; decoded from leaderboard image, verify):
  Fluo-N3DH-CE  : THU-CN 0.850 | CZB-US/ultrack 0.844 | KTH-SE 0.829
  Fluo-N3DL-DRO : CZB-US 0.708 | KTH-SE 0.617 | JAN-US/linajea 0.591
  Fluo-N3DL-TRIF: CZB-US 0.841 | MPI-GE 0.804

## The ILP substrate: motile

`motile` (github.com/funkelab/motile, v1.0.1 2026-07-16) is the shared ILP layer.
Trackastra's ILP mode IS motile. linajea is its ancestor. Backend ilpy -> Gurobi/SCIP.
No detection - takes a candidate TrackGraph with learnable costs.
API churn is real (v0.4 -> v1.0 in four months) - PIN THE VERSION.

### linajea division constraints (the formulation to copy)

    sum_prev x_e + appear - x_n = 0      # exactly one parent
    sum_next x_e - 2*x_n <= 0            # THE ONE-TO-TWO ALLOWANCE
    sum_next x_e - split <= 1            # split reification
    sum_next x_e - 2*split >= 0
    split + child + continuation - x_n = 0

Split/child carry LEARNED costs -> division is scored, not merely permitted.
Contrast our current pipeline: fixed division_weight=1.0, then rate-cap the output.

## TGMM's mitosis detector — the only trained one in the field

Amat et al. 2014, Nat Methods 11:951, doi:10.1038/nmeth.3036. Software is DEAD (2018,
CUDA-era Windows C++) but the design transplants:
  - gentleBoost on 3D ELLIPTICAL HAAR FEATURES in the nucleus's own ellipsoidal frame
  - KL-divergence splitScore between parent and candidate daughter Gaussians
  - Mahalanobis arbitration: parent centroid vs daughter axes
  - TGMM 2.0 adds max-flow/min-cut partition of supervoxels into two non-touching sets
Free signal: a nucleus elongates (anisotropic covariance) BEFORE splitting; each
daughter's covariance determinant ~ half the parent's after.

## Maintenance status (2026-07)

ALIVE     : ultrack (v0.7.2), motile (v1.0.1), Trackastra (v0.5.4), Mastodon,
            TrackMate, ELEPHANT (a Mastodon client extension, not an alternative)
DORMANT   : btrack (~6mo gap), linajea (2023 - lab moved to motile)
DEAD      : TGMM (2018), EmbedTrack (2022, and 2D ONLY - disqualifying for this task)
NOT A TRACKER: CellTracksColab (downstream track analysis only)

## Division handling by tool (how each represents one-to-two)

  ultrack    : explicit `division[i]` binary; nodes + division == sum(edges_out) + disappear
  linajea    : node_split / node_child / node_continuation, learned costs
  motile     : reified NodeSplit under MaxParents(1) + MaxChildren(2); also models MERGES
  Trackastra : greedy mode = one degree check (out_degree >= 2), no division cost;
               ILP mode delegates to motile
  btrack     : explicit Fates.DIVIDE hypothesis, prior scaled by lambda_branch;
               States enum (INTERPHASE/PROMETAPHASE/METAPHASE) feeds a mitotic prior
  TrackMate  : divisions only representable in LAP step 2 (segment start -> mid-segment);
               cost includes intensity-ratio term (daughter ~ half parent intensity)
  ELEPHANT   : NO explicit division model; relies on flow net trained on validated links
  Mastodon   : GOTCHA - the Simple LAP Linker has split detection DISABLED.
               Use the standard Sparse LAP Linker for lineage tracing.
