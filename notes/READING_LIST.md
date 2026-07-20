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
- Held-out 44b6, 402ep support-pack weights: score 0.8596, div_jaccard 0.0127 (TP=2 FP=153 FN=2), recall 0.9987.
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

---

# MATHEMATICAL TECHNIQUES (added 2026-07-19)

## The organizing fact

Division is a ONE-TO-TWO mapping, and nearly every imported technique assumes a bijection.
  - Hungarian: feasible set contains no split
  - Optical flow: single-valued correspondence
  - Kalman: fixed state dimension
  - Balanced OT: fixed marginals, one unit of mother mass cannot become two
  - Contrastive re-ID: identity is an equivalence relation, so mother~d1 and mother~d2
    forces d1~d2 -- exactly the pair you must separate

Techniques survive in proportion to how naturally they express 1->2: either by explicitly
constraining out-degree <= 2 (ILP) or by being many-to-one by construction
(offset-to-center regression, parental softmax).

## LOAD-BEARING (2020-2026)

  ILP / min-cost-flow with explicit division variables  <- the dominant paradigm
  ultrack; linajea
  Trackastra's PARENTAL SOFTMAX  (<=1 parent, many children; one-to-many permitted,
    many-to-one structurally forbidden; division edges upweighted lambda=10)
  Offset-to-center regression (many-to-one by construction -- both daughters regress
    to the same mother center naturally)
  StarDist-3D (but its failure mode is PRECISELY mitotic: star-convexity + NMS merge
    condensed apposed daughters -- which is what ultrack's multi-hypothesis absorbs)
  Track-oriented MHT logic, in ILP disguise (Reid 1979 -> "generate candidate tracks,
    solve set-packing" is structurally identical to modern ILP formulations)
  ByteTrack-style confidence stratification
  CTC metric suite + CHOTA + traccuracy

## DISPLACED / WRONG

  Kalman filtering            -- cell appearance non-discriminative; embryo motion not
                                 smooth-linear; fixed state dim cannot express division
  DeepSORT-style re-ID        -- premise INVERTS: cells near-identical by construction,
                                 least distinctive exactly at mitotic rounding
  JPDA                        -- marginalises away the association variable, but that
                                 variable IS the scientific deliverable
  GNN message passing         -- MEASURABLY WRONG. HOCT reports adjusted homophily on the
                                 line graph of CTC candidate graphs at H_adj = 0.01 +/- 0.04,
                                 indistinguishable from random. Aggregation averages in noise.
                                 Ben-Haim's cell-tracker-GNN authors concede division edges
                                 rarely fire; lineage recovered by post-hoc heuristic.
  Optical flow                -- brightness constancy + single-valued => mitosis appears
                                 only as a divergence spike
  Spectral methods            -- cannot express in/out-degree or acyclicity constraints

## ASCENDANT

  HOCT (arXiv 2607.11754, 2026-07-13, Bragantini/Theodoro/Royer)
    Candidate LINK is the token (not the node). 3D RoPE w/ learnable per-head frequencies,
    line-to-line distance attention bias with attractive AND repulsive heads, multi-frame
    parental softmax, two-pass ILP.
    Ranks 1st overall on CLB/LNK/BIO across all 16 CTC datasets.
    Six days old, unrefereed, and reports NO zebrafish results despite the lab having data.

## ABSENT FROM THE FIELD == OPEN

  - Hypergraph cell tracking. Division IS intrinsically a 3-ary (parent,d1,d2) relation and
    nobody has written a tracker that says so directly. BUT HOCT reaches the same place via
    edge-attention -- read it before claiming novelty.
  - Unbalanced OT on 3D+t embryos. arXiv search "Sinkhorn" AND "cell tracking" -> ZERO hits.
    MECHANISM: let the mother's row marginal exceed 1 with finite tau, so the plan splits one
    row across two columns, and the tau*KL marginal-violation penalty becomes the MITOSIS PRIOR.
    (Waddington-OT's growth-rate trick applied per-object instead of per-population.)
    Nearest precedents: SCOTT (SIAM J Math Data Sci 2020, doi:10.1137/19M1253976) -- weighted
    Gromov-Wasserstein with explicit division/merge detection, but 2D ONLY.
    ARCOS.px (J Cell Sci 2025, doi:10.1242/jcs.264022) -- unbalanced Sinkhorn, but for
    signalling events, not lineages.
    CAVEAT: cite Waddington-OT as precedent that UOT encodes proliferation, NEVER as evidence
    OT works for tracking. scRNA-seq OT matches POPULATIONS across destructive snapshots;
    tracking assigns INDIVIDUAL identity across paired frames.
  - Lineage-aware / hyperbolic metrics
  - Any published quantification of sister-cell ID-swaps as a distinct error class

## PHD filters / random finite sets -- mine, don't adopt

Mahler's PHD prediction equation contains a SPAWNING term (intensity of new targets born
conditional on an existing target's state), put there for missiles separating from boosters.
That is a mathematically exact model of mitosis, not an analogy.
Nguyen/Vo/Vo/Kim/Choi, IEEE TSP 2021, doi:10.1109/TSP.2021.3111705 -- labeled RFS producing
actual lineage trees with a morphology-tuned spawning model.
But plain PHD discards identity (fatal) and none has placed competitively on CTC.

## Metric learning -- the transitivity break

Sister cells immediately post-division are SIMULTANEOUSLY the hardest possible negatives
(same genome, cell-cycle phase, fluorophore load, adjacent position, near-mirror morphology)
and the most semantically positive pair in the dataset. BATCH-HARD MINING PREFERENTIALLY
SELECTS EXACTLY THEM, concentrating maximum gradient where labels are ill-posed.
Published responses:
  - exclude divisions, recover lineage post hoc (Ben-Haim)
  - separate mitosis head, mother/daughter never posed as positive OR negative
    (Zyss et al., BMC Bioinformatics 27(1):30, 2025, doi:10.1186/s12859-025-06344-5)
  - relax identity to LINEAGE -- CELLECT (Zhou et al., Nat Methods 22:2411-2422, 2025,
    doi:10.1038/s41592-025-02886-x), 64-D per-voxel embeddings, "one-versus-two" triplet
    loss so daughters are same-lineage. >7,000 cells. Most on-target contrastive work.
  - make division the SIGNAL via time-arrow prediction (Gallusser et al., MICCAI 2023,
    arXiv:2305.05511 -- mitosis is the dominant source of temporal irreversibility)

## Metrics

AOGM/TRA (Matula et al., PLOS ONE 2015, doi:10.1371/journal.pone.0144959) is a weighted graph
edit distance; default w_FN=10, w_NS=5, w_EA=1.5 -- one missed detection costs ~7 added edges.
BC(i) = branching correctness at frame tolerance i, the division-specific measure.
REPORT BC(0..2): the spread separates "misses divisions" from "finds them but mistimes them".
`traccuracy` implements CTC metrics + generalized AOGM + DivisionMetrics + CHOTA with an
error-annotated graph export.

Why MOT metrics fail for embryos:
  (a) every MOT metric presupposes GT is a partition into identity classes bijectively
      matchable to predictions; division forces a convention and neither is a superset
  (b) error severity is EXPONENTIALLY NON-UNIFORM IN TIME -- a mis-assigned division at the
      4-cell stage corrupts thousands of descendants, but AOGM charges one edit either way
  (c) post-division daughter swaps are often biologically unresolvable yet still charged

## DATA CAVEATS

  - Zebrahub's published tracks were PRODUCED BY ULTRACK -- algorithm-generated, not gold standard.
  - There is NO zebrafish dataset in the Cell Tracking Challenge (embryo sets are C. elegans,
    Drosophila, Tribolium). CTC rankings transfer to this task only BY ANALOGY.
  - ELEPHANT evidence that division-aware flow is cheap to train: 1,162 links from 10
    timepoints, including only 18 links covering 9 divisions, sufficed.

## TALKS / VIDEO (verified URLs)

  Bragantini, "Introduction to ultrack" (I2K 2023, 1h12m)  <- BEST SINGLE RESOURCE
    https://www.youtube.com/watch?v=uBXXr43lovQ
  Bragantini, "ultrack: large-scale versatile cell tracking" (SciPy 2024, 32m)
    https://www.youtube.com/watch?v=98dahngkNOI
  Bragantini, ultrack (CBIAS 2023, 13m)  https://www.youtube.com/watch?v=SilkHzHgSk8
  Royer, "Sci Viz: Napari" (1h16m)  https://www.youtube.com/watch?v=51PV-3tf9A8
  Royer lab, "DaXi light-sheet" (1h)  https://www.youtube.com/watch?v=o1IY73Jacwg
    ^ the microscope behind Zebrahub -- explains anisotropy, view fusion, drift artifacts
  Royer lab, "Aydin: image denoising" (1h03m)  https://www.youtube.com/watch?v=LFe-Q02B1KA
  Royer, "Multi-Dimensional Microscopy Datasets" (iBiology, 20m)
    https://www.youtube.com/watch?v=tSMRoW0-NFY
  Royer, "Beyond the Cell Atlas Conference" (24m)  https://www.youtube.com/watch?v=fY4e7IvQCUI
  Keller, "Imaging and Reconstructing Mouse Development at the Single-Cell Level" (iBiology)
    https://www.youtube.com/watch?v=R2I4c6C9Ths
  Weigert, "Nuclei segmentation with StarDist" (NEUBIAS Academy@Home)
    https://www.youtube.com/watch?v=Amn_eHRGX5M
  Tinevez, "Tracking cells and organelles with TrackMate" (NEUBIAS)  https://youtu.be/ITwamUmna-Q
  Kreshuk, "ilastik beyond pixel classification" (NEUBIAS)  https://youtu.be/_ValtSLeAr0

  Royer Group talks page:  https://biohub.org/royer/talks
  NEUBIAS Academy@Home archive (highest-value index):
    https://eubias.org/NEUBIAS/training-schools/neubias-academy-home/neubias-academy-archive-spring2020/
  Robert Haase, full BioImage Analysis lecture series (TU Dresden):
    https://www.youtube.com/playlist?list=PL5ESQNfM5lc7SAMstEu082ivW4BDMvd0U
  I2K 2024 playlist: https://www.youtube.com/playlist?list=PLdA9Vgd1gxTbvxmtk9CASftUOl_XItjDN
  linajea zebrafish lineage datasets:
    https://janelia.figshare.com/articles/dataset/Zebrafish_data_for_whole-embryo_lineage_reconstruction_with_linajea/24968724

NOT FOUND (searched, absent -- do not go looking again):
  Teun Huijben, Thibaut Goldsborough, Carsten Marr, Fabrice Cordelieres, Uwe Schmidt (solo).
  No Trackastra talk recording. Jan Funke has a channel but no linajea/motile talk found.
  No Royer-lab talks on NeurIPS/CVPR/ISBI/ELMI/EMBL/CSHL channels.
