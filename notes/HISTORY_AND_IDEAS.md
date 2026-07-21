# Where This Came From: A Short History of Cell Tracking in Development

This is the background the papers assume you already have. It exists so that the
tools in this repo — GEFF files, an ILP solver, a "min track length" filter —
stop looking like arbitrary machinery and start looking like the current answer
to a 150-year-old question. Read it once; the rest of the repo will make more
sense.

> **How to read this.** Each section ends with **→ in this repo**, tying the
> history to something concrete you can run. The point is not trivia — it is to
> see *why* each technique exists, because that is what tells you when it will
> fail.

---

## 1. The question: what is a lineage, and can we read it?

In 1878 Charles Otis Whitman watched leech eggs divide and realized the pattern
of divisions was not random — you could, in principle, draw a *tree* of which
cell came from which. That tree is a **cell lineage**, and reconstructing it is
one of the oldest goals in developmental biology: if you know the lineage, you
know where every cell in the adult came from and when its fate was decided.

For a century this was done by eye. The monument is *C. elegans*: **Sulston,
Horvitz, Kimble and Thomson (1977–1983)** mapped the **complete** lineage of the
nematode — all 959 somatic cells of the adult — by staring down a microscope for
years, tracking each division by hand. It is deterministic: every wild-type worm
follows the same tree. That work won the 2002 Nobel Prize in Physiology or
Medicine, and it set the bar. The dream ever since: do this **automatically**,
for **any** organism, from **images** — including animals like zebrafish whose
lineages are *not* fixed and must be measured afresh in every embryo.

**→ in this repo:** the competition's ground truth is exactly a lineage tree,
stored as a graph (`.geff`). A division is a node with two children. When we
report `division_jaccard`, we are scoring how much of the *tree structure* we
recovered — the same object Sulston drew by hand.

---

## 2. The enabling technology: light-sheet microscopy

You cannot track what you cannot image. The bottleneck broke around **2004** with
**Selective Plane Illumination Microscopy (SPIM / light-sheet)** — Huisken,
Stelzer and colleagues — which illuminates a thin sheet of the sample and images
it from the side. It is fast and gentle enough to record a *living* embryo in 3D
for hours without cooking it with light.

**Keller et al. (2008)** used it to reconstruct the "digital embryo" of
zebrafish, imaging nuclei over the first 24 hours of development. Suddenly the
data existed: terabyte 3D+time movies of every nucleus in a developing animal.
The problem became computational overnight — and it has stayed that way, because
imaging outran analysis. The CZ Biohub's Zebrahub, the source of this
competition's data, is a direct descendant of that lineage of work.

**→ in this repo:** the `.zarr` volumes are `(t, z, y, x)` light-sheet stacks.
The **4:1 anisotropy** (z-voxels are 1.625 µm, xy are 0.40625 µm) is a physical
fact of light-sheet: axial resolution is worse than lateral. Every distance we
compute has to correct for it — see `VOXEL_SCALE` in `scripts/postprocess.py`.

---

## 3. The core algorithm: tracking-by-detection

Once you have images, essentially every modern tracker splits the job in two:

1. **Detection** — find the cells/nuclei in each frame independently.
2. **Linking** — decide which detection in frame *t* is the same cell as which
   detection in frame *t+1*.

Linking is an **assignment problem**. The foundational move, **Jaqaman et al.
(2008, `u-track`)**, framed frame-to-frame linking as *linear assignment* solved
with the **Hungarian algorithm** — cheap, optimal for one-to-one matching, and
still the workhorse. You build a cost matrix (how unlikely is it that detection
*i* became detection *j*?) and find the minimum-cost matching.

But greedy frame-by-frame assignment is short-sighted: a locally cheap link can
be globally wrong. It also has no natural way to express a **division** (one cell
becoming two is a one-*to-two* event, which pure assignment forbids).

**→ in this repo:** the baseline's detector is a 3D U-Net; its recall is already
0.9987, which is why we say *detection is solved and not worth your time*. The
`motion_relink` step you'll find in `postprocess.py` is a small Hungarian
assignment — the u-track idea, applied as a repair pass to reconnect tracks the
main solver broke.

---

## 4. Going global: tracking as combinatorial optimization (ILP)

The fix for short-sightedness is to stop deciding links one frame at a time and
instead solve the **whole movie at once**. Formulate tracking as an **Integer
Linear Program (ILP)**: binary variables for "is this candidate link real?", plus
variables for appearance, disappearance, and — crucially — **division**, all tied
together by flow-conservation constraints so the solution is globally consistent
(a cell can't vanish and reappear; a division must have one parent and two
children).

This paradigm (Kausler & Hamprecht and others, ~2012 onward) dominates 3D+t
embryo tracking because it handles divisions *natively* and enforces
consistency. Two landmarks you should know:

- **linajea** (Malin-Mayor, Funke et al., ~2022) — a flow-based ILP with
  division as an explicit variable, plus a network that predicts a *movement
  vector* per cell. Division is detected when two daughters both "point back" to
  the same parent. This is the method to imitate for the division problem.
- **ultrack** (Bragantini, Royer et al., ~2024) — from the same lab that made
  this competition. Its thesis: many "missed divisions" are actually
  **segmentation errors upstream** (two cells merged into one blob), not linking
  failures. It tracks over *multiple* candidate segmentations and lets the ILP
  pick. **This is why our plan's very first division task is a diagnostic:** are
  our false forks merges or link errors? ultrack says check before you build.

**→ in this repo:** `--use-ilp` swaps greedy linking for the tracksdata ILP
solver and is worth **+0.04** — our single biggest lever. The `division_weight`
sweep in `FINDINGS.md` is us probing this exact machinery (and finding, informative­ly,
that re-weighting the solver *cannot* fix divisions without better evidence).

---

## 5. The learning turn: letting a network do the association

The newest idea is to *learn* the linking cost instead of hand-designing it.
**Trackastra (Gallusser & Weigert, ~2024)** trains a transformer to score
associations between detections across frames — attention learns which cell went
where, from data, including divisions.

**→ in this repo:** the baseline's `SimpleNodeTransformer` is exactly this — a
learned edge scorer. The `softmax(dim=0)` you'll find in the predict script is
"parental softmax": each child distributes one unit of probability over its
possible parents. That is a *learned* one-parent-per-child prior, and it is
already pulled — don't re-derive it.

---

## 6. How we know if any of it works: metrics and standards

A field cannot progress without a shared scorecard. The **Cell Tracking Challenge**
(Maška, Ulman, Ortiz-de-Solórzano et al., 2014 onward) provided it: standard
datasets and metrics (DET for detection, TRA for tracking) that made methods
comparable. The `traccuracy` library packages these metrics; our plan adds it for
`BC(0..2)` — the "how many frames off is this division" spread.

Two data-standards matter because analysis is now multi-tool and polyglot:

- **OME-Zarr / NGFF** — the chunked, cloud-friendly standard for the *images*.
- **GEFF (Graph Exchange File Format)** — a 2026 standard from the
  live-image-tracking-tools group (the same community: Funke, Royer, Schwartz,
  Malin-Mayor et al.) for the *tracking graphs*. It exists so a graph written by
  ultrack can be read by napari can be scored by traccuracy, regardless of
  language. It is Zarr underneath — one storage substrate, two metadata
  conventions (`multiscales` for images, `geff` for graphs).

**→ in this repo:** everything you load with `td.graph.IndexedRXGraph.from_geff`
is riding this standard. See the competition's own scorer for how these metrics
are actually computed, and `scripts/eval_per_sample.py` for the traps in doing it
honestly.

---

## 7. The road not (yet) taken: optimal transport

One more thread, because this repo has a research track exploring it. **Optimal
transport (OT)** asks: what is the cheapest way to move one distribution of mass
onto another? **Schiebinger et al. (2019, Waddington-OT)** used it to infer
developmental trajectories from single-cell RNA snapshots — matching *populations*
of cells across time.

Nobody (as of this writing) has published **unbalanced OT for per-cell division
detection** in 3D+t embryos. The intuition: a dividing cell is a point of *mass
creation* — one cell's worth of "stuff" becomes two — and unbalanced OT is
precisely the framework that allows mass to not be conserved. Whether that signal
survives real (noisy) detections is an open question our probe is testing, with an
explicit kill criterion. That is what honest research looks like: a clear
hypothesis, a clear way to be wrong.

**→ in this repo:** `scripts/ot_division_probe.py` and the OT sections of
`FINDINGS.md`. It may lead nowhere — and if so, the negative result is still
worth publishing, because no one has ruled it out yet.

---

## The through-line

Every technique here is an answer to fragmentation and ambiguity — the two ways a
tracker fails. Detection finds too many or too few. Linking connects the wrong
pair. Divisions get missed or invented. A hundred and fifty years of work has
converged on: **detect well, link globally, respect the biology of division, and
measure yourself honestly.** The last one — *measure yourself honestly* — is the
part the current leaderboard forgets, and it is the part this repo is built to
teach.

---

*Dates and attributions are approximate and meant as signposts, not citations —
follow [`READING_LIST.md`](READING_LIST.md) to the primary sources, and correct
anything here that a paper contradicts. If you are new to the field and spot an
error, fixing it is a genuine contribution.*
