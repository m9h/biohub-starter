# CV / LB correlation log

The single most important table in the project: does our embryo-held-out CV
predict the leaderboard? Every submission appends one row. If CV and LB
decorrelate, the local signal is broken and all tuning halts until resolved.

**Held-out CV** = embryo-disjoint, fold 0 (train `6bba` -> test `44b6`, 20 videos),
size-weighted aggregate. **Public LB** = the current public test set, which until
the final rerun is the **4 placeholder datasets** (2 per embryo, so *not*
embryo-disjoint and *not* the hidden ~199). The two are deliberately different
measurements; we track co-movement, not equality.

| date (UTC) | kernel v | config | held-out CV | public LB | CV−LB | notes |
|---|---|---|---|---|---|---|
| 2026-07-20 | v3 | 402ep + `--use-ilp`, det 0.96875, **no LB-tuned constants** | 0.9012 | **0.867** | +0.034 | First point. Tracks — CV mildly optimistic, no decorrelation. |

## Reading of the first point

- **Correlation holds.** CV 0.034 above LB, same order — the local signal is
  trustworthy. Not the divergence that would trip the kill-trigger.
- **0.867 with no LB-probed constants** sits ~0.042 below the public clean-recipe
  cluster (~0.909). That gap is the post-processing we have not built yet
  (motion relink, gap closing, min-track-length) — headroom is where the plan
  places it, not in training or detection.
- **Runtime, measured on Kaggle (T4x2):** 96.6 s/video -> **5.34 h projected for
  199** of a 12 h cap. ~55% margin; room for light TTA later.

## Packaging facts earned here (offline no-internet rerun)

1. **Pin the GPU.** `machine_shape: "NvidiaTeslaT4"` in kernel-metadata (or
   `--accelerator NvidiaTeslaT4`). Default `enable_gpu` can assign a **P100
   (sm_60)**, which Kaggle's PyTorch (sm_70+) cannot run — dies at the first CUDA op.
2. **Install wheels with `--no-deps`** and install the wheel *files* directly.
   Without `--no-deps`, pip pulls the pack's numpy 2.4.6 / scipy 1.18.0 over
   Kaggle's consistent stack and scipy's compiled ext breaks
   (`cannot import name '_center' from 'numpy._core.umath'`).
3. **Exclude numpy / scipy / scikit-image / pandas** from the install — Kaggle
   ships a mutually-consistent, GPU-matched build (numpy 2.0.2 / scipy 1.16.3).
   Layer only the tracking libs (tracksdata, geff, ilpy, pyscipopt, ...) on top.
4. The pack's `biohub_tracking` lives under `repo/src`; scripts import it
   unqualified, so `PYTHONPATH=repo/src` is required.
5. A code-competition **notebook run does not consume a daily submission** — only
   attaching a kernel version to the competition does. Debug reruns are free.
