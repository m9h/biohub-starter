---
marp: true
theme: default
paginate: true
header: "Learning Cell Tracking in Development — Biohub Kaggle Competition"
footer: "biohub-starter Baseline Evaluation | Morgan Hough et al."
style: |
  section {
    font-family: 'Inter', system-ui, sans-serif;
    background-color: #0f172a;
    color: #f8fafc;
  }
  h1, h2, h3 {
    color: #38bdf8;
  }
  code {
    background-color: #1e293b;
    color: #f1f5f9;
  }
  table {
    font-size: 0.85em;
  }
---

# Learning Cell Tracking in Development
### An Empirical Evaluation & Starter Kit for 3D+t Lineage Reconstruction

**Kaggle Competition Evaluation & Methodology**  
*CZ Biohub SF / Royer Lab Benchmark*  

Repository: `github.com/m9h/biohub-starter`  

---

## 1. The Core Scientific & Computational Problem

* **The Goal:** Given a 3D+time light-sheet movie of a living zebrafish embryo, detect every cell nucleus, track each nucleus through time, and capture every cell division event.
* **The Deliverable:** Reconstruct a complete **lineage tree** graph (`.geff` format).
* **The Bottleneck:** Light-sheet microscopes generate terabytes of 3D data faster than humans can annotate. Manual tracking is impossible.

```
       3D+t Volume               Detect Detections            ILP / Linking               Lineage Graph (.geff)
  [100, 64, 256, 256]   --->   (z, y, x) Centroids   --->   Flow Conservation   --->   Tree with Division Nodes
```

---

## 2. Why Most Validation Pipelines Fail (The Leakage Trap)

* **Dataset Fact:** 199 training videos come from only **2 embryos** (`6bba`: 128, `44b6`: 71).
* **Test Condition:** The hidden competition test set is **embryo-disjoint**.
* **The Flaw in Public Notebooks:** Standard random 90/10 splits leak frames from the *same embryo* on both sides of the split.
* **The Solution:** Embryo-disjoint cross-validation (`embryo_splits.json`).

| Fold | Held-Out Embryo | Train Set | Test Set | Purpose |
|:---:|:---:|:---:|:---:|:---|
| **0** | `44b6` | 128 (`6bba`) | 71 (`44b6`) | Hyperparameter & Geometry Tuning |
| **1** | `6bba` | 71 (`44b6`) | 128 (`6bba`) | Generalization Confirmation |

---

## 3. Ground Truth & Metric Reality

* **Detection is Solved:** 3D U-Net node recall is **0.9987**. Detection tuning is a dead end.
* **Metric Structure:**
  $$\text{Score} = \text{adj\_edge\_jaccard} + 0.1 \times \text{division\_jaccard}$$
* **The `node_recall` Trap:** Unmatched predicted nodes do not count as false positives. Raising recall can *lower* score due to the node count penalty multiplier!
* **Scoring is Micro-Averaged:** Evaluation requires paired bootstrap resampling over videos ($B=10,000$) rather than simple $t$-tests.

---

## 4. Key Benchmark Results & Improvements

All metrics evaluated on held-out embryo `44b6` (Fold 0):

| Pipeline Stage | Config / Parameters | CV Score | Public LB | Delta | Significance |
|:---|:---|:---:|:---:|:---:|:---:|
| **Greedy Baseline** | 402ep weights, det threshold 0.96875 | 0.8596 | — | — | — |
| **+ Global ILP** | `--use-ilp` (default params) | 0.9012 | 0.867 | **+0.0416** | $p < 0.0001$ |
| **+ Length Filter** | `min_track_len = 4` | 0.9107 | — | **+0.0095** | $p < 0.0001$ |
| **+ Motion Relink** | `relink_max_um = 8.0` $\mu$m | **0.9181** | **0.879** | **+0.0074** | $p < 0.0001$ |

**CV and LB Co-move:** Local CV improvements (+0.020) directly transfer to Public LB (+0.012).

**Generalizes across embryos:** the chain (tuned on Fold 0) also helps on **Fold 1**
(`6bba`, 128 videos): 0.9092 → 0.9195, **+0.0103**, $p < 0.0001$. Significant on
*both* embryos — not fold-0 overfitting.

---

## 5. Correction: The "50ep" Support Pack Myth

> **Common Misconception:** Public notebooks load `biohub-tracking-support-pack-50ep-v1` assuming it was trained for 50 epochs.

* **Audit Fact:** Inspecting `checkpoint_last.pth` reveals `epoch = 402`.
* **Empirical Checkpoint Benchmark:**

| Checkpoint | Epochs | Held-Out Score | Edge Jaccard | Div Jaccard |
|:---|:---:|:---:|:---:|:---:|
| 300ep pin | 300 | 0.8392 | 0.8409 | 0.0102 |
| 350ep pin | 350 | 0.8395 | 0.8412 | 0.0119 |
| **Support Pack** | **402** | **0.8596** | **0.8567** | **0.0127** |

* **Takeaway:** Training performance improves **monotonically** with more epochs.

---

## 6. The Division Precision Bottleneck (The +0.10 Opportunity)

* **Extremely Rare Event:** Only 151 GT divisions in 199 videos (0.76/video; 112 videos have zero).
* **Extreme Over-prediction:** Baseline emits ~7.75 divisions/video $\to$ **Precision $\approx 1.3\%$**.
* **5-Generation Window Rule:** A division TP requires intact track lineage 2 frames before & after:
  $$\text{parent} \to \text{divider} \to \text{daughters} \to \text{granddaughters}$$
* **Current Result:** `--use-ilp` predicts **0 divisions** by default, gaining score by eliminating 153 false positive forks!

---

## 7. Negative Results (Save Your Time!)

Do **NOT** spend effort on these paths — empirical testing proves they do not help:

1. ❌ **Anisotropic U-Net Pooling:** Redundant because `--downsample 1,4,4` already yields isotropic $64^3$ voxels.
2. ❌ **Faster Hungarian/Graph Code:** Assignment costs $<15$s across the whole test set. Not a bottleneck.
3. ❌ **Embryo-wide Division Synchrony (Kane & Kimmel Prior):** KS test vs uniform $p=0.010$ (weak). Imaging is post-MBT; synchrony is lost.
4. ❌ **Tuning ILP Division Weight:** Lowering division cost re-introduces garbage greedy forks (154 FPs for 2 TPs), dropping overall score.

---

## 8. Actionable Biological & Feature Priors

Where the real **+0.05 to +0.10** headroom lies:

1. **Linajea-Style Offset Regression:** Daughters regress displacement vectors back to parent centroid; agreement = division signal.
2. **Nuclear Volume Halving:** Post-mitosis daughter volume $\approx 0.5 \times$ parent volume.
3. **Displacement Symmetry:** Subtract local tissue flow before measuring daughter separation symmetry.
4. **Refractory Period Constraint:** Enforce minimum cycle duration between consecutive division events along a lineage.

---

## 9. Kaggle Code Competition Gotchas & Rerun Specs

* **Constraint:** 12-hour limit, T4x2 GPU, **No Internet**.
* **Runtime Projection:**
  * Greedy linking: 21.6 s/video $\to \sim 1.2$ h for 199 samples.
  * ILP solving: 81.6 s/video $\to \sim 4.5$ h for 199 samples.
* **Packaging Facts:**
  1. Pin GPU to `NvidiaTeslaT4` (prevents P100 sm_60 crashes).
  2. Install wheels with `--no-deps` to preserve Kaggle's NumPy 2.0 / SciPy 1.16 stack.
  3. Export `PYTHONPATH=repo/src`.

---

## 10. Summary & Recommended Action Plan

```
  [1] Setup Embryo CV  --->  [2] Enable Global ILP  --->  [3] Post-Process Geometry  --->  [4] Model Divisions
  (make_embryo_splits)        (--use-ilp +0.042)           (min_len=4, relink=8.0µm)     (Offset Reg / Volume)
```

1. **Adopt Embryo-Disjoint Validation:** Stop tuning on leaked random splits.
2. **Derive Local Constants:** Use `min_track_len = 4` and `relink_max_um = 8.0` $\mu$m.
3. **Focus Entirely on Division Evidence:** Implement feature-based false positive filters before turning division solver weights.

*Full details in `FINDINGS.md`, `notes/PAPER.md`, and `CV_LB_LOG.md`.*
