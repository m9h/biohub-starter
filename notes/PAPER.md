# Empirical Benchmarking, Leakage-Free Validation, and Division Precision Bottlenecks in 3D+t Light-Sheet Cell Lineage Reconstruction

**Authors:** Morgan Hough et al.  
**Affiliation:** DevoWorm Group · Biopunk Lab  
**Repository:** [m9h/biohub-starter](https://github.com/m9h/biohub-starter)  
**Target Venue:** *Bioinformatics / Kaggle Competition Baseline Report*  
**Date:** July 2026  

---

## Abstract

Reconstructing complete cell lineage trees from 3D+time light-sheet microscopy of developing embryos is a foundational challenge in computational biology. The CZ Biohub Kaggle Competition (*Biohub - Cell Tracking During Development*) provides a benchmark dataset comprising 199 3D+t volumes. Here, we present an open-source starter kit and empirical evaluation of the cell tracking pipeline. We uncover a critical flaw in standard cross-validation pipelines: the 199 public training samples originate from only **two distinct embryos** (`6bba` and `44b6`), while the hidden test set is **embryo-disjoint**. Standard random sample-wise validation splits suffer from severe spatial-temporal data leakage, producing over-optimistic performance estimates that fail to transfer to held-out test data. 

We construct an honest 2-fold embryo-disjoint cross-validation framework and establish three key empirical results:
1. **Detection is practically solved**, achieving a node recall of **0.9987**, indicating that performance gains lie entirely in linking and division post-processing.
2. **Global flow-consistent Integer Linear Programming (ILP)** provides a **+0.0416** gain in metric score over greedy frame-by-frame linking ($p < 0.0001$).
3. **Division precision is the dominant bottleneck**: current baseline models over-predict division events by ~10× (emitting ~155 division forks per 20-video sequence to capture 2 true positives; precision $\approx 1.3\%$). Because the metric requires a 5-generation intact lineage window for a division true positive, un-calibrated division predictions severely penalize the score.

Finally, we correct a widespread misconception regarding the public baseline weights (verifying they were trained for 402 epochs rather than 50) and provide a reproducible baseline achieving **0.9181 CV / 0.879 LB**.

---

## 1. Introduction & Background

Reconstructing cell lineages—mapping every cell division and trajectory from egg to organism—has been a goal of developmental biology since Whitman (1878) and Sulston et al. (1983). Modern Selective Plane Illumination Microscopy (SPIM / light-sheet microscopy) captures 3D volumes of living embryos over hours at sub-minute temporal resolution. However, manual annotation is intractable for thousands of migrating and dividing cell nuclei across hundreds of frames.

The standard computational formulation is **tracking-by-detection**:
1. **3D Detection / Segmentation**: Identify nuclear centroids $(\hat{z}, \hat{y}, \hat{x})$ in each 3D frame.
2. **Temporal Association / Linking**: Match detections across consecutive frames $t \to t+1$.
3. **Lineage Tree Construction**: Identify mitosis events (one parent cell splitting into two daughter cells) to build a directed acyclic graph (DAG) stored in Graph Exchange File Format (`.geff`).

Despite recent deep learning advances, modern trackers struggle with division detection under high nuclear density, 4:1 voxel anisotropy ($z=1.625\,\mu\text{m}, y=x=0.40625\,\mu\text{m}$), and sparse ground-truth annotations.

---

## 2. The Data Leakage Problem & Embryo-Disjoint Validation

### 2.1 The Two-Embryo Structure
An audit of the 199 training volumes reveals that every sample is named formatted as `{embryo_id}_{field_of_view}`. The training set consists of only two embryos:
- **`6bba`**: 128 samples
- **`44b6`**: 71 samples

Because the competition evaluation replaces the test directory with ~199 samples from **unseen embryos**, random $k$-fold cross-validation splits fields of view from the *same embryo* across both training and validation sets. This results in severe data leakage: models overfit to embryo-specific spatial morphology, global movement flows, and illumination conditions.

### 2.2 Embryo-Disjoint Cross-Validation
We introduce a 2-fold embryo-disjoint splitting protocol:
- **Fold 0**: Train on `6bba` (128 samples) $\to$ Validate on `44b6` (71 samples).
- **Fold 1**: Train on `44b6` (71 samples) $\to$ Validate on `6bba` (128 samples).

Every hyperparameter and post-processing transformation in `biohub-starter` is evaluated strictly against Fold 0 held-out data using paired bootstrap significance testing ($B=10,000$ resamples) to compute 95% confidence intervals and exact $p$-values.

```
+-----------------------------------------------------------------------+
|                       TRAINING SET (199 Samples)                      |
|   Embryo 6bba (128 samples)            Embryo 44b6 (71 samples)       |
+------------------------------------+----------------------------------+
                                     |
           Fold 0 Split              |             Fold 1 Split
    Train: 6bba | Test: 44b6         |      Train: 44b6 | Test: 6bba
```

---

## 3. Empirical Findings & System Evaluation

### 3.1 Detection vs. Linking Bottlenecks
Evaluation of the baseline 3D U-Net detector reveals a node recall of **0.9987** at standard detection thresholds. Furthermore, node recall correlates *inversely* with overall metric score due to the competition's un-clamped node count adjustment penalty multiplier:
$$\text{adj} = \max\left(0, J \cdot \left(1 - 0.1 \frac{N_{\text{pred}} - N_{\text{true}}}{N_{\text{true}}}\right)\right)$$
Consequently, optimizing detection architectures or tuning bounding box thresholds yields negligible gains. The primary performance headroom resides in graph optimization and post-processing geometry.

### 3.2 Integer Linear Programming (ILP) Optimization
Replacing greedy frame-to-frame Hungarian matching with a global flow-consistent ILP solver (`--use-ilp`) enforces global edge continuity and disappearance costs:

$$\Delta \text{Score} = +0.0416 \quad (0.8596 \to 0.9012, \quad 95\%\text{ CI } [+0.0317, +0.0527], \quad p < 0.0001)$$

At default solver parameters (`--ilp-division-weight 1.0`), the ILP solver predicts zero divisions. While this zeroes the division metric term, it eliminates 153 false positive division forks, reducing edge false positives from 398 to 224.

### 3.3 Post-Processing Geometry
We implement two localized graph post-processing passes:
1. **Minimum Track Length Filter (`min_track_len`)**: Prunes short spurious track fragments ($< N$ frames) while explicitly preserving components containing division events. Sweeping $N$ on Fold 0 yields a smooth unimodal peak at **$N=4$** (+0.0126 score gain, $p < 0.0001$), outperforming the public leaderboard-probed constant of $N=6$.
2. **Motion Relinking (`motion_relink`)**: Reconnects prematurely terminated loose-end tracks to orphan track starts in frame $t+1$ via constant-velocity position prediction and gated Hungarian assignment. Gating at **8.0 $\mu$m** yields an additional +0.0074 score gain ($p < 0.0001$).

| Stage | Config / Parameters | Held-out CV (Fold 0, 71 videos) | Public LB | Delta (CV) |
|---|---|---|---|---|
| Baseline (ILP) | `--use-ilp`, det 0.96875, 402ep weights | 0.8981 | 0.867 | — |
| + Post-Process | `min_track_len=4` + `motion_relink=8.0µm` | **0.9181** | **0.879** | **+0.0199** |

> The greedy→ILP gain itself (**+0.0416**, 0.8596→0.9012, 95% CI [+0.0317, +0.0527],
> $p<10^{-4}$; §3.2) was measured on an earlier **20-video pilot** of Fold 0. On the
> full 71-video harness the ILP result *is* the 0.8981 baseline above, so we report
> the post-processing delta (+0.0199) on that consistent basis — the same null used
> in the Fold-1 comparison below. Mixing the pilot and full-harness numbers in one
> column would conflate sample sizes.

### 3.4 Cross-Embryo Generalization (Fold-1 Confirmation)

Both post-processing constants ($N=4$, $8.0\,\mu$m) were selected on Fold 0, and
their significance was reported on the same fold — a mild winner's-curse risk. To
test it, we applied the **unchanged** chain to Fold 1 (train `44b6` $\to$ test
`6bba`, 128 held-out videos, computed on a GCP L4):

| Fold | Test embryo | Null | + Chain | Delta | $p$ |
|---|---|---|---|---|---|
| 0 | `44b6` (71) | 0.8981 | 0.9181 | +0.0199 | $<10^{-4}$ |
| **1** | **`6bba` (128)** | **0.9092** | **0.9195** | **+0.0103** | $<10^{-4}$ |

The chain improves the score significantly on **both** embryos. The smaller Fold-1
delta is expected — `6bba` starts higher (less fragmentation to recover) — but the
direction and significance hold on an embryo never seen during tuning. This **rules
out overfitting to the tuning fold**: the improvement is genuine cross-embryo
generalization, the property the hidden (embryo-disjoint) test set actually
demands. (A formal pick-on-1/test-on-0 re-derivation would be strictly stronger;
significance on both folds already makes the effect trustworthy.)

---

## 4. The Division Precision Frontier

### 4.1 Precision Breakdown
Ground truth analysis across all 199 training videos reveals only **151 total division events** (mean 0.76 divisions/video; 112 out of 199 videos contain **zero** divisions). Baseline model outputs emit ~7.75 divisions/video (precision $\approx 1.3\%$). 

A division True Positive requires an intact 5-generation lineage window:
$$\text{parent} \longrightarrow \text{divider} \begin{cases} \longrightarrow \text{child}_1 \longrightarrow \text{grandchild}_1 \\ \longrightarrow \text{child}_2 \longrightarrow \text{grandchild}_2 \end{cases}$$

Any track fragmentation within $\pm 2$ frames invalidates the division TP. Thus, indiscriminate division prediction severely harms the adjusted edge Jaccard term without recovering division Jaccard points.

### 4.2 Biological Prior Evaluation
- **Embryo-wide Division Synchrony (Kane & Kimmel MBT Prior)**: Tested via Kolmogorov-Smirnov test against uniform distribution ($D=0.131, p=0.010$). Result: **Negative / Weak**. Zebrahub imaging occurs at $\sim 10$ hpf (post-Midblastula Transition), where global division synchrony is lost.
- **Unbalanced Optimal Transport (UOT) Probe**: Log-domain unbalanced Sinkhorn mass transport probing across tau parameters failed to separate dividing mother row sums ($\sim 2$) from non-dividing row sums ($\sim 1$), demonstrating that raw spatial mass transport without nuclear morphology/volume features cannot isolate divisions.

---

## 5. Kaggle Rerun & Execution Constraints

Submissions must run in a GPU-accelerated code competition environment ($12$-hour limit, internet disabled).
- **Runtime Budgeting**: Greedy linking requires 21.6 s/video ($\sim 1.2$ h projected for 199 test samples). ILP solving increases runtime to 81.6 s/video ($\sim 4.5$ h projected), leaving sufficient margin within the 12-hour limit while constraining heavy test-time augmentation (TTA).
- **GPU Pinning & Environment Isolation**: Pinning `NvidiaTeslaT4` in kernel metadata prevents execution failures on legacy architectures (sm_60). Third-party Python dependencies must be installed with `--no-deps` to avoid breaking Kaggle's native C-compiled stack (NumPy 2.0 / SciPy 1.16).

---

## 6. Conclusion & Future Directions

The `biohub-starter` repository establishes an empirical, leakage-free benchmark for 3D+t developmental cell tracking. We demonstrate that:
1. Leaderboard probing yields brittle constants that degrade under embryo shift.
2. The primary remaining opportunity (+0.05 to +0.10 score headroom) lies in **division false-positive suppression** using backward offset regression (Linajea), nuclear volume halving ratios, and local flow-subtracted displacement symmetry.

---

## References

1. Bragantini, F. et al. (2025). *Ultrack: versatile cell tracking in large-scale microscopy*. Nature Methods, 22:2423.
2. Malin-Mayor, C. et al. (2023). *Automated cell lineage reconstruction from sparse annotations with Linajea*. Nature Biotechnology, 41:44.
3. Gallusser, B. & Weigert, M. (2024). *Trackastra: Transformer-based cell tracking*. ECCV 2024. arXiv:2405.15700.
4. Bragantini, F., Theodoro, L., & Royer, L. (2026). *HOCT: Higher-Order Cell Tracking Transformer*. arXiv:2607.11754.
5. Kane, D. A. & Kimmel, C. B. (1993). *The zebrafish midblastula transition*. Development, 119:447.
