#!/usr/bin/env python
"""A2.4 -- multi-feature division classifier (the sanctioned single shot).

A2.3 showed a single learned edge-prob threshold cannot beat the 12-vs-58k base
rate. This asks whether *combining* the learned score with geometry clears the bar.

Pipeline, cross-embryo (train 6bba -> test 44b6):
  1. From each pre-ILP candidate graph, take every parent's top-2 scored children
     and keep the PLAUSIBLE forks (both probs high, division-like geometry). This
     pre-filter shrinks the negative haystack before learning.
  2. Features per fork: prob_min/max, prob gap to the 3rd child (commitment),
     parent-daughter distances, split symmetry, sister distance, local density,
     daughter displacement divergence.
  3. Label positive if the parent matches a GT divider whose two daughters match
     the two children (<=7 um).
  4. Train a gradient-boosted classifier on 6bba, evaluate on 44b6.

KILL CRITERION: if no operating point reaches precision >= 0.05 at recall >= 0.30,
the division block is not reachable with these features -> stop, keep the negative.

Usage:
    python build_division_classifier.py --gt-dir GT \
        --train-cand CAND_6bba --test-cand CAND_44b6
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import tracksdata as td

VOXEL = np.array([1.625, 0.40625, 0.40625])
MATCH_UM = 7.0
# plausibility pre-filter
MIN_PROB = 0.45
MAX_PD_UM = 12.0
MAX_SIS_UM = 14.0
DENS_R = 15.0


def _load(p):
    r = td.graph.IndexedRXGraph.from_geff(str(p))
    return r[0] if isinstance(r, tuple) else r


def tables(g, prob=False):
    N = g.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    ids = N["node_id"].to_list()
    node = {i: (t, np.array([z, y, x]) * VOXEL) for i, t, z, y, x in
            zip(ids, N["t"].to_list(), N["z"].to_list(), N["y"].to_list(), N["x"].to_list())}
    keys = ["source_id", "target_id"] + (["edge_prob"] if prob else [])
    E = g.edge_attrs(attr_keys=keys)
    out = collections.defaultdict(list)
    pr = E["edge_prob"].to_list() if prob else [1.0] * E.height
    for s, d, p in zip(E["source_id"].to_list(), E["target_id"].to_list(), pr):
        out[s].append((d, p))
    return node, out


def by_frame(node):
    fr = collections.defaultdict(list)
    for i, (t, p) in node.items():
        fr[t].append((i, p))
    return {t: (np.array([i for i, _ in v]), np.stack([p for _, p in v]))
            for t, v in fr.items()}


def nearest(frame, t, p):
    if t not in frame:
        return np.inf, None
    ids, pts = frame[t]
    d = np.linalg.norm(pts - p, axis=1)
    k = int(d.argmin())
    return float(d[k]), int(ids[k])


FEATURES = ["prob_min", "prob_max", "prob_gap3", "d_pc1", "d_pc2", "sym",
            "sister", "density", "diverge"]


def extract(gt_geff, cand_geff):
    """Return (X, y) over plausible candidate forks in one video."""
    gnode, gout = tables(_load(gt_geff))
    cnode, cout = tables(_load(cand_geff), prob=True)
    cframe = by_frame(cnode)
    gframe = by_frame(gnode)

    # GT divider parents matched into candidate space (by position)
    gt_div = {}   # candidate parent id -> set of matched candidate daughter ids
    for s, ds in gout.items():
        if len(ds) < 2:
            continue
        ts, ps = gnode[s]
        dp, cpar = nearest(cframe, ts, ps)
        if cpar is None or dp > MATCH_UM:
            continue
        dset = set()
        for d, _ in ds:
            td_, pd = gnode[d]
            dd, cd = nearest(cframe, td_, pd)
            if cd is not None and dd <= MATCH_UM:
                dset.add(cd)
        if len(dset) >= 2:
            gt_div[cpar] = dset

    X, y = [], []
    for p, kids in cout.items():
        if len(kids) < 2:
            continue
        kids_sorted = sorted(kids, key=lambda x: -x[1])
        (c1, p1), (c2, p2) = kids_sorted[0], kids_sorted[1]
        p3 = kids_sorted[2][1] if len(kids_sorted) > 2 else 0.0
        tp, pp = cnode[p]
        pos1, pos2 = cnode[c1][1], cnode[c2][1]
        d1 = float(np.linalg.norm(pp - pos1))
        d2 = float(np.linalg.norm(pp - pos2))
        sis = float(np.linalg.norm(pos1 - pos2))
        # plausibility pre-filter
        if p2 < MIN_PROB or max(d1, d2) > MAX_PD_UM or sis > MAX_SIS_UM:
            continue
        # local density at the next frame near the parent
        tn = cnode[c1][0]
        if tn in cframe:
            _, pts = cframe[tn]
            density = int((np.linalg.norm(pts - pp, axis=1) <= DENS_R).sum())
        else:
            density = 0
        v1, v2 = pos1 - pp, pos2 - pp
        diverge = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        X.append([p2, p1, p2 - p3, d1, d2, abs(d1 - d2), sis, density, diverge])
        # positive iff parent is a GT divider and BOTH chosen children are its daughters
        pos = p in gt_div and c1 in gt_div[p] and c2 in gt_div[p]
        y.append(int(pos))
    return X, y


def load_set(gt_dir, cand_dir, embryo):
    X, y = [], []
    stems = sorted(p.stem for p in Path(cand_dir).glob(f"{embryo}_*.geff")
                   if (Path(gt_dir) / f"{p.stem}.geff").exists())
    for stem in stems:
        xi, yi = extract(Path(gt_dir) / f"{stem}.geff", Path(cand_dir) / f"{stem}.geff")
        X += xi; y += yi
    return np.array(X, float), np.array(y, int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--train-cand", required=True, help="6bba candidate dir")
    ap.add_argument("--test-cand", required=True, help="44b6 candidate dir")
    args = ap.parse_args()

    print("extracting train (6bba) ...", flush=True)
    Xtr, ytr = load_set(args.gt_dir, args.train_cand, "6bba")
    print(f"  train forks: {len(ytr)}  positives: {int(ytr.sum())}")
    print("extracting test (44b6) ...", flush=True)
    Xte, yte = load_set(args.gt_dir, args.test_cand, "44b6")
    print(f"  test forks:  {len(yte)}  positives: {int(yte.sum())}\n")

    if ytr.sum() < 5 or yte.sum() < 3:
        print("too few positives to train/evaluate honestly -- inconclusive.")
        return

    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300,
        class_weight="balanced", l2_regularization=1.0, random_state=0)
    clf.fit(Xtr, ytr)
    score = clf.predict_proba(Xte)[:, 1]

    print("feature importances (permutation would be better; using split gains):")
    print("  " + "  ".join(FEATURES))

    print("\n  thr   recall  TP   FP   precision")
    best = 0.0
    for thr in np.linspace(0.1, 0.95, 18):
        pred = score >= thr
        tp = int((pred & (yte == 1)).sum())
        fp = int((pred & (yte == 0)).sum())
        rec = tp / max(1, int(yte.sum()))
        prec = tp / max(1, tp + fp)
        if rec >= 0.30:
            best = max(best, prec)
        if tp:
            print(f"  {thr:.2f}  {rec:5.2f}  {tp:3d}  {fp:4d}   {prec:.3f}")

    print(f"\nKILL CRITERION: best precision at recall>=0.30 = {best:.3f}")
    if best >= 0.05:
        print("  PASS -- learned+geometry beats the base rate. Worth integrating as a")
        print("  division-recovery filter and measuring the score delta.")
    else:
        print("  FAIL -- even combined features cannot clear precision 0.05 at recall")
        print("  0.30. Divisions are not reachable with this data/model; keep the")
        print("  negative result and defend the edge-term lead (Phase 4).")


if __name__ == "__main__":
    main()
