#!/usr/bin/env python
"""A2.0 -- the blocking Phase 2 diagnostic.

Ultrack's central claim is that missed/false divisions are *segmentation merges
upstream*, not linking failures. Before building any division model we test it,
because the answer decides where Phase 2 aims: at detection or at the linker.

Two questions, answered per fold from GT + prediction .geff:

  Q1 (recall -- the FN divisions). For each GT division (parent with >=2 children),
      are the daughter cells DETECTED in the prediction (a predicted node within
      MATCH_UM of each GT daughter at the daughter's frame)?
        both daughters detected  -> we could have linked it: LINKING miss
        a daughter undetected    -> DETECTION/segmentation miss
      This is the decisive number: it says whether the recall ceiling is the
      linker or the detector.

  Q2 (precision -- the FP forks). For each predicted fork (parent with >=2
      children) that is NOT a true division, is the parent node co-located with
      TWO GT nuclei (MERGE signature: the detector fused two cells, the solver
      split them) or ONE GT nucleus (LINK signature: a spurious extra child)?

Distances use the physical voxel scale (z=1.625, y=x=0.40625 um) and a 7 um gate,
matching the competition metric's node matching.

Usage:
    python diagnose_divisions.py --gt-dir GT --pred-dir PRED [--embryo 44b6]
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import tracksdata as td

VOXEL = np.array([1.625, 0.40625, 0.40625])  # z, y, x um/voxel
MATCH_UM = 7.0


def _load(p: Path):
    r = td.graph.IndexedRXGraph.from_geff(str(p))
    return r[0] if isinstance(r, tuple) else r


def graph_tables(g):
    """Return node dict {id:(t,pos)} and out-adjacency {src:[dst,...]}."""
    N = g.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    ids = N["node_id"].to_list()
    t = N["t"].to_list()
    pos = np.stack([np.asarray(N["z"].to_list()), np.asarray(N["y"].to_list()),
                    np.asarray(N["x"].to_list())], axis=1) * VOXEL
    node = {i: (tt, p) for i, tt, p in zip(ids, t, pos)}
    E = g.edge_attrs(attr_keys=["source_id", "target_id"])
    adj = collections.defaultdict(list)
    for s, d in zip(E["source_id"].to_list(), E["target_id"].to_list()):
        adj[s].append(d)
    return node, adj


def by_frame(node):
    """{t: (ids array, pos array)} for fast nearest-neighbour within a frame."""
    fr = collections.defaultdict(list)
    for i, (t, p) in node.items():
        fr[t].append((i, p))
    return {t: (np.array([i for i, _ in v]),
                np.stack([p for _, p in v])) for t, v in fr.items()}


def nearest(frame_index, t, p):
    """Min distance (um) from point p to any predicted node at frame t; inf if none."""
    if t not in frame_index:
        return np.inf, None
    ids, pts = frame_index[t]
    d = np.linalg.norm(pts - p, axis=1)
    k = int(d.argmin())
    return float(d[k]), int(ids[k])


def divisions(node, adj):
    """List of (parent_id, [child_ids]) for out-degree >= 2 nodes."""
    return [(s, ds) for s, ds in adj.items() if len(ds) >= 2]


def diagnose(gt_dir: Path, pred_dir: Path, embryo: str | None):
    stems = sorted(p.stem for p in pred_dir.glob("*.geff")
                   if (gt_dir / f"{p.stem}.geff").exists()
                   and (embryo is None or p.stem.startswith(embryo)))

    q1 = collections.Counter()   # both / one / none daughters detected
    q2 = collections.Counter()   # merge / link / (fork is actually a TP)
    n_gt_div = n_pred_fork = 0

    for stem in stems:
        gnode, gadj = graph_tables(_load(gt_dir / f"{stem}.geff"))
        pnode, padj = graph_tables(_load(pred_dir / f"{stem}.geff"))
        pframe = by_frame(pnode)
        gframe = by_frame(gnode)

        # Q1: for each GT division, are both daughters detected?
        for parent, kids in divisions(gnode, gadj):
            n_gt_div += 1
            hits = 0
            for c in kids[:2]:
                tc, pc = gnode[c]
                d, _ = nearest(pframe, tc, pc)
                hits += (d <= MATCH_UM)
            q1["both" if hits == 2 else ("one" if hits == 1 else "none")] += 1

        # Q2: for each predicted fork, is it a real division, else merge or link?
        gt_div_parents = {p for p, _ in divisions(gnode, gadj)}
        for parent, kids in divisions(pnode, padj):
            n_pred_fork += 1
            tp, pp = pnode[parent]
            # is this fork near a GT division parent? -> true-ish division
            dpar, gmatch = nearest(gframe, tp, pp)
            if gmatch in gt_div_parents and dpar <= MATCH_UM:
                q2["true_division"] += 1
                continue
            # count GT nuclei within MATCH_UM of the parent at frame tp
            if tp in gframe:
                ids, pts = gframe[tp]
                n_near = int((np.linalg.norm(pts - pp, axis=1) <= MATCH_UM).sum())
            else:
                n_near = 0
            q2["merge(>=2 GT here)" if n_near >= 2 else "link(<=1 GT here)"] += 1

    return stems, n_gt_div, n_pred_fork, q1, q2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--embryo", type=str, default=None, help="e.g. 44b6")
    args = ap.parse_args()

    stems, n_gt_div, n_fork, q1, q2 = diagnose(args.gt_dir, args.pred_dir, args.embryo)
    print(f"videos: {len(stems)}   GT divisions: {n_gt_div}   predicted forks: {n_fork}\n")

    print("Q1  Missed-division bottleneck -- are the GT daughters detected?")
    tot = max(1, sum(q1.values()))
    for k in ("both", "one", "none"):
        v = q1[k]
        tag = {"both": "LINKING miss (cells detected, fork not linked)",
               "one": "partial detection",
               "none": "DETECTION/segmentation miss"}[k]
        print(f"    {k:5s} {v:4d}  ({100*v/tot:4.1f}%)  {tag}")

    print("\nQ2  False-fork nature -- merge vs link (Ultrack test)")
    tot2 = max(1, sum(q2.values()))
    for k in ("true_division", "merge(>=2 GT here)", "link(<=1 GT here)"):
        if k in q2:
            print(f"    {k:20s} {q2[k]:4d}  ({100*q2[k]/tot2:4.1f}%)")

    both = q1["both"]
    print("\nVERDICT:")
    if both / tot >= 0.6:
        print(f"  {100*both/tot:.0f}% of missed divisions have BOTH daughters detected ->")
        print("  the recall ceiling is a LINKING problem. Phase 2 targets the linker")
        print("  (division-aware edges / offset regression), not segmentation.")
    else:
        print(f"  only {100*both/tot:.0f}% have both daughters detected ->")
        print("  detection/segmentation is losing daughters. Phase 2 must go upstream.")


if __name__ == "__main__":
    main()
