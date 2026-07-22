#!/usr/bin/env python
"""Does the transformer's learned candidate score separate real division sisters
from false forks? (Deciding whether learned edge-probs can drive division recovery
without training a new head.)

The pre-ILP candidate graph keeps, per parent, up to `max_children_per_node`
scored out-edges (edge_prob). For a real division the parent has two children; the
*weaker* of the two edges is what a recoverer must trust. We compare, on fold 0:

  - real divisions: for each GT divider matched to a candidate node, the two
    candidate out-edges to the matched daughters -> min(edge_prob).
  - false forks: every other candidate node with >=2 out-edges -> min of its top-2
    edge_probs.

If the real-division distribution sits clearly above the false-fork one, a simple
threshold on the second-edge prob recovers divisions with usable precision.

Usage:
    python analyze_candidate_divisions.py --gt-dir GT --cand-dir CAND [--embryo 44b6]
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import tracksdata as td

VOXEL = np.array([1.625, 0.40625, 0.40625])
MATCH_UM = 7.0


def _load(p: Path):
    r = td.graph.IndexedRXGraph.from_geff(str(p))
    return r[0] if isinstance(r, tuple) else r


def tables(g, with_prob=False):
    N = g.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    ids = N["node_id"].to_list()
    node = {i: (t, np.array([z, y, x]) * VOXEL) for i, t, z, y, x in
            zip(ids, N["t"].to_list(), N["z"].to_list(),
                N["y"].to_list(), N["x"].to_list())}
    keys = ["source_id", "target_id"] + (["edge_prob"] if with_prob else [])
    E = g.edge_attrs(attr_keys=keys)
    out = collections.defaultdict(list)
    src, dst = E["source_id"].to_list(), E["target_id"].to_list()
    pr = E["edge_prob"].to_list() if with_prob else [1.0] * len(src)
    for s, d, p in zip(src, dst, pr):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--cand-dir", type=Path, required=True)
    ap.add_argument("--embryo", type=str, default=None)
    args = ap.parse_args()

    stems = sorted(p.stem for p in args.cand_dir.glob("*.geff")
                   if (args.gt_dir / f"{p.stem}.geff").exists()
                   and (args.embryo is None or p.stem.startswith(args.embryo)))

    real_min, false_min = [], []
    matched_divs = 0
    for stem in stems:
        gnode, gout = tables(_load(args.gt_dir / f"{stem}.geff"))
        cnode, cout = tables(_load(args.cand_dir / f"{stem}.geff"), with_prob=True)
        cframe = by_frame(cnode)

        gt_divs = [(s, [d for d, _ in ds]) for s, ds in gout.items() if len(ds) >= 2]
        real_parents = set()

        for parent, kids in gt_divs:
            tp, pp = gnode[parent]
            dp, cpar = nearest(cframe, tp, pp)
            if cpar is None or dp > MATCH_UM:
                continue
            # match each GT daughter to a candidate node, then find the parent's
            # candidate out-edge prob to that matched node
            child_probs = {d: p for d, p in cout.get(cpar, [])}
            got = []
            for c in kids[:2]:
                tc, pc = gnode[c]
                dc, cc = nearest(cframe, tc, pc)
                if cc is not None and dc <= MATCH_UM and cc in child_probs:
                    got.append(child_probs[cc])
            if len(got) == 2:
                real_min.append(min(got))
                real_parents.add(cpar)
                matched_divs += 1

        # false forks: candidate nodes with >=2 out-edges that are not real dividers
        for s, ds in cout.items():
            if len(ds) >= 2 and s not in real_parents:
                top2 = sorted((p for _, p in ds), reverse=True)[:2]
                false_min.append(min(top2))

    real_min = np.array(real_min)
    false_min = np.array(false_min)
    print(f"videos {len(stems)}   real divisions matched w/ both daughters: {matched_divs}")
    print(f"candidate false forks: {len(false_min)}\n")

    def stats(a, name):
        if len(a) == 0:
            print(f"  {name}: (none)"); return
        q = np.percentile(a, [10, 50, 90])
        print(f"  {name:16s} n={len(a):4d}  min-edge-prob  p10={q[0]:.3f} med={q[1]:.3f} p90={q[2]:.3f}")
    stats(real_min, "real divisions")
    stats(false_min, "false forks")

    # separability: at each threshold on the weaker edge prob, precision/recall
    if len(real_min) and len(false_min):
        print("\n  threshold  recall(real)  #false>=thr  precision")
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            tp = int((real_min >= thr).sum())
            fp = int((false_min >= thr).sum())
            rec = tp / len(real_min)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            print(f"    {thr:.2f}      {rec:5.2f}        {fp:5d}      {prec:.3f}")
        print("\n  (A usable threshold has high recall AND non-trivial precision. If")
        print("   precision stays ~random at all thresholds, the learned linking score")
        print("   is not division-discriminative -> need a trained division head.)")


if __name__ == "__main__":
    main()
