#!/usr/bin/env python
"""B3.1 -- does the unbalanced-OT division signal survive DETECTOR output?

The GT-node probe (ot_division_probe.py) separated dividers from non-dividers at
Cohen's d ~ 2.9 on clean, sparse ground-truth points. The real question is whether
that survives dense detector output, where a divider sits among many non-daughter
neighbours that dilute the mass-splitting signal.

Here the OT runs over the PREDICTED detections (positions), and a source detection
is labelled a divider iff it matches (<=7 um) a GT divider at that frame. We sweep
(epsilon, tau) and report Cohen's d between divider and non-divider row sums.

PRE-REGISTERED KILL CRITERION: if no (epsilon, tau) reaches Cohen's d >= 1.0, the
signal did not survive real detections -> stop Track B's OT-as-detector line.

Usage:
    python ot_probe_detector.py --gt-dir GT --pred-dir PRED --embryo 44b6
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np

VOXEL = np.array([1.625, 0.40625, 0.40625])
MATCH_UM = 7.0


def _lse(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def unbalanced_sinkhorn(a, b, cost, epsilon, tau, num_iters=200):
    lam = tau / (tau + epsilon)
    log_a, log_b = np.log(np.maximum(a, 1e-30)), np.log(np.maximum(b, 1e-30))
    log_K = -cost / epsilon
    f = np.zeros_like(a); g = np.zeros_like(b)
    for _ in range(num_iters):
        f = lam * log_a - lam * _lse(log_K + g[None, :], axis=1)
        g = lam * log_b - lam * _lse(log_K + f[:, None], axis=0)
    return np.exp(log_K + f[:, None] + g[None, :])


def _load(p):
    import tracksdata as td
    r = td.graph.IndexedRXGraph.from_geff(str(p))
    return r[0] if isinstance(r, tuple) else r


def frames_and_divs(gt_geff, pred_geff):
    """pred nodes by frame (positions) + GT divider positions by frame."""
    g = _load(gt_geff)
    N, E = g.node_attrs(), g.edge_attrs()
    gid = np.asarray(N["node_id"]); gt = np.asarray(N["t"])
    gpos = np.stack([np.asarray(N["z"]), np.asarray(N["y"]), np.asarray(N["x"])], 1).astype(float) * VOXEL
    gpos_by = {int(i): (int(tt), p) for i, tt, p in zip(gid, gt, gpos)}
    outdeg = collections.Counter(np.asarray(E["source_id"]).tolist())
    gdiv_by_t = collections.defaultdict(list)
    for s, k in outdeg.items():
        if k >= 2:
            tt, p = gpos_by[int(s)]
            gdiv_by_t[tt].append(p)

    p = _load(pred_geff)
    Np = p.node_attrs(attr_keys=["t", "z", "y", "x"])
    pt = np.asarray(Np["t"])
    ppos = np.stack([np.asarray(Np["z"]), np.asarray(Np["y"]), np.asarray(Np["x"])], 1).astype(float) * VOXEL
    pred_by_t = collections.defaultdict(list)
    for tt, pp in zip(pt, ppos):
        pred_by_t[int(tt)].append(pp)
    pred_by_t = {t: np.stack(v) for t, v in pred_by_t.items()}
    return pred_by_t, gdiv_by_t


def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return (a.mean() - b.mean()) / max(np.sqrt(0.5 * (a.var() + b.var())), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--embryo", type=str, default="44b6")
    ap.add_argument("--epsilons", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    ap.add_argument("--taus", type=float, nargs="+", default=[1.0, 2.0, 5.0])
    args = ap.parse_args()

    stems = sorted(p.stem for p in args.pred_dir.glob(f"{args.embryo}_*.geff")
                   if (args.gt_dir / f"{p.stem}.geff").exists())

    # precompute frame data
    data = []
    for stem in stems:
        pred_by_t, gdiv_by_t = frames_and_divs(
            args.gt_dir / f"{stem}.geff", args.pred_dir / f"{stem}.geff")
        if any(gdiv_by_t.values()):
            data.append((pred_by_t, gdiv_by_t))
    n_div = sum(len(v) for _, gd in data for v in gd.values())
    print(f"videos with GT divisions: {len(data)}   GT dividers: {n_div}\n")

    print(f"{'eps':>5} {'tau':>5} | {'divider rs':>18} | {'non-div rs':>18} | {'d':>6}")
    print("-" * 62)
    best = -1e9
    for eps in args.epsilons:
        for tau in args.taus:
            div_rs, non_rs = [], []
            for pred_by_t, gdiv_by_t in data:
                for t in sorted(gdiv_by_t):
                    if t not in pred_by_t or (t + 1) not in pred_by_t:
                        continue
                    S, Dst = pred_by_t[t], pred_by_t[t + 1]
                    C = ((S[:, None, :] - Dst[None, :, :]) ** 2).sum(-1)
                    C = C / max(C.mean(), 1e-9)
                    T = unbalanced_sinkhorn(np.ones(len(S)), np.ones(len(Dst)), C, eps, tau)
                    rs = T.sum(1)
                    divpos = gdiv_by_t[t]
                    for k in range(len(S)):
                        is_div = any(np.linalg.norm(S[k] - dp) <= MATCH_UM for dp in divpos)
                        (div_rs if is_div else non_rs).append(rs[k])
            d = cohens_d(div_rs, non_rs)
            best = max(best, d if d == d else -1e9)
            print(f"{eps:5.2f} {tau:5.1f} | {np.mean(div_rs):7.3f} +/-{np.std(div_rs):<7.3f} | "
                  f"{np.mean(non_rs):7.3f} +/-{np.std(non_rs):<7.3f} | {d:6.2f}")

    print(f"\nBEST Cohen's d = {best:.2f}   (GT-node probe reached ~2.9)")
    if best >= 1.0:
        print("PASS -- OT signal survives detector output. Proceed to B3.2 (vs naive")
        print("geometric NN-ratio baseline).")
    else:
        print("FAIL (d<1.0) -- the mass-splitting signal is washed out by dense")
        print("neighbours in real detections. Stop the OT-as-detector line; the GT-only")
        print("result stands as a characterisation, not a usable detector.")


if __name__ == "__main__":
    main()
