#!/usr/bin/env python
"""Does unbalanced OT detect cell division as marginal mass imbalance?

HYPOTHESIS
----------
A dividing mother transports mass to TWO daughters, so in a transport plan T
between frame t and t+1 its ROW SUM should be ~2, while a non-dividing cell's
row sums to ~1. Balanced OT cannot express this (it pins T1 == a exactly).
Unbalanced OT relaxes it with a KL penalty, and the tau that governs that
penalty becomes a mitosis prior.

This probes whether ANY tau separates dividing from non-dividing rows, using
ground-truth nodes only (no detector noise). If it fails here, it cannot work
downstream.

Implements the same log-domain unbalanced Sinkhorn as devograph._ot:
    min_T <C,T> + eps*KL(T||ab^T) + tau*KL(T1||a) + tau*KL(T^T1||b)
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np

VOXEL = np.array([1.625, 0.40625, 0.40625])  # z, y, x um/voxel


def unbalanced_sinkhorn(a, b, cost, epsilon=0.1, tau=1.0, num_iters=200):
    """Log-domain unbalanced Sinkhorn. Returns transport plan T."""
    lam = tau / (tau + epsilon)
    log_a, log_b = np.log(np.maximum(a, 1e-30)), np.log(np.maximum(b, 1e-30))
    log_K = -cost / epsilon
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    for _ in range(num_iters):
        f = lam * log_a - lam * _lse(log_K + g[None, :], axis=1)
        g = lam * log_b - lam * _lse(log_K + f[:, None], axis=0)
    return np.exp(log_K + f[:, None] + g[None, :])


def _lse(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def collect_frame_pairs(geff_path):
    """Yield (t, src_ids, src_xyz, dst_ids, dst_xyz, dividing_src_ids)."""
    import tracksdata as td

    g, _ = td.graph.IndexedRXGraph.from_geff(str(geff_path))
    N, E = g.node_attrs(), g.edge_attrs()
    nid = np.asarray(N["node_id"])
    t = np.asarray(N["t"])
    pos = np.stack([np.asarray(N["z"]), np.asarray(N["y"]), np.asarray(N["x"])], 1).astype(float)
    pos_um = pos * VOXEL
    by_id = {int(i): (int(tt), p) for i, tt, p in zip(nid, t, pos_um)}

    src = np.asarray(E["source_id"])
    dst = np.asarray(E["target_id"])
    outdeg = collections.Counter(src.tolist())
    divs = {int(s) for s, k in outdeg.items() if k >= 2}

    # group GT nodes by timepoint
    per_t = collections.defaultdict(list)
    for i, (tt, _) in by_id.items():
        per_t[tt].append(i)

    for tt in sorted(per_t):
        if tt + 1 not in per_t:
            continue
        s_ids = sorted(per_t[tt])
        d_ids = sorted(per_t[tt + 1])
        if not s_ids or not d_ids:
            continue
        yield (tt,
               s_ids, np.stack([by_id[i][1] for i in s_ids]),
               d_ids, np.stack([by_id[i][1] for i in d_ids]),
               {i for i in s_ids if i in divs})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="max geffs to scan")
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    args = ap.parse_args()

    geffs = sorted(args.data_dir.glob("*.geff"))
    if args.limit:
        geffs = geffs[: args.limit]

    # gather only frame pairs that CONTAIN a division, plus controls
    div_pairs, ctrl_pairs = [], []
    for p in geffs:
        for rec in collect_frame_pairs(p):
            (div_pairs if rec[5] else ctrl_pairs).append(rec)
    print(f"scanned {len(geffs)} videos")
    print(f"frame pairs WITH a division : {len(div_pairs)}")
    print(f"frame pairs without         : {len(ctrl_pairs)}")
    if not div_pairs:
        raise SystemExit("no division-containing frame pairs found")

    print(f"\nepsilon={args.epsilon}   cost = squared um distance / mean")
    print(f"{'tau':>6} | {'divider rowsum':>22} | {'non-divider rowsum':>22} | {'sep':>6}")
    print("-" * 68)

    for tau in args.taus:
        div_sums, non_sums = [], []
        for (_t, s_ids, s_xyz, d_ids, d_xyz, divs) in div_pairs:
            C = ((s_xyz[:, None, :] - d_xyz[None, :, :]) ** 2).sum(-1)
            C = C / max(C.mean(), 1e-9)
            a = np.ones(len(s_ids))
            b = np.ones(len(d_ids))
            T = unbalanced_sinkhorn(a, b, C, epsilon=args.epsilon, tau=tau)
            rs = T.sum(1)
            for k, sid in enumerate(s_ids):
                (div_sums if sid in divs else non_sums).append(rs[k])
        d_, n_ = np.array(div_sums), np.array(non_sums)
        # separation = normalised difference of means
        sep = (d_.mean() - n_.mean()) / max(np.sqrt(0.5 * (d_.var() + n_.var())), 1e-9)
        print(f"{tau:6.1f} | {d_.mean():8.3f} +/- {d_.std():<9.3f} | "
              f"{n_.mean():8.3f} +/- {n_.std():<9.3f} | {sep:6.2f}")

    print(f"\nn dividers={len(div_sums)}  n non-dividers={len(non_sums)}")
    print("sep = (mean_div - mean_nondiv) / pooled_sd   (Cohen's d)")
    print("Interpretation: hypothesis needs divider rowsum ~2 vs ~1, sep >> 1.")


if __name__ == "__main__":
    main()
