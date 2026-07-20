#!/usr/bin/env python
"""Paired comparison of two configs over the SAME held-out videos.

WHY A BOOTSTRAP AND NOT A t-TEST
--------------------------------
The competition statistic is not a mean of per-video scores:

    adj_edge_jaccard = sum_i(w_i * adj_i) / sum_i(w_i),  w_i = TP_i + FP_i + FN_i
    division_jaccard = MICRO-averaged (counts summed, then Jaccard)
    score            = adj_edge_jaccard + 0.1 * division_jaccard

A paired t-test on unweighted per-video means therefore does NOT test the quantity
the leaderboard reports. Instead we bootstrap over videos, recomputing the full
weighted/micro-averaged statistic on each resample, so the CI is for the actual
score. Videos are resampled as PAIRS, preserving the pairing.

Usage
-----
    python compare_configs.py --a base.csv --b variant.csv
    python compare_configs.py --a base.csv --b variant.csv --metric adj_edge_jaccard
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

DIV_WEIGHT = 0.1  # tracking_cellmot.metrics.SCORE_DIVISION_WEIGHT


def load(path: Path) -> dict[str, dict]:
    with path.open() as fh:
        return {r["dataset"]: r for r in csv.DictReader(fh)}


def _f(r: dict, k: str) -> float:
    v = r[k]
    return float("nan") if v == "" else float(v)


def statistic(rows: list[dict], metric: str) -> float:
    """Recompute the aggregate exactly as tracking_cellmot.metrics.summarise does."""
    valid = [r for r in rows if _f(r, "edge_tp") == _f(r, "edge_tp")]
    if not valid:
        return float("nan")

    adj_rows = [r for r in valid if _f(r, "adj_edge_jaccard") == _f(r, "adj_edge_jaccard")]
    w = np.array([_f(r, "edge_tp") + _f(r, "edge_fp") + _f(r, "edge_fn") for r in adj_rows])
    a = np.array([_f(r, "adj_edge_jaccard") for r in adj_rows])
    adj = float((w * a).sum() / w.sum()) if w.sum() > 0 else float("nan")
    if metric == "adj_edge_jaccard":
        return adj

    dtp = sum(_f(r, "division_tp") for r in valid)
    dfp = sum(_f(r, "division_fp") for r in valid)
    dfn = sum(_f(r, "division_fn") for r in valid)
    den = dtp + dfp + dfn
    div = dtp / den if den > 0 else float("nan")
    if metric == "division_jaccard":
        return div

    etp = sum(_f(r, "edge_tp") for r in valid)
    efp = sum(_f(r, "edge_fp") for r in valid)
    efn = sum(_f(r, "edge_fn") for r in valid)
    if metric == "edge_jaccard":
        d = etp + efp + efn
        return etp / d if d > 0 else float("nan")

    if metric == "score":
        return adj if den == 0 else adj + DIV_WEIGHT * div
    raise ValueError(f"unknown metric {metric}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="baseline CSV")
    ap.add_argument("--b", type=Path, required=True, help="variant CSV")
    ap.add_argument("--metric", default="score",
                    choices=["score", "adj_edge_jaccard", "division_jaccard", "edge_jaccard"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    shared = sorted(set(A) & set(B))
    if not shared:
        raise SystemExit("no shared datasets between the two CSVs")
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    if only_a or only_b:
        print(f"WARNING: unpaired datasets dropped -- only in A: {len(only_a)}, "
              f"only in B: {len(only_b)}")

    ra = [A[n] for n in shared]
    rb = [B[n] for n in shared]
    sa, sb = statistic(ra, args.metric), statistic(rb, args.metric)

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(shared))
    deltas = np.empty(args.n_boot)
    for k in range(args.n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)  # resample PAIRS
        deltas[k] = (statistic([rb[i] for i in pick], args.metric)
                     - statistic([ra[i] for i in pick], args.metric))

    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    # two-sided bootstrap p: how often the resampled delta crosses zero
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())

    label_a = ra[0].get("label") or args.a.stem
    label_b = rb[0].get("label") or args.b.stem
    print(f"\npaired over {len(shared)} videos   metric = {args.metric}")
    print(f"  A  {label_a:28s} {sa:.4f}")
    print(f"  B  {label_b:28s} {sb:.4f}")
    print(f"  delta (B - A)                {sb - sa:+.4f}")
    print(f"  95% CI                       [{lo:+.4f}, {hi:+.4f}]")
    print(f"  bootstrap p                  {p:.4f}   ({args.n_boot} resamples)")
    verdict = ("SIGNIFICANT" if (lo > 0 or hi < 0) else
               "not significant (CI spans 0)")
    print(f"  verdict                      {verdict}")


if __name__ == "__main__":
    main()
