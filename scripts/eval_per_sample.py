#!/usr/bin/env python
"""Score a prediction directory and emit NAME-TAGGED per-sample metrics as CSV.

WHY THIS EXISTS
---------------
`scripts/evaluate.py` in the baseline repo returns per-sample rows, but
`per_sample_metrics()` emits only METRIC_COLUMNS -- **no dataset name**. Rows are
associated with names only positionally, inside a loop that discards them, and
there is no CSV output. Without names you cannot do a paired per-video comparison
between two configs, which makes every "improvement" unfalsifiable.

This wraps `evaluate_pairs()` (it does NOT reimplement the metric) and recovers
the name ordering from the same sorted-intersection logic, minus `skipped`.

It also ASSERTS n_adj == n. If a GT geff lacks `estimated_number_of_nodes`,
that sample's adj_edge_jaccard is NaN and it silently drops out of `score` while
still contributing to the micro-averaged edge/division Jaccard -- quietly biasing
any comparison built on the aggregate.

Usage
-----
    python eval_per_sample.py --pred-dir PRED --gt-dir GT --out run.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _repo_scripts_on_path(repo: Path) -> None:
    sys.path.insert(0, str(repo / "scripts"))
    sys.path.insert(0, str(repo / "src"))


def name_order(pred_dir: Path, gt_dir: Path, skipped: list[str]) -> list[str]:
    """Reproduce evaluate_pairs' row ordering: sorted(pred & gt), minus skipped."""
    pred = {p.stem for p in pred_dir.glob("*.geff")}
    gt = {p.stem for p in gt_dir.glob("*.geff")}
    return [n for n in sorted(pred & gt) if n not in set(skipped)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="CSV output path")
    ap.add_argument("--repo", type=Path,
                    default=Path("/mnt/t9/biohub-cell-tracking/kaggle-cell-tracking-competition"),
                    help="baseline repo root (for scripts/evaluate.py)")
    ap.add_argument("--max-distance", type=float, default=7.0)
    ap.add_argument("--label", type=str, default="", help="config label, stored in the CSV")
    ap.add_argument("--allow-nan-adj", action="store_true",
                    help="do not fail when n_adj < n (default: hard error)")
    args = ap.parse_args()

    _repo_scripts_on_path(args.repo)
    from evaluate import evaluate_pairs  # noqa: E402
    from tracking_cellmot.metrics import METRIC_COLUMNS, summarise  # noqa: E402

    rows, skipped = evaluate_pairs(args.pred_dir, args.gt_dir,
                                   max_distance=args.max_distance)
    names = name_order(args.pred_dir, args.gt_dir, skipped)
    if len(names) != len(rows):
        raise SystemExit(
            f"name/row mismatch: {len(names)} names vs {len(rows)} rows. "
            "evaluate_pairs' ordering logic may have changed -- fix name_order()."
        )

    agg = summarise(rows)
    n, n_adj = agg["n"], agg["n_adj"]
    print(f"\naggregate: score={agg['score']:.4f} "
          f"adj_edge_jaccard={agg['adj_edge_jaccard']:.4f} "
          f"division_jaccard={agg['division_jaccard']:.4f} n={n} n_adj={n_adj}")

    if n_adj != n:
        msg = (f"n_adj ({n_adj}) != n ({n}): {n - n_adj} sample(s) are missing "
               f"`estimated_number_of_nodes` and are SILENTLY EXCLUDED from "
               f"adj_edge_jaccard and score, while still contributing to the "
               f"micro-averaged edge/division Jaccard. Any comparison built on "
               f"this aggregate is biased.")
        if not args.allow_nan_adj:
            raise SystemExit(f"ERROR: {msg}\n(pass --allow-nan-adj to override)")
        print(f"WARNING: {msg}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "label", *METRIC_COLUMNS]
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for name, r in zip(names, rows):
            w.writerow({"dataset": name, "label": args.label,
                        **{c: r[c] for c in METRIC_COLUMNS}})
    print(f"wrote {len(rows)} per-sample rows -> {args.out}")


if __name__ == "__main__":
    main()
