#!/usr/bin/env python
"""Measure the properties of the Biohub cell-tracking training set that
actually drive strategy.

Reports:
  1. Embryo composition and volume-shape uniformity
  2. Annotation sparsity (annotated nodes per timepoint vs. real cell count)
  3. Division statistics -- count, per-video rate, distribution over time
  4. A synchrony test (KS vs uniform) for the Kane & Kimmel MBT prior

Every number in ../FINDINGS.md is reproduced by this script.

Usage
-----
    python analyze_dataset.py --data-dir /path/to/train
    python analyze_dataset.py --data-dir /path/to/train --limit 25   # quick
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only read the first N geffs (faster, approximate).")
    args = ap.parse_args()

    import tracksdata as td
    import zarr

    geffs = sorted(args.data_dir.glob("*.geff"))
    if args.limit:
        geffs = geffs[: args.limit]
    if not geffs:
        raise SystemExit(f"No .geff files in {args.data_dir}")

    # ---- 1. embryo composition + shape uniformity -------------------------
    embryos = collections.Counter(p.name.split("_")[0] for p in geffs)
    print("=" * 62)
    print(f"EMBRYOS  ({len(geffs)} samples, {len(embryos)} embryo(s))")
    for e, n in embryos.most_common():
        print(f"  {e}: {n}")
    print("  NOTE: hidden test is embryo-DISJOINT -> random splits leak.")

    shapes = collections.Counter()
    for p in geffs:
        z = args.data_dir / f"{p.stem}.zarr"
        if z.exists():
            shapes[zarr.open_group(str(z), mode="r")["0"].shape] += 1
    print(f"\nVOLUME SHAPES ({len(shapes)} distinct)")
    for s, n in shapes.most_common(5):
        print(f"  {n} x {s}")

    # ---- 2/3. annotation sparsity + divisions -----------------------------
    tot_nodes = 0
    per_tp: list[float] = []
    div_times: list[tuple[str, int]] = []
    divs_per_video: list[int] = []

    for p in geffs:
        g, _ = td.graph.IndexedRXGraph.from_geff(str(p))
        N, E = g.node_attrs(), g.edge_attrs()
        t = np.asarray(N["t"])
        node_t = dict(zip(np.asarray(N["node_id"]).tolist(), t.tolist()))
        tot_nodes += len(t)
        per_tp.append(len(t) / max(1, len(np.unique(t))))

        outdeg = collections.Counter(np.asarray(E["source_id"]).tolist())
        d = [nid for nid, k in outdeg.items() if k >= 2]
        divs_per_video.append(len(d))
        emb = p.name.split("_")[0]
        div_times += [(emb, node_t[nid]) for nid in d]

    print("\n" + "=" * 62)
    print("ANNOTATION SPARSITY")
    print(f"  total annotated nodes : {tot_nodes}")
    print(f"  annotated per timepoint: mean {np.mean(per_tp):.2f}  max {max(per_tp):.1f}")
    print("  NOTE: a volume holds ~30-130 real nuclei -> only a few % are labelled.")
    print("  Unlabelled cells are NOT negatives (positive-unlabelled problem).")

    print("\n" + "=" * 62)
    print("DIVISIONS  (node with out-degree >= 2)")
    print(f"  total            : {len(div_times)}")
    print(f"  per video        : mean {np.mean(divs_per_video):.2f}  max {max(divs_per_video)}")
    print(f"  videos with zero : {sum(1 for x in divs_per_video if x == 0)}/{len(divs_per_video)}")
    print("  -> use this rate to calibrate division post-processing;")
    print("     public notebooks reverse-engineer it from the leaderboard instead.")

    # ---- 4. synchrony test ------------------------------------------------
    print("\n" + "=" * 62)
    print("SYNCHRONY TEST (Kane & Kimmel 1993 MBT prior)")
    ts_all = np.array([t for _, t in div_times])
    if len(ts_all) < 10:
        print("  too few divisions to test")
        return
    for emb in sorted({e for e, _ in div_times}):
        ts = np.array([t for e, t in div_times if e == emb])
        occ = len(np.unique(ts))
        print(f"  {emb}: n={len(ts):3d}  mean t={ts.mean():5.1f} sd={ts.std():5.1f}  "
              f"distinct timepoints={occ}")
    try:
        from scipy import stats
        span = max(1, ts_all.max())
        d, pv = stats.kstest(ts_all / span, "uniform")
        print(f"  KS vs uniform: D={d:.3f} p={pv:.4f}")
    except ImportError:
        print("  (scipy not available -- skipping KS test)")
    print("  MEASURED RESULT: weak/negative. Zebrahub images post-MBT, where")
    print("  Kane & Kimmel synchrony (cycles 1-9) is lost by design.")
    print("  Sister-phase inheritance is the untested, more promising variant.")
    print("  CAVEAT: pools across videos assuming comparable t -- unverified.")


if __name__ == "__main__":
    main()
