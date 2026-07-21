#!/usr/bin/env python
"""Phase 1 post-processing on predicted .geff tracking graphs.

Operates on a directory of predicted `.geff` graphs and writes a filtered copy
to a new directory, so it composes after `predict_unet_transformer.py` without
re-running the model. Each transform is independently switchable and each is
gated on a paired bootstrap (see `compare_configs.py`) before it ships.

Currently implemented
---------------------
--min-track-len N   Drop every weakly-connected lineage component whose temporal
                    span (max_t - min_t + 1) is < N frames. Rationale: the ILP
                    leaves short spurious fragments that hurt the size-weighted
                    edge Jaccard. A division TP needs an intact 5-generation
                    window, so we DON'T want to delete real (short) lineages that
                    contain a division -- --keep-division-components (default on)
                    exempts any component containing a node of out-degree >= 2.

Why span, not node count: a lineage is a tree once divisions appear; span is the
biologically meaningful "how long did we track this clone" and is invariant to
how many branches it has.

Usage
-----
    python postprocess.py --in-dir PRED --out-dir PRED_FILTERED --min-track-len 6
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import tracksdata as td


def _load(geff_path: Path) -> td.graph.BaseGraph:
    res = td.graph.IndexedRXGraph.from_geff(str(geff_path))
    return res[0] if isinstance(res, tuple) else res


class _UnionFind:
    def __init__(self, ids):
        self.p = {i: i for i in ids}

    def find(self, x):
        root = x
        while self.p[root] != root:
            root = self.p[root]
        while self.p[x] != root:  # path compression
            self.p[x], x = root, self.p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def short_component_nodes(
    graph: td.graph.BaseGraph, min_track_len: int, keep_division_components: bool
) -> list[int]:
    """Return the node ids belonging to components shorter than min_track_len."""
    nodes = graph.node_attrs(attr_keys=["node_id", "t"])
    node_ids = nodes["node_id"].to_list()
    node_t = dict(zip(node_ids, nodes["t"].to_list()))
    if not node_ids:
        return []

    edges = graph.edge_attrs(attr_keys=["source_id", "target_id"])
    src = edges["source_id"].to_list()
    dst = edges["target_id"].to_list()

    uf = _UnionFind(node_ids)
    for s, d in zip(src, dst):
        uf.union(s, d)

    # Nodes that are the parent side of a division (out-degree >= 2).
    dividing = set(graph.dividing_nodes()) if keep_division_components else set()

    comp_nodes: dict[int, list[int]] = {}
    comp_has_div: dict[int, bool] = {}
    for n in node_ids:
        r = uf.find(n)
        comp_nodes.setdefault(r, []).append(n)
        if n in dividing:
            comp_has_div[r] = True

    drop: list[int] = []
    for r, members in comp_nodes.items():
        if comp_has_div.get(r, False):
            continue
        ts = [node_t[n] for n in members]
        span = max(ts) - min(ts) + 1
        if span < min_track_len:
            drop.extend(members)
    return drop


def process_geff(
    in_geff: Path, out_geff: Path, min_track_len: int, keep_division_components: bool
) -> tuple[int, int, int, int]:
    graph = _load(in_geff)
    n0, e0 = graph.num_nodes(), graph.num_edges()

    if min_track_len > 1:
        drop = short_component_nodes(graph, min_track_len, keep_division_components)
        if drop:
            graph.bulk_remove_nodes(drop)

    if out_geff.exists():
        shutil.rmtree(out_geff)
    out_geff.parent.mkdir(parents=True, exist_ok=True)
    graph.to_geff(str(out_geff))
    return n0, e0, graph.num_nodes(), graph.num_edges()


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 graph post-processing.")
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-track-len", type=int, default=1,
                    help="Drop lineage components spanning < N frames (1 = off).")
    ap.add_argument("--no-keep-division-components", action="store_true",
                    help="Also drop short components even if they contain a division.")
    args = ap.parse_args()

    geffs = sorted(args.in_dir.glob("*.geff"))
    if not geffs:
        raise SystemExit(f"no .geff in {args.in_dir}")

    tot_n0 = tot_n1 = tot_e0 = tot_e1 = 0
    for g in geffs:
        n0, e0, n1, e1 = process_geff(
            g, args.out_dir / g.name, args.min_track_len,
            not args.no_keep_division_components,
        )
        tot_n0 += n0; tot_n1 += n1; tot_e0 += e0; tot_e1 += e1
        print(f"{g.stem}: nodes {n0}->{n1} ({n0-n1} dropped), edges {e0}->{e1}")

    print(f"\nTOTAL nodes {tot_n0}->{tot_n1} ({tot_n0-tot_n1} dropped, "
          f"{100*(tot_n0-tot_n1)/max(1,tot_n0):.1f}%), edges {tot_e0}->{tot_e1}")


if __name__ == "__main__":
    main()
