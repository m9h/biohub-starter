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


VOXEL_SCALE = np.array([1.625, 0.40625, 0.40625])  # z, y, x microns/voxel


def motion_relink(graph: td.graph.BaseGraph, max_um: float) -> int:
    """Reconnect prematurely-terminated tracks to orphan track-starts one frame
    later, using a constant-velocity prediction and gated Hungarian assignment.

    A "loose end" is a node with out-degree 0 that is not in the last frame; an
    "orphan start" is a node with in-degree 0 that is not in the first frame. Both
    conditions guarantee that adding a single loose_end -> orphan edge creates
    neither a division (loose end had out-degree 0) nor a merge (orphan had
    in-degree 0), so relinking never fabricates lineage branch points.

    Velocity is estimated from the loose end's own incoming edge; with no history
    the prediction falls back to a stationary node. Returns the number of edges
    added.
    """
    from scipy.optimize import linear_sum_assignment

    N = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    ids = N["node_id"].to_list()
    t = dict(zip(ids, N["t"].to_list()))
    pos = {i: np.array([z, y, x]) * VOXEL_SCALE
           for i, z, y, x in zip(ids, N["z"].to_list(), N["y"].to_list(), N["x"].to_list())}
    if not ids:
        return 0
    t_min, t_max = min(t.values()), max(t.values())

    E = graph.edge_attrs(attr_keys=["source_id", "target_id"])
    src = E["source_id"].to_list()
    dst = E["target_id"].to_list()
    out_deg: dict[int, int] = {}
    in_deg: dict[int, int] = {}
    parent: dict[int, int] = {}  # target -> its (single) source, for velocity
    for s, d in zip(src, dst):
        out_deg[s] = out_deg.get(s, 0) + 1
        in_deg[d] = in_deg.get(d, 0) + 1
        parent[d] = s

    loose = [i for i in ids if out_deg.get(i, 0) == 0 and t[i] < t_max]
    orphan = [i for i in ids if in_deg.get(i, 0) == 0 and t[i] > t_min]
    if not loose or not orphan:
        return 0

    orphan_by_t: dict[int, list[int]] = {}
    for o in orphan:
        orphan_by_t.setdefault(t[o], []).append(o)

    added = 0
    new_edges = []
    for L in sorted({t[i] for i in loose}):
        ends = [i for i in loose if t[i] == L]
        starts = orphan_by_t.get(L + 1, [])
        if not starts:
            continue
        # constant-velocity predicted position of each loose end at frame L+1
        pred = {}
        for e in ends:
            v = pos[e] - pos[parent[e]] if e in parent else np.zeros(3)
            pred[e] = pos[e] + v
        cost = np.full((len(ends), len(starts)), 1e6)
        for a, e in enumerate(ends):
            for b, s in enumerate(starts):
                dist = float(np.linalg.norm(pred[e] - pos[s]))
                if dist <= max_um:
                    cost[a, b] = dist
        rows, cols = linear_sum_assignment(cost)
        for a, b in zip(rows, cols):
            if cost[a, b] <= max_um:
                new_edges.append({"source_id": ends[a], "target_id": starts[b],
                                  "solution": True, "edge_dist": float(cost[a, b]),
                                  "edge_prob": 0.5})
                added += 1
    if new_edges:
        graph.bulk_add_edges(new_edges)
    return added


def process_geff(
    in_geff: Path, out_geff: Path, min_track_len: int, keep_division_components: bool,
    relink_max_um: float = 0.0,
) -> tuple[int, int, int, int]:
    graph = _load(in_geff)
    n0, e0 = graph.num_nodes(), graph.num_edges()

    # Order matters: relink first (rescue fragments into longer tracks), THEN
    # filter by length, so the filter judges tracks after they've been repaired.
    if relink_max_um > 0:
        motion_relink(graph, relink_max_um)

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
    ap.add_argument("--relink-max-um", type=float, default=0.0,
                    help="Motion-relink loose ends to next-frame orphans within N um "
                         "(0 = off). Runs before the length filter.")
    args = ap.parse_args()

    geffs = sorted(args.in_dir.glob("*.geff"))
    if not geffs:
        raise SystemExit(f"no .geff in {args.in_dir}")

    tot_n0 = tot_n1 = tot_e0 = tot_e1 = 0
    for g in geffs:
        n0, e0, n1, e1 = process_geff(
            g, args.out_dir / g.name, args.min_track_len,
            not args.no_keep_division_components, args.relink_max_um,
        )
        tot_n0 += n0; tot_n1 += n1; tot_e0 += e0; tot_e1 += e1
        print(f"{g.stem}: nodes {n0}->{n1} ({n0-n1} dropped), edges {e0}->{e1}")

    print(f"\nTOTAL nodes {tot_n0}->{tot_n1} ({tot_n0-tot_n1} dropped, "
          f"{100*(tot_n0-tot_n1)/max(1,tot_n0):.1f}%), edges {tot_e0}->{tot_e1}")


if __name__ == "__main__":
    main()
