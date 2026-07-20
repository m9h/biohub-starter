#!/usr/bin/env python
"""Build embryo-disjoint CV splits for the Biohub cell-tracking competition.

WHY THIS EXISTS
---------------
The 199 training samples come from only TWO embryos (`6bba`, `44b6`); the
sample name is `{embryo_id}_{field_of_view}`.  The competition's hidden test
set is *embryo-disjoint* from training.

The baseline repo's default split is random **by sample**, so both embryos land
on both sides -- every score measured that way is inflated by embryo leakage.
Of six top public notebooks surveyed (2026-07-19), *none* did any
cross-validation at all; all tuned directly against the 29% public leaderboard.

This script emits a 2-fold embryo-held-out split, which is the only local proxy
for the real test condition.

Usage
-----
    python make_embryo_splits.py --data-dir /path/to/train
    # -> writes <data-dir>/embryo_splits.json

Then point the baseline scripts at it:

    predict_unet_transformer.py --splits <data-dir>/embryo_splits.json --split 0
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def sample_stems(data_dir: Path) -> list[str]:
    """Return sample names having BOTH a .zarr volume and a .geff graph."""
    return sorted(
        p.name[:-5]
        for p in data_dir.glob("*.zarr")
        if (data_dir / f"{p.name[:-5]}.geff").exists()
    )


def embryo_of(stem: str) -> str:
    """`44b6_0113de3b` -> `44b6`. Folder names are {embryo_id}_{field_of_view}."""
    return stem.split("_")[0]


def build_folds(stems: list[str]) -> list[dict]:
    """One fold per embryo: hold that embryo out, train on the rest."""
    by_embryo: dict[str, list[str]] = {}
    for s in stems:
        by_embryo.setdefault(embryo_of(s), []).append(s)

    folds = []
    for i, held_out in enumerate(sorted(by_embryo)):
        test = by_embryo[held_out]
        train = [s for e, ss in by_embryo.items() if e != held_out for s in ss]
        folds.append({"split": i, "held_out_embryo": held_out,
                      "train": train, "test": test})
    return folds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Directory containing *.zarr and *.geff (the train/ folder).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path (default: <data-dir>/embryo_splits.json)")
    args = ap.parse_args()

    stems = sample_stems(args.data_dir)
    if not stems:
        raise SystemExit(f"No paired .zarr/.geff samples found in {args.data_dir}")

    counts = Counter(embryo_of(s) for s in stems)
    print(f"{len(stems)} samples across {len(counts)} embryo(s):")
    for e, n in counts.most_common():
        print(f"  {e}: {n}")

    if len(counts) < 2:
        raise SystemExit("Need >= 2 embryos to build an embryo-disjoint split.")

    folds = build_folds(stems)
    for f in folds:
        overlap = set(f["train"]) & set(f["test"])
        assert not overlap, f"LEAKAGE in split {f['split']}: {sorted(overlap)[:3]}"
        print(f"  fold {f['split']}: hold out {f['held_out_embryo']} "
              f"-> {len(f['train'])} train / {len(f['test'])} test")

    out = args.out or (args.data_dir / "embryo_splits.json")
    out.write_text(json.dumps(folds, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
