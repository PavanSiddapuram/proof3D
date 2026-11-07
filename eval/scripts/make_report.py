"""Regenerate the results tables from stored results (standards 7: ``make report``).

Input: every ``eval/results/**/*.json`` file, one per (product, condition, held-out photo):

    {"product": "mug-01", "condition": "C", "held_out": "three_quarter_left",
     "passed": true, "fit_iou": 0.94, "metrics": {"lpips_masked": 0.11, ...},
     "unverified_surface_share": 0.18, "wall_s": 412.0, "gpu_s": 380.0,
     "provenance": {...}}

Output: ``eval/report/results.md``. Every number in it is an aggregate of those files; if
there are none, the script says so and writes nothing.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "results.md"

# For each metric, which direction is worse; drives the "worst" column.
HIGHER_IS_BETTER = {"silhouette_iou": True, "dino_similarity": True, "edge_agreement": True}


def load(results_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.rglob("*.json"))]


def summarise(rows: list[dict[str, Any]]) -> str:
    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    lines = ["# Results", "", f"{len(rows)} scored results from `eval/results/`.", ""]
    lines += [
        "| Condition | Products | Pass rate | Median unverified surface | Median time (s) |",
        "|---|---|---|---|---|",
    ]
    for cond in sorted(by_cond):
        rs = by_cond[cond]
        scored = [r for r in rs if r.get("passed") is not None]
        rate = f"{sum(r['passed'] for r in scored) / len(scored):.0%}" if scored else "n/a"
        unv = [
            r["unverified_surface_share"]
            for r in rs
            if r.get("unverified_surface_share") is not None
        ]
        wall = [r["wall_s"] for r in rs if r.get("wall_s") is not None]
        lines.append(
            f"| {cond} | {len({r['product'] for r in rs})} | {rate} | "
            f"{f'{statistics.median(unv):.0%}' if unv else 'n/a'} | "
            f"{f'{statistics.median(wall):.0f}' if wall else 'n/a'} |"
        )

    metrics = sorted({m for r in rows for m in r.get("metrics", {})})
    lines += ["", "| Metric | " + " | ".join(f"{c} median / worst" for c in sorted(by_cond)) + " |"]
    lines.append("|---|" + "---|" * len(by_cond))
    for m in metrics:
        cells = []
        for cond in sorted(by_cond):
            vals = [r["metrics"][m] for r in by_cond[cond] if m in r.get("metrics", {})]
            if not vals:
                cells.append("n/a")
                continue
            worst = min(vals) if HIGHER_IS_BETTER.get(m, False) else max(vals)
            cells.append(f"{statistics.median(vals):.3f} / {worst:.3f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = load(RESULTS)
    if not rows:
        print(f"No stored results under {RESULTS}; nothing to report.", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(summarise(rows), encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
