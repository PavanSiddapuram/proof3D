"""Check any 3D model against product photos, on CPU:

    uv run python eval/scripts/check_model.py model.glb \\
        --photo photos/front.png:front --photo photos/left.png:left \\
        --held-out photos/back.png:back --out .data/checks/sneaker-versa.json

* ``model.glb``: any generator's output (VersaAI, Tripo, Rodin, ours).
* ``--photo``: a photo the generator was given. ``--held-out``: one it never saw; only these
  can show the model is accurate rather than just consistent with its inputs.
* Photos must be background-removed PNGs: transparency marks the product. Ingest will do this
  automatically once it is built; until then use any background remover.
* ``:label`` (front, back, left, right, three_quarter_left, three_quarter_right, top) narrows
  the camera search. Leave it off, or pass ``--no-priors``, if the model's front is not +z.

Prints a summary and, with ``--out``, writes the full report as JSON. Write it under .data/
(git-ignored), not eval/results/, which holds only the evaluation runs ``make report`` reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from proof3d_geometry import PhotoInput, check_model

VIEW_LABELS = {
    "front",
    "back",
    "left",
    "right",
    "three_quarter_left",
    "three_quarter_right",
    "top",
    "close_up",
}


def parse_photo(spec: str, held_out: bool, use_priors: bool) -> PhotoInput:
    path, label = spec, None
    head, sep, tail = spec.rpartition(":")
    if sep and tail in VIEW_LABELS:  # rpartition, so Windows drive letters survive
        path, label = head, tail
    with Image.open(path) as img:
        if "A" not in img.getbands():
            raise SystemExit(f"{path}: needs a background-removed PNG with transparency")
        alpha = np.asarray(img.getchannel("A"))
    mask = alpha > 127
    if not mask.any():
        raise SystemExit(f"{path}: the transparency mask is empty")
    return PhotoInput(Path(path).name, mask, label if use_priors else None, held_out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", help="GLB, OBJ or PLY file")
    parser.add_argument("--photo", action="append", default=[], help="PATH[:label]")
    parser.add_argument("--held-out", action="append", default=[], help="PATH[:label]")
    parser.add_argument("--no-priors", action="store_true", help="ignore view labels")
    parser.add_argument("--out", type=Path, help="write the full report as JSON")
    args = parser.parse_args(argv)

    photos = [parse_photo(p, False, not args.no_priors) for p in args.photo]
    photos += [parse_photo(p, True, not args.no_priors) for p in args.held_out]
    if not photos:
        parser.error("give at least one --photo or --held-out")
    mesh = trimesh.load(args.model, force="mesh")
    report = check_model(np.asarray(mesh.vertices), np.asarray(mesh.faces), photos)

    print(f"Model: {args.model} ({report.faces_used} faces checked)")
    for p in report.photos:
        role = "held-out" if p.held_out else "input   "
        flags = (
            (" DISAGREES" if p.held_out else " poor fit")
            if p.low_iou
            else (" ambiguous" if p.ambiguous else "")
        )
        print(
            f"  {role} {p.name:<28} silhouette IoU {p.silhouette_iou:.3f}  "
            f"shows {p.surface_shown:5.1%} of surface{flags}"
        )
    if report.held_out_disagreements:
        print(f"Result: DISAGREES with held-out {', '.join(report.held_out_disagreements)}")
    elif report.checkable_share > 0:
        print("Result: silhouette agrees with every held-out photo (appearance not scored yet)")
    print(f"Checkable by held-out photos: {report.checkable_share:.1%} of the surface")
    print(f"Shown by any photo:           {report.evidenced_share:.1%}")
    print(f"Shown by no photo (invented): {report.invented_share:.1%}")
    for s in report.suggestions:
        print(f"  Next photo: {s.view} -> shows {s.gain:.1%} more ({s.covered_after:.1%} total)")
    for note in report.notes:
        print(f"Note: {note}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
