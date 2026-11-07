"""Orbit priors for each photo view label (PRD 5.1). Azimuth convention: see camera.py."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewPrior:
    azimuth_deg: float
    elevation_deg: float
    wide_search: bool = False  # no reliable prior: search the full orbit


VIEW_PRIORS: dict[str, ViewPrior] = {
    "front": ViewPrior(0.0, 10.0),
    "three_quarter_left": ViewPrior(45.0, 15.0),
    "left": ViewPrior(90.0, 10.0),
    "back": ViewPrior(180.0, 10.0),
    "right": ViewPrior(-90.0, 10.0),
    "three_quarter_right": ViewPrior(-45.0, 15.0),
    "top": ViewPrior(0.0, 75.0),
    "close_up": ViewPrior(0.0, 10.0, wide_search=True),
}


def prior_for(label: str | None) -> ViewPrior:
    if label is None or label not in VIEW_PRIORS:
        return ViewPrior(0.0, 15.0, wide_search=True)
    return VIEW_PRIORS[label]
