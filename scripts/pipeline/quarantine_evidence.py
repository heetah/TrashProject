"""Reason classification for vehicle-contained litter quarantine.

The tracker owns carrier lookup and state mutation.  This module owns only the
unit-aware release-evidence decision so sampled failures can be grouped without
changing thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuarantineReleaseDecision:
    release_ready: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "release_ready": bool(self.release_ready),
            "reason": str(self.reason),
        }


def classify_quarantine_release(
    *,
    observations: int,
    required_observations: int,
    relative_displacement: float,
    required_displacement: float,
    relative_downward: float,
    required_downward: float,
    downward_steps: int,
    required_downward_steps: int,
) -> QuarantineReleaseDecision:
    """Classify release evidence using existing inclusive thresholds.

    Units: observation values are counts; displacement values are image pixels;
    downward steps are count of consecutive downward transitions.  First-failing
    reason is deterministic and does not represent a probability.
    """
    if int(observations) < int(required_observations):
        return QuarantineReleaseDecision(False, "insufficient_observations")
    if float(relative_displacement) < float(required_displacement):
        return QuarantineReleaseDecision(
            False, "relative_displacement_insufficient"
        )
    if float(relative_downward) < float(required_downward):
        return QuarantineReleaseDecision(False, "relative_downward_insufficient")
    if int(downward_steps) < int(required_downward_steps):
        return QuarantineReleaseDecision(False, "downward_steps_insufficient")
    return QuarantineReleaseDecision(True, "released")


__all__ = ["QuarantineReleaseDecision", "classify_quarantine_release"]
