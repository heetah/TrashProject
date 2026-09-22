"""Deterministic litter-event confirmation policy.

The tracker computes geometry, motion, and quarantine evidence.  This module
only combines those already-computed booleans and emits an auditable reason;
it does not lower thresholds or infer attribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ConfirmationDecision:
    confirmed: bool
    rule: str | None
    reason: str
    evidence: Mapping[str, bool]

    def as_dict(self) -> dict:
        return {
            "confirmed": bool(self.confirmed),
            "rule": self.rule,
            "reason": str(self.reason),
            "evidence": {str(key): bool(value) for key, value in self.evidence.items()},
        }


def evaluate_litter_confirmation(
    *,
    vehicle_quarantine_active: bool,
    by_trajectory: bool,
    by_motion: bool,
    vehicle_fast_drop: bool,
    fall_then_stable: bool,
) -> ConfirmationDecision:
    """Combine existing confirmation gates without changing their thresholds.

    Rule order matches the production conditional for deterministic audit output.
    Quarantine is a hard veto even when another path reports sufficient motion.
    """
    evidence = {
        "by_trajectory": bool(by_trajectory),
        "by_motion": bool(by_motion),
        "vehicle_fast_drop": bool(vehicle_fast_drop),
        "fall_then_stable": bool(fall_then_stable),
    }
    if bool(vehicle_quarantine_active):
        return ConfirmationDecision(
            confirmed=False,
            rule=None,
            reason="vehicle_quarantine",
            evidence=evidence,
        )
    for rule_name, supported in evidence.items():
        if supported:
            return ConfirmationDecision(
                confirmed=True,
                rule=rule_name,
                reason="supported",
                evidence=evidence,
            )
    return ConfirmationDecision(
        confirmed=False,
        rule=None,
        reason="insufficient_confirmation_evidence",
        evidence=evidence,
    )


__all__ = ["ConfirmationDecision", "evaluate_litter_confirmation"]
