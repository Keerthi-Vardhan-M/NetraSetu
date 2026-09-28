"""Transparent, versioned referral rules for the screening prototype."""

from __future__ import annotations

RULE_VERSION = "hackathon-rules-v1"

EMERGENCY_SYMPTOMS = {
    "Sudden vision loss",
    "Curtain-like shadow",
    "New flashes or many floaters",
}


def determine_triage(
    right_grade: int | None,
    left_grade: int | None,
    symptoms: list[str],
    *,
    ungradable: bool = False,
    low_confidence: bool = False,
) -> dict:
    """Return an operational priority without changing the retinal grade."""

    reasons: list[str] = []
    if set(symptoms) & EMERGENCY_SYMPTOMS:
        reasons.append("Emergency visual symptom reported")
        return _result("URGENT", 1, reasons)

    grades = [grade for grade in (right_grade, left_grade) if grade is not None]
    max_grade = max(grades) if grades else None

    if max_grade == 4:
        reasons.append("Proliferative DR screening grade in at least one eye")
        return _result("URGENT", 1, reasons)
    if max_grade == 3:
        reasons.append("Severe NPDR screening grade in at least one eye")
        return _result("PRIORITY", 2, reasons)
    if ungradable:
        reasons.append("At least one image is ungradable")
        return _result("MANUAL REVIEW", 3, reasons)
    if low_confidence:
        reasons.append("AI result is unavailable or below the confidence threshold")
        return _result("MANUAL REVIEW", 3, reasons)
    if max_grade == 2:
        reasons.append("Moderate NPDR screening grade in at least one eye")
        return _result("REFERRAL", 4, reasons)
    if max_grade == 1:
        reasons.append("Mild NPDR screening grade")
        return _result("ROUTINE", 5, reasons)
    if max_grade == 0:
        reasons.append("No apparent DR detected by screening model")
        return _result("ROUTINE", 6, reasons)

    reasons.append("No reliable grade available")
    return _result("MANUAL REVIEW", 3, reasons)


def _result(label: str, rank: int, reasons: list[str]) -> dict:
    return {
        "priority": label,
        "priority_rank": rank,
        "reasons": reasons,
        "rule_version": RULE_VERSION,
    }

