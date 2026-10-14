from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from difflib import SequenceMatcher

DATA_DIR = Path(__file__).parent.parent / "data"


def _load_patients() -> list[dict]:
    with open(DATA_DIR / "patients.json", encoding="utf-8") as f:
        return json.load(f)


def _load_reports() -> list[dict]:
    with open(DATA_DIR / "reports.json", encoding="utf-8") as f:
        return json.load(f)


def _token_similarity(query: str, name: str) -> float:
    """
    Token-aware name similarity.
    - Exact full-name match → 1.0
    - Token-prefix matching for partial names
    - Blend with SequenceMatcher for robustness
    """
    q_lower = query.lower().strip()
    n_lower = name.lower().strip()

    # Exact match shortcut
    if q_lower == n_lower:
        return 1.0

    q_tokens = q_lower.split()
    n_tokens = n_lower.split()

    # Count how many query tokens are prefixes of any name token
    prefix_hits = 0
    for qt in q_tokens:
        if any(nt.startswith(qt) for nt in n_tokens):
            prefix_hits += 1

    prefix_score = prefix_hits / max(len(q_tokens), 1)

    # Full string similarity
    seq_score = SequenceMatcher(None, q_lower, n_lower).ratio()

    # Weighted blend: prefix matching dominates for partial queries
    return 0.65 * prefix_score + 0.35 * seq_score


def search_patients_by_name(name: str) -> list[dict]:
    """
    Return patients whose name fuzzy-matches the query.
    Always includes an exact (case-insensitive) full-name match at score 1.0,
    flagged with ``_exact = True`` so the caller can treat a full-name match as
    definitive (exact full names must resolve; partial names fall back to the
    gap-based ambiguity rule).

    The ambiguity decision is made by the caller, not here.
    Threshold for inclusion: 0.45
    """
    patients = _load_patients()
    threshold = 0.45
    normalised_query = " ".join(name.split()).casefold()
    matches = []
    for p in patients:
        sim = _token_similarity(name, p["name"])
        if sim >= threshold:
            is_exact = " ".join(p["name"].split()).casefold() == normalised_query
            matches.append({**p, "_score": round(sim, 3), "_exact": is_exact})
    matches.sort(key=lambda x: x["_score"], reverse=True)
    return matches


def get_patient_by_id(patient_id: str) -> Optional[dict]:
    for p in _load_patients():
        if p["id"] == patient_id:
            return p
    return None


def get_reports_for_patient(patient_id: str, report_type: Optional[str] = None) -> list[dict]:
    """Return all reports for a patient, optionally filtered by type."""
    reports = _load_reports()
    result = [r for r in reports if r["patient_id"] == patient_id]
    if report_type:
        result = [r for r in result if r["type"].upper() == report_type.upper()]
    result.sort(key=lambda r: r["date"], reverse=True)
    return result


def get_latest_report_for_patient(patient_id: str, report_type: Optional[str] = None) -> Optional[dict]:
    """Return the most recent report for a patient."""
    reports = get_reports_for_patient(patient_id, report_type)
    return reports[0] if reports else None


def get_report_by_id(report_id: str) -> Optional[dict]:
    for r in _load_reports():
        if r["id"] == report_id:
            return r
    return None
