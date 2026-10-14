"""
Tests for Viscan AI backend.
All tests run WITHOUT paid API calls (demo mode via DEMO_MODE=True).
"""
from __future__ import annotations
import sys
import os
import pytest
from pathlib import Path

# Ensure backend is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force demo mode for all tests — never make paid API calls
os.environ["OPENAI_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# ── Test 1: Correct report retrieval ────────────────────────────────────────

def test_retrieve_report_by_patient_id():
    """Direct patient ID lookup + report fetch is always deterministic."""
    resp = client.get("/patients/P001/reports")
    assert resp.status_code == 200
    data = resp.json()
    opg = [r for r in data if r["type"] == "OPG"]
    assert len(opg) >= 1
    assert opg[0]["patient_id"] == "P001"


def test_retrieve_specific_report_by_id():
    """Fetching report R001 by ID always returns Priya Sharma's OPG."""
    resp = client.get("/reports/R001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "OPG"
    assert data["patient_id"] == "P001"
    assert data["id"] == "R001"


def test_retrieve_report_full_name_resolves():
    """Querying 'Priya Sharma' returns a retrieve_report intent.
    Because the DB has 'Priyanka Sharma' too, the system may surface ambiguity
    for patient safety — but must not silently pick the wrong patient."""
    resp = client.post("/query", json={"text": "Open Priya Sharma's latest OPG report"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "retrieve_report"
    # Either: unambiguous match to P001, OR: correct ambiguity surfaced
    if data["ambiguous"]:
        # Safety: both Sharmas should be candidates
        names = [p["name"] for p in data["matched_patients"]]
        assert any("Priya" in n for n in names)
        # No report should be returned when ambiguous
        assert data["report"] is None
    else:
        # If unambiguous, it must be the correct patient
        assert data["report"] is not None
        assert data["report"]["patient_id"] == "P001"
        assert data["report"]["type"] == "OPG"


def test_exact_full_name_resolves_unambiguously():
    """An exact full-name match must resolve to that exact patient (P001),
    not trigger ambiguity with the similar name 'Priyanka Sharma'."""
    resp = client.post("/query", json={"text": "Open Priya Sharma's latest OPG report"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ambiguous"] is False
    assert data["report"] is not None
    assert data["report"]["patient_id"] == "P001"
    assert data["report"]["type"] == "OPG"
    assert [p["id"] for p in data["matched_patients"]] == ["P001"]


def test_partial_name_query_is_ambiguous():
    """A partial first name matching two patients must ask for clarification
    and must not return a report — including fully-lowercase input."""
    resp = client.post("/query", json={"text": "open priya's opg"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "retrieve_report"
    assert data["ambiguous"] is True
    assert data["report"] is None
    ids = [p["id"] for p in data["matched_patients"]]
    assert "P001" in ids and "P002" in ids


def test_retrieve_report_rohan_bitewing():
    """Rohan Verma's latest bitewing should return R010 (no similar-name conflict)."""
    resp = client.post("/query", json={"text": "Show Rohan Verma's bitewing report"})
    assert resp.status_code == 200
    data = resp.json()
    # Rohan Verma has a unique enough name - no ambiguity expected
    assert data["ambiguous"] is False
    assert data["report"] is not None
    assert data["report"]["type"] == "BITEWING"
    assert data["report"]["patient_id"] == "P005"


def test_retrieve_report_nonexistent_patient():
    """Querying an unknown name should return no report and a helpful message."""
    resp = client.post("/query", json={"text": "Open John Smith's OPG"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["report"] is None
    assert "not found" in data["message"].lower() or "no patient" in data["message"].lower() or "match" in data["message"].lower()


# ── Test 2: Ambiguous patient identity ──────────────────────────────────────

def test_ambiguous_patient_priya():
    """'Priya' alone must surface both Priya Sharma and Priyanka Sharma as candidates."""
    resp = client.post("/query", json={"text": "Open Priya's OPG report"})
    assert resp.status_code == 200
    data = resp.json()
    # Must be ambiguous — Priya matches both
    assert data["ambiguous"] is True
    assert len(data["matched_patients"]) >= 2
    names = [p["name"] for p in data["matched_patients"]]
    assert any("Priya Sharma" in n for n in names)
    assert any("Priyanka" in n for n in names)
    # Must NOT silently return a report for one of them
    assert data["report"] is None


def test_ambiguous_explicit_sharma():
    """Searching 'Sharma' (last name only) must return multiple candidates."""
    from services.retrieval import search_patients_by_name
    candidates = search_patients_by_name("Sharma")
    assert len(candidates) >= 2
    names = [c["name"] for c in candidates]
    assert any("Priya Sharma" in n for n in names)
    assert any("Priyanka Sharma" in n for n in names)


def test_no_silent_wrong_patient_selection():
    """When ambiguous, report must be None (no silent selection)."""
    resp = client.post("/query", json={"text": "Open Priya's reports"})
    assert resp.status_code == 200
    data = resp.json()
    if data["ambiguous"]:
        assert data["report"] is None


# ── Test 3: Unsupported clinical claims in dictation ────────────────────────

def test_dictation_does_not_invent_findings():
    """Demo mode dictation must not add clinical findings beyond the input."""
    transcript = "Patient presents with mild sensitivity in lower left region."
    resp = client.post("/dictation", json={"transcript": transcript})
    assert resp.status_code == 200
    data = resp.json()
    # requires_review must always be True
    assert data["requires_review"] is True
    # Draft report must not contain fabricated diagnoses
    draft = data["draft_report"].lower()
    # The word "abscess" was not in the transcript — should not appear
    assert "abscess" not in draft
    # Uncertainties must exist in demo mode
    assert isinstance(data["uncertainties"], list)


def test_dictation_requires_review_always():
    """requires_review must always be True regardless of input."""
    resp = client.post("/dictation", json={"transcript": "No significant findings noted."})
    assert resp.status_code == 200
    assert resp.json()["requires_review"] is True


def test_dictation_empty_transcript_rejected():
    """Empty transcript must be rejected with a 422."""
    resp = client.post("/dictation", json={"transcript": "   "})
    assert resp.status_code == 422


# ── Test 4: API health & demo mode ──────────────────────────────────────────

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["demo_mode"] is True  # no API key set


def test_health_reports_llm_provider():
    """In tests no key is configured → provider must report None."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["demo_mode"] is True
    assert body["llm_provider"] is None


def test_follow_up_invalid_report():
    """Follow-up with unknown report_id must return 404."""
    resp = client.post("/follow-up", json={"question": "Summarise it", "report_id": "DOES_NOT_EXIST"})
    assert resp.status_code == 404


def test_follow_up_demo_mode():
    """Follow-up in demo mode must return an answer with evidence."""
    resp = client.post("/follow-up", json={"question": "What is the impression?", "report_id": "R001"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"]
    assert isinstance(data["evidence"], list)
    assert data["demo_mode"] is True


def test_follow_up_intent_uses_session_context():
    """"Summarize her report" with an open report resolves to that report so the
    frontend can call /follow-up with the right report_id (contextual follow-up)."""
    resp = client.post("/query", json={
        "text": "Summarize her report",
        "session_context": {"active_report_id": "R001"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "follow_up"
    assert data["report"] is not None
    assert data["report"]["id"] == "R001"

    # …and the report-scoped endpoint answers with evidence from that report
    fu = client.post("/follow-up", json={"question": "Summarize her report", "report_id": "R001"})
    assert fu.status_code == 200
    body = fu.json()
    assert body["answer"]
    assert body["source_report_id"] == "R001"
    assert len(body["evidence"]) > 0


def test_follow_up_without_context_guides_user():
    """"Summarize her report" with no open report must not invent context."""
    resp = client.post("/query", json={"text": "Summarize her report"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "follow_up"
    assert data["report"] is None
    assert "open a report" in data["message"].lower()


def test_follow_up_stale_session_report():
    """A session pointing at a deleted/unknown report fails safely."""
    resp = client.post("/query", json={
        "text": "Summarize her report",
        "session_context": {"active_report_id": "R999"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["report"] is None
    assert "not found" in data["message"].lower()


def test_list_patients():
    """Patient list must include all 5 synthetic patients."""
    resp = client.get("/patients")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 5
    names = [p["name"] for p in data]
    assert "Priya Sharma" in names
    assert "Priyanka Sharma" in names


def test_patient_reports_list():
    """Each patient must have at least 2 reports."""
    for pid in ["P001", "P002", "P003", "P004", "P005"]:
        resp = client.get(f"/patients/{pid}/reports")
        assert resp.status_code == 200
        assert len(resp.json()) >= 2
