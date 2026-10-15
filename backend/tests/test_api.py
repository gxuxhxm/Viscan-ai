"""
Tests for Viscan AI backend.
All tests run WITHOUT paid API calls (demo mode via DEMO_MODE=True).
"""
from __future__ import annotations
import sys
import os
import json
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


# ── Additive report schema (tooth / encounter / proposed billing) ───────────

def test_report_detail_includes_additive_fields():
    """Report detail must expose the new synthetic fields, labelled as proposed."""
    data = client.get("/reports/R001").json()
    assert data["tooth_number"] == "26"
    assert data["encounter_type"] == "initial"
    assert data["procedure_date"] == data["date"]
    assert data["billing_codes"]["icd_10"] == "K04.5"
    assert data["billing_codes"]["cdt_proposed"] == "D0150"
    f = data["findings"][0]
    assert f["category"] == "Periapical pathology"
    assert f["resolution_plan"]
    assert isinstance(f["complication_detected"], bool)


def test_report_summary_includes_additive_fields():
    data = client.get("/patients/P001/reports").json()
    opg = next(r for r in data if r["id"] == "R001")
    assert opg["tooth_number"] == "26"
    assert opg["encounter_type"] == "initial"
    assert opg["billing_codes"]["cdt_proposed"] == "D0150"


def test_report_with_null_tooth_and_billing():
    data = client.get("/reports/R004").json()
    assert data["tooth_number"] is None
    assert data["billing_codes"]["icd_10"] is None
    assert data["billing_codes"]["cdt_proposed"] == "D0274"


# ── Dictation transparency (metadata split, transcript echo) ────────────────

def test_dictation_transparency_fields():
    """Header lines are echoed as metadata, the original transcript is
    preserved, and the header never becomes a finding."""
    resp = client.post("/dictation", json={
        "transcript": "name: Priya Sharma\ntooth: 26\nMild bone loss noted at lower left molar.",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["transcript"].startswith("name: Priya Sharma")
    assert data["metadata"]["name"] == "Priya Sharma"
    assert data["metadata"]["tooth"] == "26"
    assert all("Priya" not in str(f.get("observation", "")) for f in data["findings"])
    assert data["requires_review"] is True
    # demo mode → structured by the deterministic parser, not a fallback
    assert data["fallback_reason"] is None


def test_dictation_metadata_only_no_findings():
    """Header-only dictation returns an explicit no-findings result."""
    resp = client.post("/dictation", json={"transcript": "name: Priya Sharma\ndob: 1990-05-14"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["findings"] == []
    assert data["fallback_reason"] is not None
    assert data["metadata"]["name"] == "Priya Sharma"
    assert data["requires_review"] is True


# ── Conversational intents: greetings, assistant identity, unsupported ───────

def test_greeting_gets_natural_response():
    """"Hi" must be answered with a friendly capability intro, not a report error."""
    resp = client.post("/query", json={"text": "hi"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "greeting"
    assert data["report"] is None
    assert data["matched_patients"] == []
    assert "Hi!" in data["message"]
    assert "retrieve reports" in data["message"]


def test_greeting_variants():
    for text in ("Hello there!", "Hey there", "good morning"):
        data = client.post("/query", json={"text": text}).json()
        assert data["intent"] == "greeting", text


def test_greeting_with_task_is_not_a_greeting():
    """"hi, open Priya Sharma's OPG report" mixes a greeting with a real request."""
    data = client.post("/query", json={"text": "hi, open Priya Sharma's OPG report"}).json()
    assert data["intent"] == "retrieve_report"
    assert data["report"] is not None
    assert data["report"]["patient_id"] == "P001"


def test_identity_question_answered_truthfully():
    """"Are you Gemini?" → honest answer: the app USES the Gemini API for its AI
    tasks; it never claims the whole app IS Gemini."""
    data = client.post("/query", json={"text": "Are you Gemini?"}).json()
    assert data["intent"] == "assistant_identity"
    msg = data["message"].lower()
    assert "gemini api" in msg      # names what powers the app's AI tasks
    assert "not gemini" in msg      # never claims to BE Gemini
    assert "viscan" in msg
    assert data["report"] is None


def test_identity_question_variants():
    for text in ("Who are you?", "Are you an AI?"):
        data = client.post("/query", json={"text": text}).json()
        assert data["intent"] == "assistant_identity", text


def test_unsupported_request_gets_guidance():
    """"Unsupported" is answered with helpful scope guidance, not a report error."""
    data = client.post("/query", json={"text": "Write me a poem about the seaside"}).json()
    assert data["intent"] in ("unknown", "unsupported")
    assert "report" in data["message"].lower()


# ── Unknown patients: interpret, search real data, never invent ──────────────

def test_rahul_report_unknown_patient():
    """"Rahul report" is interpreted as retrieval, searched against the mock
    patient data, and answered with a clear 'no such patient' message — a
    patient is never invented."""
    data = client.post("/query", json={"text": "Rahul report"}).json()
    assert data["intent"] == "retrieve_report"
    assert data["patient_name"] == "Rahul"
    assert data["report"] is None
    assert data["matched_patients"] == []
    assert "no patient" in data["message"].lower()
    assert "Rahul" in data["message"]


def test_open_rahul_unknown_patient():
    data = client.post("/query", json={"text": "open Rahul"}).json()
    assert data["intent"] == "retrieve_report"
    assert data["patient_name"] == "Rahul"
    assert data["report"] is None
    assert "no patient" in data["message"].lower()


# ── Contextual follow-ups preserve the active report ─────────────────────────

def test_contextual_summary_preserves_active_report():
    """"Summarize this report please" with an open report must resolve to that
    report, preserving the active report context."""
    resp = client.post("/query", json={
        "text": "Summarize this report please",
        "session_context": {"active_report_id": "R001"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "follow_up"
    assert data["report"] is not None
    assert data["report"]["id"] == "R001"


def test_contextual_summary_without_report_never_invents_context():
    data = client.post("/query", json={"text": "Give me a summary"}).json()
    assert data["intent"] in ("follow_up", "unknown")
    assert data["report"] is None


# ── Gemini actually classifies chat messages (no paid calls: stubbed LLM) ────

def _force_live_mode(monkeypatch):
    """Simulate a configured provider end-to-end (service + API layers)."""
    from services import llm_service
    import main as main_module
    monkeypatch.setattr(llm_service, "DEMO_MODE", False)
    monkeypatch.setattr(main_module, "DEMO_MODE", False)
    return llm_service


def test_greeting_classified_by_gemini_not_hardcoded(monkeypatch):
    """Natural greeting phrasing the deterministic parser would NOT match
    ("well hello there, how are things going today?") is classified as a
    greeting because Gemini said so — and the chat request really hit the LLM."""
    llm_service = _force_live_mode(monkeypatch)
    seen = {}

    def _fake_llm(system, user):
        seen["text"] = user
        return json.dumps({"intent": "greeting"})

    monkeypatch.setattr(llm_service, "_call_llm", _fake_llm)
    text = "well hello there, how are things going today?"
    resp = client.post("/query", json={"text": text})
    assert resp.status_code == 200
    assert seen["text"] == text          # the request actually called the LLM
    data = resp.json()
    assert data["intent"] == "greeting"
    assert "Hi!" in data["message"]
    assert data["fallback_reason"] is None


def test_identity_question_classified_by_gemini(monkeypatch):
    llm_service = _force_live_mode(monkeypatch)
    monkeypatch.setattr(
        llm_service, "_call_llm",
        lambda *a, **k: json.dumps({"intent": "assistant_identity"}),
    )
    data = client.post("/query", json={"text": "just curious — which model am i talking to?"}).json()
    assert data["intent"] == "assistant_identity"
    assert "gemini api" in data["message"].lower()
    assert "not gemini" in data["message"].lower()


def test_natural_report_request_interpreted_by_gemini(monkeypatch):
    """Colloquial phrasing without standard command verbs resolves via Gemini
    to a real report from the mock data."""
    llm_service = _force_live_mode(monkeypatch)
    monkeypatch.setattr(
        llm_service, "_call_llm",
        lambda *a, **k: json.dumps({
            "intent": "retrieve_report",
            "patient_name": "Rohan Verma",
            "report_type": "BITEWING",
        }),
    )
    data = client.post("/query", json={
        "text": "could you pull up the most recent scan for rohan verma?",
    }).json()
    assert data["intent"] == "retrieve_report"
    assert data["report"] is not None
    assert data["report"]["patient_id"] == "P005"
    assert data["report"]["type"] == "BITEWING"


def test_contextual_summary_classified_by_gemini(monkeypatch):
    llm_service = _force_live_mode(monkeypatch)
    monkeypatch.setattr(
        llm_service, "_call_llm", lambda *a, **k: json.dumps({"intent": "follow_up"})
    )
    data = client.post("/query", json={
        "text": "Give me a summary",
        "session_context": {"active_report_id": "R001"},
    }).json()
    assert data["intent"] == "follow_up"
    assert data["report"] is not None
    assert data["report"]["id"] == "R001"


def test_gemini_failure_gives_clear_fallback_message(monkeypatch):
    """When Gemini fails, the user sees a clear outage message — not 'you typed
    an unrecognised report command' wording."""
    llm_service = _force_live_mode(monkeypatch)

    def _boom(*args, **kwargs):
        raise ConnectionError("provider down")

    monkeypatch.setattr(llm_service, "_call_llm", _boom)
    data = client.post("/query", json={"text": "write me a poem about the seaside"}).json()
    assert "couldn't reach the ai service" in data["message"].lower()
    assert "didn't understand" not in data["message"].lower()
    assert data["fallback_reason"] is not None


def test_gemini_failure_still_retrieves_real_reports(monkeypatch):
    """The deterministic fallback keeps handling simple real-report commands."""
    llm_service = _force_live_mode(monkeypatch)

    def _boom(*args, **kwargs):
        raise ConnectionError("provider down")

    monkeypatch.setattr(llm_service, "_call_llm", _boom)
    data = client.post("/query", json={"text": "open priya sharma's opg"}).json()
    assert data["intent"] == "retrieve_report"
    assert data["report"] is not None
    assert data["report"]["patient_id"] == "P001"
    assert data["fallback_reason"] is not None


def test_gemini_failure_still_answers_greeting(monkeypatch):
    llm_service = _force_live_mode(monkeypatch)

    def _boom(*args, **kwargs):
        raise ConnectionError("provider down")

    monkeypatch.setattr(llm_service, "_call_llm", _boom)
    data = client.post("/query", json={"text": "hi"}).json()
    assert data["intent"] == "greeting"
    assert "Hi!" in data["message"]
