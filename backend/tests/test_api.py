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

# Force demo mode for all tests, never make paid API calls
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
    """Querying 'Priya Sharma' returns a retrieve_report intent and resolves to
    that exact patient, so 'Priyanka Sharma' never escalates it to ambiguity."""
    resp = client.post("/query", json={"text": "Open Priya Sharma's latest OPG report"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "retrieve_report"
    assert data["ambiguous"] is False
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


def test_exact_first_name_resolves_to_priya_sharma():
    """Regression: 'Priya' is an exact first name for P001 and only a prefix of
    'Priyanka', so every phrasing of it must resolve to Priya Sharma, never
    surface Priyanka Sharma as a competing candidate."""
    for text in ("Priya", "Priya's report", "Open Priya's latest OPG report",
                 "open priya", "open priya's opg"):
        data = client.post("/query", json={"text": text}).json()
        assert data["intent"] == "retrieve_report", text
        assert data["ambiguous"] is False, text
        assert data["report"] is not None, text
        assert data["report"]["patient_id"] == "P001", text
        assert [p["id"] for p in data["matched_patients"]] == ["P001"], text
        assert all("Priyanka" not in p["name"] for p in data["matched_patients"]), text


def test_priyanka_query_resolves_to_priyanka_sharma():
    """A Priyanka-only query resolves to P002 (never P001): exact first name
    wins even though her name also starts with 'Priya'."""
    for text in ("Priyanka", "Priyanka's report", "open priyanka's latest opg report"):
        data = client.post("/query", json={"text": text}).json()
        assert data["intent"] == "retrieve_report", text
        assert data["ambiguous"] is False, text
        assert data["report"] is not None, text
        assert data["report"]["patient_id"] == "P002", text
        assert [p["id"] for p in data["matched_patients"]] == ["P002"], text


def test_gemini_extracted_first_name_resolves(monkeypatch):
    """The AI-extracted name is the search intent: whatever Gemini extracts as
    'Priya' is resolved against the database (exact first name) instead of
    being re-interpreted by the fuzzy scorer."""
    llm_service = _force_live_mode(monkeypatch)
    monkeypatch.setattr(
        llm_service, "_call_llm",
        lambda *a, **k: json.dumps({
            "intent": "retrieve_report",
            "patient_name": "Priya",
            "report_type": "OPG",
        }),
    )
    data = client.post("/query", json={"text": "open priya's latest opg report"}).json()
    assert data["patient_name"] == "Priya"
    assert data["ambiguous"] is False
    assert data["report"] is not None
    assert data["report"]["patient_id"] == "P001"
    assert [p["id"] for p in data["matched_patients"]] == ["P001"]


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

def test_priya_resolves_despite_similar_priyanka():
    """Regression: 'Priya' must resolve to Priya Sharma, since no other patient has
    the exact first name Priya, so no choice is asked for."""
    resp = client.post("/query", json={"text": "Open Priya's OPG report"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "retrieve_report"
    assert data["ambiguous"] is False
    assert [p["id"] for p in data["matched_patients"]] == ["P001"]
    assert data["report"] is not None
    assert data["report"]["patient_id"] == "P001"
    assert data["report"]["type"] == "OPG"


def test_shared_last_name_is_genuine_ambiguity():
    """"Sharma" matches two patients at the SAME priority (exact last name) →
    the user is asked to choose and no report is silently picked."""
    resp = client.post("/query", json={"text": "Open Sharma's OPG"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "retrieve_report"
    assert data["ambiguous"] is True
    assert data["report"] is None
    ids = [p["id"] for p in data["matched_patients"]]
    assert ids[:2] == ["P001", "P002"]
    assert "select the correct patient" in data["message"].lower()


def test_two_patients_sharing_a_first_name_asks_the_user(monkeypatch):
    """Genuine ambiguity: when two patients share the exact same first name,
    neither wins, so the user must choose."""
    from services import retrieval

    monkeypatch.setattr(retrieval, "_load_patients", lambda: [
        {"id": "P101", "name": "Priya Sharma", "dob": "1990-01-01", "gender": "F"},
        {"id": "P102", "name": "Priya Singh", "dob": "1991-02-02", "gender": "F"},
    ])
    data = client.post("/query", json={"text": "open Priya's OPG report"}).json()
    assert data["ambiguous"] is True
    assert data["report"] is None
    assert [p["id"] for p in data["matched_patients"]] == ["P101", "P102"]


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
    resp = client.post("/query", json={"text": "Open Sharma's reports"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ambiguous"] is True
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
    # The word "abscess" was not in the transcript, so it should not appear
    assert "abscess" not in draft
    # Uncertainties must exist in demo mode
    assert isinstance(data["uncertainties"], list)
    # A genuine clinical transcript is still structured into a draft
    assert data["usable"] is True
    assert data["draft_report"].strip()


def test_dictation_requires_review_always():
    """requires_review must always be True regardless of input."""
    resp = client.post("/dictation", json={"transcript": "No significant findings noted."})
    assert resp.status_code == 200
    assert resp.json()["requires_review"] is True


def test_dictation_empty_transcript_rejected():
    """Empty transcript must be rejected with a 422."""
    resp = client.post("/dictation", json={"transcript": "   "})
    assert resp.status_code == 422


def test_dictation_useless_input_gets_no_draft_and_no_save():
    """Reported bug: noise such as 'what the dark report for 2:36@' used to be
    structured into an editable draft that could be saved. It must now come
    back unusable: no findings, no draft, so nothing can be filed."""
    resp = client.post("/dictation", json={"transcript": "what the dark report for 2:36@"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["usable"] is False
    assert data["findings"] == []
    assert data["draft_report"] == ""
    assert data["requires_review"] is True
    assert data["fallback_reason"]            # reason surfaced for the UI chip
    assert any("clinical" in u.lower() for u in data["uncertainties"])
    # The transcript itself is still echoed, so the clinician can retry
    assert data["transcript"] == "what the dark report for 2:36@"


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
    """Patient list must include every synthetic patient."""
    resp = client.get("/patients")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 6
    names = [p["name"] for p in data]
    assert "Priya Sharma" in names
    assert "Priyanka Sharma" in names


def test_patient_reports_list():
    """Each patient must have at least 2 reports."""
    for pid in ["P001", "P002", "P003", "P004", "P005"]:
        resp = client.get(f"/patients/{pid}/reports")
        assert resp.status_code == 200
        assert len(resp.json()) >= 2


def test_patient_summary_report_count():
    """Patient summaries must carry their on-file report count for the sidebar."""
    data = client.get("/patients").json()
    assert len(data) == 6
    for patient in data:
        listed = client.get(f"/patients/{patient['id']}/reports").json()
        assert patient["report_count"] == len(listed)


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
    assert data["usable"] is False   # nothing clinical → no savable draft


# Real follow-up OPG dictation: header-like opening line, prior-study findings,
# and a current exam that explicitly reports NO new findings.
_FOLLOWUP_OPG_TRANSCRIPT = (
    "We have an existing patient, Arjun Mehta, patient ID P003, date of birth "
    "November 8, 1975. The examination date is October 15, 2026.\n\n"
    "This is a follow-up orthopantomogram for comparison with the patient's "
    "previous OPG dated September 20, 2025, and CBCT dated July 11, 2025.\n\n"
    "The previous OPG documented generalized moderate horizontal alveolar bone "
    "loss consistent with chronic periodontitis, heavy calculus deposits in the "
    "lower anterior region, and a fully erupted, apparently normal lower right "
    "third molar, tooth 48.\n\n"
    "The previous CBCT documented mild mucosal thickening in the right maxillary "
    "sinus and adequate bone volume at the upper left first molar, tooth 26, "
    "implant site.\n\n"
    "Please generate a structured follow-up OPG report. Separate historical "
    "findings from current observations. No new radiographic findings have been "
    "provided, so do not claim that any condition has improved, worsened, or "
    "remained unchanged. Do not invent measurements, diagnoses, or treatment "
    "recommendations. Clearly indicate that current findings require radiographic "
    "assessment and clinician review."
)


def test_followup_opg_dictation_is_structured_not_metadata_only(monkeypatch):
    """Regression (reported bug): the follow-up OPG dictation must come back
    structured. Even in the worst case (the model answering with an empty
    findings list on BOTH attempts) the response must never claim the input
    was 'only header/metadata lines' or show an empty findings panel."""
    llm_service = _force_live_mode(monkeypatch)
    monkeypatch.setattr(
        llm_service,
        "_call_llm",
        lambda *a, **k: json.dumps({
            "findings": [],
            "anatomical_site": "",
            "draft_report": "CLINICAL DICTATION DRAFT",
            "uncertainties": [],
            "requires_review": True,
        }),
    )

    resp = client.post("/dictation", json={"transcript": _FOLLOWUP_OPG_TRANSCRIPT})
    assert resp.status_code == 200
    data = resp.json()

    # The narrative survived the metadata split and reached the structurer
    assert "chronic periodontitis" in json.dumps(data["findings"]).lower()
    assert data["anatomical_site"] != "No clinical findings detected"
    uncertainties = " ".join(data["uncertainties"]).lower()
    assert "only header/metadata lines provided" not in uncertainties
    assert "header/metadata" not in (data["fallback_reason"] or "").lower()
    assert data["requires_review"] is True
    assert "CLINICAL DICTATION DRAFT" in data["draft_report"]
    assert data["transcript"].startswith("We have an existing patient, Arjun Mehta")


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
    """"Are you Gemini?" → the reply describes Viscan AI and its capabilities,
    and never names the underlying AI provider."""
    data = client.post("/query", json={"text": "Are you Gemini?"}).json()
    assert data["intent"] == "assistant_identity"
    msg = data["message"].lower()
    assert "viscan" in msg
    assert "gemini" not in msg      # never names the underlying provider
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
    patient data, and answered with a clear 'no such patient' message, a
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
    greeting because Gemini said so, and the chat request really hit the LLM."""
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
    data = client.post("/query", json={"text": "just curious, which model am i talking to?"}).json()
    assert data["intent"] == "assistant_identity"
    assert "gemini" not in data["message"].lower()


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
    """When Gemini fails, the user sees a clear outage message, not 'you typed
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


# ── Patient info (contact / demographic lookup) ─────────────────────────────

def test_contact_details_of_open_report_patient():
    """"Contact details of this person" with an open report resolves to that
    report's patient and returns the real phone/email on file."""
    resp = client.post("/query", json={
        "text": "give me contact details of this person",
        "session_context": {"active_report_id": "R001"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "patient_info"
    assert "Priya Sharma" in data["message"]
    assert "+91-98765-43210" in data["message"]
    assert "priya.sharma@example.com" in data["message"]


def test_contact_details_by_explicit_name():
    """A named contact request resolves without needing an open report."""
    resp = client.post("/query", json={"text": "what is Rohan Verma's phone number?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "patient_info"
    assert "+91-77700-33333" in data["message"]


def test_contact_details_unknown_patient_never_invented():
    """An unknown name must not yield fabricated contact info."""
    resp = client.post("/query", json={"text": "contact details for ZZZ Nowhere"})
    data = resp.json()
    assert data["intent"] == "patient_info"
    assert "@example.com" not in data["message"]


def test_contact_details_no_context_no_name_guides_user():
    """With no open report and no name, ask the user rather than guessing."""
    resp = client.post("/query", json={"text": "give me the contact details"})
    data = resp.json()
    assert data["intent"] == "patient_info"
    assert "open" in data["message"].lower()


def test_contact_details_demo_mode_no_llm():
    """Demo mode answers contact requests from real data with no LLM call."""
    import services.llm_service as llm_service
    _orig = llm_service.DEMO_MODE
    llm_service.DEMO_MODE = True
    try:
        resp = client.post("/query", json={"text": "contact details for Priya Sharma"})
    finally:
        llm_service.DEMO_MODE = _orig
    data = resp.json()
    assert data["intent"] == "patient_info"
    assert "priya.sharma@example.com" in data["message"]


# ── info_fields arrive in the USER'S wording in live mode ────────────────────
# The interpreter prompt tells the LLM to echo the user's own phrasing, so the
# formatter must map free-text labels onto schema fields instead of treating an
# unfamiliar label as "not recorded" (a false denial of stored data).

def test_live_phone_number_wording_returns_stored_phone(monkeypatch):
    """info_fields ["phone number"] must show the real stored phone, not claim
    the phone is unrecorded."""
    llm_service = _force_live_mode(monkeypatch)
    monkeypatch.setattr(
        llm_service, "_call_llm",
        lambda *a, **k: json.dumps({
            "intent": "patient_info",
            "patient_name": "Priya Sharma",
            "info_fields": ["phone number"],
        }),
    )
    data = client.post("/query", json={"text": "what is Priya Sharma's phone number?"}).json()
    assert data["intent"] == "patient_info"
    assert "+91-98765-43210" in data["message"]
    assert "no phone number recorded" not in data["message"].lower()
    assert "won't guess" not in data["message"].lower()


def test_live_age_wording_derives_from_dob(monkeypatch):
    """info_fields ["age"] is answerable from the stored dob, never 'not recorded'."""
    llm_service = _force_live_mode(monkeypatch)
    monkeypatch.setattr(
        llm_service, "_call_llm",
        lambda *a, **k: json.dumps({
            "intent": "patient_info",
            "patient_name": "Priya Sharma",
            "info_fields": ["age"],
        }),
    )
    data = client.post("/query", json={"text": "how old is Priya Sharma?"}).json()
    assert data["intent"] == "patient_info"
    assert "Age:" in data["message"]
    assert "not recorded" not in data["message"].lower()


def test_live_unstored_detail_still_answered_honestly(monkeypatch):
    """A detail the schema genuinely never stores is still reported missing —
    normalisation must not start inventing values for it."""
    llm_service = _force_live_mode(monkeypatch)
    monkeypatch.setattr(
        llm_service, "_call_llm",
        lambda *a, **k: json.dumps({
            "intent": "patient_info",
            "patient_name": "Priya Sharma",
            "info_fields": ["blood group"],
        }),
    )
    data = client.post("/query", json={"text": "what is Priya Sharma's blood group?"}).json()
    assert data["intent"] == "patient_info"
    assert "no blood group recorded" in data["message"].lower()
    assert "+91-98765-43210" not in data["message"]   # never substitutes another detail


def test_live_mixed_known_and_unknown_fields(monkeypatch):
    """A known label and an unstored one in the same answer: show the stored
    value and name the missing one — never drop either."""
    llm_service = _force_live_mode(monkeypatch)
    monkeypatch.setattr(
        llm_service, "_call_llm",
        lambda *a, **k: json.dumps({
            "intent": "patient_info",
            "patient_name": "Priya Sharma",
            "info_fields": ["e-mail address", "allergies"],
        }),
    )
    data = client.post("/query", json={"text": "Priya Sharma email and allergies"}).json()
    assert data["intent"] == "patient_info"
    assert "priya.sharma@example.com" in data["message"]
    assert "no allergies recorded" in data["message"].lower()


def test_format_patient_info_normalizes_and_dedupes():
    """Direct unit check: synonym labels collapse onto one canonical field so a
    stored phone is shown exactly once."""
    import main as main_module
    patient = {"id": "P001", "name": "Priya Sharma", "dob": "1988-03-15",
               "gender": "F", "phone": "+91-98765-43210",
               "email": "priya.sharma@example.com"}
    out = main_module._format_patient_info(
        patient, ["phone number", "Phone", "CONTACT DETAILS"]
    )
    assert out.count("+91-98765-43210") == 1
    assert "has no" not in out.lower()           # answered from the record, not a "missing" fallback
    assert "won't guess" not in out
    # unknown wording passes through as an honest "not recorded"
    out2 = main_module._format_patient_info(patient, ["phone number", "ziggo"])
    assert "ziggo" in out2 and "won't guess" in out2
