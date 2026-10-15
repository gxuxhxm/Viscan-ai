"""
Direct unit tests for retrieval and LLM service (no HTTP layer).
"""
from __future__ import annotations
import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
# Force demo mode for all tests — never make paid API calls
os.environ["OPENAI_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""

import json

from services.retrieval import (
    search_patients_by_name,
    get_latest_report_for_patient,
    get_report_by_id,
    get_reports_for_patient,
)
from services.llm_service import (
    _demo_interpret_query,
    _demo_structure_dictation,
    structure_dictation,
    _split_metadata,
    _is_metadata_finding,
    _describe_llm_error,
)


def test_retrieval_exact_name():
    results = search_patients_by_name("Priya Sharma")
    assert results[0]["id"] == "P001"


def test_retrieval_partial_name_ambiguous():
    results = search_patients_by_name("Priya")
    ids = [r["id"] for r in results]
    assert "P001" in ids  # Priya Sharma
    assert "P002" in ids  # Priyanka Sharma


def test_retrieval_no_match():
    results = search_patients_by_name("xyznonexistent")
    assert len(results) == 0


def test_latest_opg_report_p001():
    report = get_latest_report_for_patient("P001", "OPG")
    assert report is not None
    assert report["id"] == "R001"
    assert report["type"] == "OPG"


def test_latest_report_type_filter():
    report = get_latest_report_for_patient("P005", "BITEWING")
    assert report is not None
    assert report["type"] == "BITEWING"


def test_missing_patient_reports():
    reports = get_reports_for_patient("P999")
    assert reports == []


def test_demo_interpret_retrieve():
    result = _demo_interpret_query("Open Priya's OPG report")
    assert result.intent == "retrieve_report"


def test_demo_interpret_followup():
    result = _demo_interpret_query("Summarize her report")
    assert result.intent == "follow_up"


def test_demo_dictation_no_fabrication():
    transcript = "Mild bone loss noted at lower left molar."
    result = _demo_structure_dictation(transcript)
    assert result.requires_review is True
    # Must not add diagnoses beyond the transcript
    assert "abscess" not in result.draft_report.lower()


# ── Patient-name extraction (no command-verb swallowing) ─────────────────────

def test_name_extraction_strips_command_verbs():
    """'Open Priya…' must extract 'Priya', never 'Open Priya'."""
    from services.llm_service import _demo_interpret_query

    assert _demo_interpret_query("Open Priya's latest OPG report").patient_name == "Priya"
    assert (
        _demo_interpret_query("Open Priya Sharma's latest OPG report").patient_name
        == "Priya Sharma"
    )
    assert (
        _demo_interpret_query("Show Rohan Verma's bitewing report").patient_name
        == "Rohan Verma"
    )
    # Fully-lowercase input is also handled
    assert _demo_interpret_query("open priya's opg").patient_name == "Priya"
    # No name in a pure follow-up question
    assert _demo_interpret_query("Summarize her report").patient_name is None


def test_exact_full_name_is_flagged():
    """An exact full-name match must be flagged so the caller resolves it."""
    results = search_patients_by_name("Priya Sharma")
    assert results[0]["id"] == "P001"
    assert results[0]["_exact"] is True
    # A partial name is NOT exact → falls back to the ambiguity rule
    partial = search_patients_by_name("Priya")
    assert all(not r["_exact"] for r in partial)


# ── Provider failure / malformed output resilience ───────────────────────────

def _force_live_mode(monkeypatch):
    """Simulate live mode (DEMO_MODE=False) so _call_llm is actually used."""
    from services import llm_service
    monkeypatch.setattr(llm_service, "DEMO_MODE", False)
    return llm_service


def test_interpret_provider_error_falls_back(monkeypatch):
    """A provider exception must fall back to the deterministic interpreter."""
    llm_service = _force_live_mode(monkeypatch)

    def _boom(*args, **kwargs):
        raise ConnectionError("provider down")

    monkeypatch.setattr(llm_service, "_call_llm", _boom)
    result = llm_service.interpret_query("Open Priya Sharma's latest OPG report")
    assert result.intent == "retrieve_report"
    assert result.patient_name == "Priya Sharma"
    assert result.fallback_reason is not None
    assert "provider" in result.fallback_reason.lower()


def test_interpret_malformed_llm_output_falls_back(monkeypatch):
    """Non-JSON LLM output must fall back to the deterministic interpreter."""
    llm_service = _force_live_mode(monkeypatch)
    monkeypatch.setattr(llm_service, "_call_llm", lambda *a, **k: "not json {{")
    result = llm_service.interpret_query("Summarise her report")
    assert result.intent == "follow_up"
    assert result.fallback_reason is not None


def test_interpret_invalid_intent_falls_back(monkeypatch):
    """Valid JSON with an invalid intent must also fall back safely."""
    llm_service = _force_live_mode(monkeypatch)
    monkeypatch.setattr(
        llm_service, "_call_llm",
        lambda *a, **k: '{"intent": "hack_the_planet", "patient_name": "X"}',
    )
    result = llm_service.interpret_query("Open Priya's OPG report")
    assert result.intent in {"retrieve_report", "dictation", "follow_up", "unknown"}


def test_dictation_provider_error_falls_back(monkeypatch):
    """Provider failure during dictation must never invent findings."""
    llm_service = _force_live_mode(monkeypatch)

    def _boom(*args, **kwargs):
        raise TimeoutError("provider timeout")

    monkeypatch.setattr(llm_service, "_call_llm", _boom)
    transcript = "Mild bone loss noted at lower left molar."
    result = llm_service.structure_dictation(transcript)
    assert result.requires_review is True
    assert "abscess" not in result.draft_report.lower()
    assert transcript[:50] in result.draft_report  # transcript preserved verbatim
    assert len(result.uncertainties) > 0


def test_follow_up_provider_error_falls_back(monkeypatch):
    """Provider failure during follow-up must still return report evidence."""
    llm_service = _force_live_mode(monkeypatch)

    def _boom(*args, **kwargs):
        raise ConnectionError("provider down")

    monkeypatch.setattr(llm_service, "_call_llm", _boom)
    report = get_report_by_id("R001")
    result = llm_service.answer_follow_up("Summarise it", report)
    assert result["source_report_id"] == "R001"
    assert result["evidence"]  # evidence still comes from the deterministic record
    assert result.get("error") == "llm_provider_error"


# ── Demo dictation parser (deterministic structure, no invention) ────────────

def test_demo_dictation_structures_sentences():
    """Punctuation-separated dictation becomes one finding per sentence with
    site/severity derived only from words actually present."""
    transcript = (
        "Periapical radiolucency at tooth 26, moderate. "
        "Mild bone loss at lower left molar."
    )
    result = _demo_structure_dictation(transcript)
    assert len(result.findings) == 2
    assert result.findings[0]["severity"] == "moderate"
    assert result.findings[1]["severity"] == "mild"
    assert "lower left" in result.findings[1]["site"]
    assert "molar" in result.findings[1]["site"]
    # Observations are verbatim
    assert result.findings[1]["observation"] == "Mild bone loss at lower left molar."
    assert "tooth 26" in result.anatomical_site


def test_demo_dictation_severity_only_when_stated():
    """No severity word in the transcript → severity must be 'unspecified',
    never a guessed label."""
    result = _demo_structure_dictation("Restoration present on tooth 16.")
    assert all(f["severity"] == "unspecified" for f in result.findings)
    assert any("severity" in u.lower() for u in result.uncertainties)


def test_demo_dictation_negated_severity_ignored():
    """'no mild …' must not be labelled mild."""
    result = _demo_structure_dictation("There is no mild decay visible.")
    assert result.findings[0]["severity"] == "unspecified"


def test_demo_dictation_site_unspecified_when_absent():
    result = _demo_structure_dictation("Some vague note without a location.")
    assert "unspecified" in result.anatomical_site
    assert any("site" in u.lower() for u in result.uncertainties)


def test_demo_dictation_never_adds_clinical_terms():
    """A term absent from the transcript must never appear in the draft."""
    result = _demo_structure_dictation("Mild bone loss noted at lower left molar.")
    draft = result.draft_report.lower()
    assert "abscess" not in draft
    assert "caries" not in draft
    assert "impaction" not in draft
    assert result.requires_review is True
    assert "not finalised" in result.draft_report


# ── Gemini provider routing & output-safety guards ───────────────────────────

def test_call_llm_routes_to_gemini(monkeypatch):
    """With a Gemini key configured, _call_llm must go through Gemini."""
    from services import llm_service
    monkeypatch.setattr(llm_service, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm_service, "_call_gemini", lambda s, u: '{"ok": true}')
    assert llm_service._call_llm("sys", "user") == '{"ok": true}'


def test_parse_llm_json_tolerates_code_fences():
    """Gemini often wraps JSON in ```json fences — parsing must tolerate them."""
    from services.llm_service import _parse_llm_json
    raw = '```json\n{"intent": "follow_up", "patient_name": null}\n```'
    assert _parse_llm_json(raw) == {"intent": "follow_up", "patient_name": None}
    with pytest.raises(Exception):
        _parse_llm_json("not json {{")
    with pytest.raises(Exception):
        _parse_llm_json("[1, 2, 3]")  # not a JSON object


def test_interpret_drops_invented_patient_name(monkeypatch):
    """A patient name absent from the user's text must be dropped, so
    retrieval can never target an identity the model fabricated."""
    llm_service = _force_live_mode(monkeypatch)
    monkeypatch.setattr(
        llm_service, "_call_llm",
        lambda *a, **k: '{"intent": "retrieve_report", "patient_name": "John Smith"}',
    )
    result = llm_service.interpret_query("Open Priya Sharma's latest OPG report")
    assert result.patient_name is None
    assert result.fallback_reason is None  # response itself was valid


def test_dictation_wholly_invented_output_rejected(monkeypatch):
    """If NO observation appears in the transcript, the AI output must be
    discarded entirely in favour of the safe transcript-only fallback."""
    llm_service = _force_live_mode(monkeypatch)
    invented = json.dumps({
        "findings": [{"site": "upper right", "observation": "Acute abscess at 11", "severity": "severe"}],
        "anatomical_site": "upper right",
        "draft_report": "CLINICAL DICTATION DRAFT\nAcute abscess at 11.",
        "uncertainties": [],
        "requires_review": False,  # model must not be able to disable review
    })
    monkeypatch.setattr(llm_service, "_call_llm", lambda *a, **k: invented)
    transcript = "Mild bone loss noted at lower left molar."
    result = llm_service.structure_dictation(transcript)
    assert result.requires_review is True
    assert "abscess" not in result.draft_report.lower()
    assert transcript[:50] in result.draft_report


def test_dictation_grounded_findings_kept_and_invented_dropped(monkeypatch):
    """Verbatim findings survive; findings not in the transcript are dropped
    and surfaced as an uncertainty."""
    llm_service = _force_live_mode(monkeypatch)
    raw = json.dumps({
        "findings": [
            {"site": "lower left molar", "observation": "Mild bone loss noted at lower left molar", "severity": "mild"},
            {"site": "upper right", "observation": "Large periapical abscess", "severity": "severe"},
        ],
        "anatomical_site": "lower left molar",
        "draft_report": "CLINICAL DICTATION DRAFT\nMild bone loss noted at lower left molar.",
        "uncertainties": [],
        "requires_review": False,
    })
    monkeypatch.setattr(llm_service, "_call_llm", lambda *a, **k: raw)
    transcript = "Mild bone loss noted at lower left molar."
    result = llm_service.structure_dictation(transcript)
    assert len(result.findings) == 1
    assert result.findings[0]["observation"].lower().startswith("mild bone loss")
    assert "abscess" not in json.dumps(result.findings).lower()
    assert any("discarded" in u for u in result.uncertainties)
    assert result.requires_review is True


def test_follow_up_fabricated_evidence_replaced(monkeypatch):
    """Evidence strings not found in the report must be replaced with the
    report's own findings — fabricated evidence never reaches the client."""
    llm_service = _force_live_mode(monkeypatch)
    fake = json.dumps({
        "answer": "Mild horizontal bone loss is present.",
        "evidence": ["Bone loss of 9mm everywhere"],
    })
    monkeypatch.setattr(llm_service, "_call_llm", lambda *a, **k: fake)
    report = get_report_by_id("R001")
    result = llm_service.answer_follow_up("Summarise it", report)
    assert result["source_report_id"] == "R001"
    assert "9mm everywhere" not in " ".join(result["evidence"])
    assert result["evidence"] == [f["observation"] for f in report["findings"]]


def test_follow_up_verbatim_evidence_kept(monkeypatch):
    llm_service = _force_live_mode(monkeypatch)
    report = get_report_by_id("R001")
    obs = report["findings"][0]["observation"]
    fake = json.dumps({"answer": "The key finding is described below.", "evidence": [obs]})
    monkeypatch.setattr(llm_service, "_call_llm", lambda *a, **k: fake)
    result = llm_service.answer_follow_up("What are the findings?", report)
    assert result["evidence"] == [obs]
    assert result["answer"]


# ── Transcript metadata split ────────────────────────────────────────────────

def test_split_metadata_separates_header_from_clinical():
    """`name:`/`dob:`/`tooth:` header lines are lifted out as metadata; the
    clinical narrative is left intact for structuring."""
    transcript = (
        "name: Priya Sharma\n"
        "dob: 1990-05-14\n"
        "tooth: 26\n"
        "Mild bone loss noted at lower left molar."
    )
    clinical, meta = _split_metadata(transcript)
    assert meta == {"name": "Priya Sharma", "dob": "1990-05-14", "tooth": "26"}
    assert "name:" not in clinical and "dob:" not in clinical
    assert "Mild bone loss" in clinical


def test_split_metadata_mixed_line_stays_clinical():
    """A line beginning with a key but carrying a sentence must NOT be
    swallowed into metadata — clinical content is preserved."""
    clinical, meta = _split_metadata("name: Priya. Mild bone loss at 46.")
    assert "name" not in meta
    assert "Mild bone loss" in clinical


def test_metadata_only_input_yields_no_findings():
    """Header-only input returns an explicit, non-AI-structured result with the
    original transcript and metadata echoed."""
    transcript = "name: Priya Sharma\ndob: 1990-05-14"
    result = structure_dictation(transcript)
    assert result.findings == []
    assert result.fallback_reason is not None
    assert "no clinical findings" in result.fallback_reason.lower()
    assert result.transcript == transcript
    assert result.metadata == {"name": "Priya Sharma", "dob": "1990-05-14"}
    assert result.requires_review is True


def test_is_metadata_finding_detects_header_line():
    assert _is_metadata_finding("name: Priya Sharma") is True
    assert _is_metadata_finding("tooth: 26") is True
    # A clinical sentence is never treated as a metadata finding
    assert _is_metadata_finding("Mild bone loss at lower left molar.") is False


# ── Provenance / fallback flag ───────────────────────────────────────────────

def test_provider_error_sets_fallback_reason(monkeypatch):
    """A provider failure must set fallback_reason so the UI can label the
    result as not AI-generated."""
    llm_service = _force_live_mode(monkeypatch)

    def _boom(*args, **kwargs):
        raise TimeoutError("provider timeout")

    monkeypatch.setattr(llm_service, "_call_llm", _boom)
    result = llm_service.structure_dictation("Mild bone loss noted at lower left molar.")
    assert result.fallback_reason is not None
    assert "provider unavailable" in result.fallback_reason.lower()
    assert result.requires_review is True


def test_grounded_ai_result_has_no_fallback_reason(monkeypatch):
    """A valid, grounded AI response is genuinely AI-structured → no fallback."""
    llm_service = _force_live_mode(monkeypatch)
    raw = json.dumps({
        "findings": [
            {"site": "lower left molar", "observation": "Mild bone loss noted at lower left molar", "severity": "mild"},
        ],
        "anatomical_site": "lower left molar",
        "draft_report": "CLINICAL DICTATION DRAFT\nMild bone loss noted at lower left molar.",
        "uncertainties": [],
        "requires_review": True,
    })
    monkeypatch.setattr(llm_service, "_call_llm", lambda *a, **k: raw)
    result = llm_service.structure_dictation("Mild bone loss noted at lower left molar.")
    assert result.fallback_reason is None
    assert len(result.findings) == 1


def test_http_error_carries_status_detail():
    """HTTP provider errors surface their status (e.g. rate limiting)."""

    class _Resp:
        status_code = 429
        reason_phrase = "Too Many Requests"

    class _Exc(Exception):
        response = _Resp()

    assert _describe_llm_error(_Exc()) == "HTTP 429 rate limited"


# ── Grounding tolerance & no-name-as-finding ─────────────────────────────────

def test_grounding_reworded_finding_survives(monkeypatch):
    """A reworded-but-supported finding must survive grounding (the old
    byte-for-byte substring check dropped it as a false 'not present')."""
    llm_service = _force_live_mode(monkeypatch)
    transcript = "Periapical radiolucency at tooth 26, approximately 4mm."
    raw = json.dumps({
        "findings": [
            {"site": "tooth 26", "observation": "A periapical radiolucency of about 4mm is present at tooth 26.", "severity": "moderate"},
        ],
        "anatomical_site": "tooth 26",
        "draft_report": "CLINICAL DICTATION DRAFT\nPeriapical radiolucency at tooth 26.",
        "uncertainties": [],
        "requires_review": True,
    })
    monkeypatch.setattr(llm_service, "_call_llm", lambda *a, **k: raw)
    result = llm_service.structure_dictation(transcript)
    assert len(result.findings) == 1
    assert result.fallback_reason is None


def test_no_name_as_finding_dropped(monkeypatch):
    """A patient-name header echoed back as a finding must never surface as
    clinical content, and must not survive grounding."""
    llm_service = _force_live_mode(monkeypatch)
    transcript = "Mild bone loss noted at lower left molar."
    raw = json.dumps({
        "findings": [
            {"site": "see header", "observation": "name: Priya Sharma", "severity": "unspecified"},
            {"site": "lower left molar", "observation": "Mild bone loss noted at lower left molar", "severity": "mild"},
        ],
        "anatomical_site": "lower left molar",
        "draft_report": "CLINICAL DICTATION DRAFT\nMild bone loss noted at lower left molar.",
        "uncertainties": [],
        "requires_review": True,
    })
    monkeypatch.setattr(llm_service, "_call_llm", lambda *a, **k: raw)
    result = llm_service.structure_dictation(transcript)
    joined = " ".join(str(f.get("observation", "")) for f in result.findings)
    assert "Priya Sharma" not in joined
    assert len(result.findings) == 1


# ── New additive report fields ───────────────────────────────────────────────

def test_report_records_carry_new_fields():
    report = get_report_by_id("R001")
    assert report["tooth_number"] == "26"
    assert report["encounter_type"] == "initial"
    assert report["procedure_date"] == report["date"]
    assert report["billing_codes"]["cdt_proposed"] == "D0150"
    f = report["findings"][0]
    assert f["category"] == "Periapical pathology"
    assert f["resolution_plan"]
    assert isinstance(f["complication_detected"], bool)


def test_report_record_null_tooth_and_billing():
    report = get_report_by_id("R004")  # bitewing, no single tooth
    assert report["tooth_number"] is None
    assert report["billing_codes"]["icd_10"] is None
    assert report["billing_codes"]["cdt_proposed"] == "D0274"


# ── Conversational fallbacks (demo mode / provider failure) ──────────────────

def test_demo_interpret_greeting_only_for_bare_salutations():
    """The deterministic fallback recognises only bare salutations — it is not
    a conversational parser."""
    assert _demo_interpret_query("hi").intent == "greeting"
    assert _demo_interpret_query("Hello there!").intent == "greeting"
    assert _demo_interpret_query("good morning").intent == "greeting"
    # A salutation glued to a real task is NOT a greeting
    assert _demo_interpret_query("hi, open Priya's OPG report").intent == "retrieve_report"


def test_demo_interpret_identity_question():
    assert _demo_interpret_query("Are you Gemini?").intent == "assistant_identity"
    assert _demo_interpret_query("Who are you?").intent == "assistant_identity"
    # A normal report question is never mistaken for an identity question
    assert _demo_interpret_query("Show Rohan Verma's bitewing report").intent == "retrieve_report"


def test_demo_interpret_unknown_patient_shorthand():
    """"Rahul report" / "open rahul report" extract the patient name so a real
    lookup happens; generic nouns never become names."""
    r = _demo_interpret_query("Rahul report")
    assert r.intent == "retrieve_report" and r.patient_name == "Rahul"
    r = _demo_interpret_query("open rahul report")
    assert r.intent == "retrieve_report" and r.patient_name == "Rahul"
    r = _demo_interpret_query("Priya Sharma report")
    assert r.intent == "retrieve_report" and r.patient_name == "Priya Sharma"
    # Request vocabulary is never a name
    assert _demo_interpret_query("the report").patient_name is None
    assert _demo_interpret_query("Summarize her report").patient_name is None


# ── Gemini classifies chat messages (stubbed LLM, no paid calls) ────────────

def test_interpret_greeting_via_llm(monkeypatch):
    """Natural greeting phrasing is classified as a greeting by the LLM."""
    llm_service = _force_live_mode(monkeypatch)
    monkeypatch.setattr(
        llm_service, "_call_llm", lambda *a, **k: json.dumps({"intent": "greeting"})
    )
    result = llm_service.interpret_query("well hello there, how are things going today?")
    assert result.intent == "greeting"
    assert result.fallback_reason is None


def test_interpret_identity_via_llm(monkeypatch):
    llm_service = _force_live_mode(monkeypatch)
    monkeypatch.setattr(
        llm_service, "_call_llm",
        lambda *a, **k: json.dumps({"intent": "assistant_identity"}),
    )
    result = llm_service.interpret_query("just curious — which model am i talking to?")
    assert result.intent == "assistant_identity"
    assert result.fallback_reason is None


def test_interpret_unsupported_via_llm(monkeypatch):
    llm_service = _force_live_mode(monkeypatch)
    monkeypatch.setattr(
        llm_service, "_call_llm",
        lambda *a, **k: json.dumps({"intent": "unsupported"}),
    )
    result = llm_service.interpret_query("explain photosynthesis please")
    assert result.intent == "unsupported"


def test_interpret_contextual_follow_up_via_llm(monkeypatch):
    """"Give me a summary" is classified by the LLM as a contextual follow-up."""
    llm_service = _force_live_mode(monkeypatch)
    monkeypatch.setattr(
        llm_service, "_call_llm", lambda *a, **k: json.dumps({"intent": "follow_up"})
    )
    result = llm_service.interpret_query(
        "Give me a summary", {"active_report_id": "R001"}
    )
    assert result.intent == "follow_up"


def test_chat_and_dictation_actually_call_the_llm(monkeypatch):
    """Spy: every chat and dictation request must really hit the provider."""
    llm_service = _force_live_mode(monkeypatch)
    calls = []

    def _spy(system, user):
        calls.append(user)
        return json.dumps({"intent": "greeting"})

    monkeypatch.setattr(llm_service, "_call_llm", _spy)
    llm_service.interpret_query("hi")
    assert len(calls) == 1

    result = llm_service.structure_dictation("Mild bone loss noted at lower left molar.")
    assert len(calls) == 2  # the dictation request also reached the provider
    assert result.fallback_reason is not None  # (stubbed output failed validation)


def test_greeting_request_routed_through_gemini(monkeypatch):
    """With a Gemini key configured, a chat request goes through _call_gemini."""
    from services import llm_service
    monkeypatch.setattr(llm_service, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm_service, "DEMO_MODE", False)
    monkeypatch.setattr(
        llm_service, "_call_gemini",
        lambda s, u: json.dumps({"intent": "greeting"}),
    )
    result = llm_service.interpret_query("hi")
    assert result.intent == "greeting"
    assert result.fallback_reason is None
