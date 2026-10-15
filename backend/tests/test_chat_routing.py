"""
Regression tests for chatbot intent routing.

Reported failures:
  * "Who are you?", "What can you do?" and "What is Viscan AI?" all returned one
    identical identity reply;
  * general dental questions were answered with the radiology-request fallback
    (or routed to the report handler);
  * greetings and jokes were answered with the radiology-request fallback;
  * "Book a scan for Priya" was treated as a report retrieval, implying an
    action the app cannot perform.

These tests pin the CLASSIFICATION and ROUTING rather than individual phrases:
every category is exercised with several phrasings (including ones the
deterministic parser has never seen), and the Gemini path is asserted through
the system prompt it receives and through stubbed model output. Report
retrieval, disambiguation, follow-up evidence and patient lookups are re-pinned
here so the routing fix cannot quietly weaken them.

No paid API calls: like the rest of the suite, this file runs in demo mode
unless a test explicitly forces (stubbed) live mode.
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
from services.llm_service import _demo_interpret_query

client = TestClient(app)

# The wording users saw for anything that was not a report request. None of the
# conversational intents below may ever come back with it.
RADIOLOGY_FALLBACK = "couldn't turn that into a radiology request"


def _ask(text: str, session_context: dict | None = None) -> dict:
    resp = client.post(
        "/query", json={"text": text, "session_context": session_context}
    )
    assert resp.status_code == 200, f"{text!r} -> {resp.status_code}"
    return resp.json()


def _force_live_mode(monkeypatch):
    """Simulate a configured provider end-to-end (service + API layers)."""
    from services import llm_service
    import main as main_module
    monkeypatch.setattr(llm_service, "DEMO_MODE", False)
    monkeypatch.setattr(main_module, "DEMO_MODE", False)
    return llm_service


# ── 1. Identity / capabilities / about-the-app are three different answers ───

def test_identity_trio_gets_three_distinct_appropriate_answers():
    """"Who are you?", "What can you do?" and "What is Viscan AI?" must each get
    their own answer — never one identical hardcoded reply."""
    who = _ask("Who are you?")
    caps = _ask("What can you do?")
    about = _ask("What is Viscan AI?")

    assert who["intent"] == "assistant_identity"
    assert caps["intent"] == "assistant_capabilities"
    assert about["intent"] == "about_app"

    messages = [who["message"], caps["message"], about["message"]]
    assert len(set(messages)) == 3, "identity trio must not share one reply"

    assert "viscan" in who["message"].lower()
    caps_msg = caps["message"].lower()
    assert "retrieve" in caps_msg and "dictation" in caps_msg   # lists capabilities
    about_msg = about["message"].lower()
    assert "dental radiology" in about_msg and "app" in about_msg  # describes the app

    for data, message in zip((who, caps, about), messages):
        assert RADIOLOGY_FALLBACK not in message
        assert message.strip()
        assert data["report"] is None
        assert data["matched_patients"] == []
        assert data["patient_name"] is None


def test_identity_trio_variants_route_to_their_own_intent():
    for text, intent in (
        ("Who are you?", "assistant_identity"),
        ("Are you Gemini?", "assistant_identity"),
        ("Are you an AI?", "assistant_identity"),
        ("What can you do?", "assistant_capabilities"),
        ("What else can you do?", "assistant_capabilities"),
        ("How do you help?", "assistant_capabilities"),
        ("What is Viscan AI?", "about_app"),
        ("Tell me about this app", "about_app"),
        ("What does this app do?", "about_app"),
    ):
        data = _ask(text)
        assert data["intent"] == intent, f"{text!r} -> {data['intent']}"
        assert RADIOLOGY_FALLBACK not in data["message"], text
        assert data["report"] is None, text


def test_identity_answers_stay_truthful_about_the_model():
    """"Are you Gemini?" → the reply describes Viscan AI and never names the
    underlying AI provider."""
    data = _ask("Are you Gemini?")
    msg = data["message"].lower()
    assert "viscan" in msg
    assert "gemini" not in msg


# ── 2. General dental knowledge: education, never a report request ───────────

_GENERAL_DENTAL_EXAMPLES = [
    "What is gingivitis?",
    "Why do my gums bleed when I brush?",
    "What causes tooth sensitivity?",
    "Is a root canal painful?",
    "At what age do wisdom teeth come in?",
    "How often should I floss?",
]


def test_general_dental_questions_are_answered_as_education():
    for text in _GENERAL_DENTAL_EXAMPLES:
        data = _ask(text)
        assert data["intent"] == "general_dental", f"{text!r} -> {data['intent']}"
        assert data["report"] is None, text
        assert data["matched_patients"] == [], text
        assert data["patient_name"] is None, text
        msg = data["message"].lower()
        assert RADIOLOGY_FALLBACK not in msg, text
        # general education, explicitly separate from patient-specific findings
        assert "general educational information" in msg, text
        assert "not a diagnosis" in msg, text
        assert "not based on any patient's report" in msg, text


def test_general_dental_stays_general_even_with_an_open_report():
    """Educational questions are never answered from (or attached to) a patient's
    report, however open it is."""
    for text in _GENERAL_DENTAL_EXAMPLES:
        data = _ask(text, {"active_report_id": "R001"})
        assert data["intent"] == "general_dental", f"{text!r} -> {data['intent']}"
        assert data["report"] is None, text


def test_general_dental_answer_comes_from_the_configured_ai(monkeypatch):
    """With a provider configured the explanation is generated by the AI under a
    safety prompt (no diagnoses, no patient data), then framed as education."""
    llm_service = _force_live_mode(monkeypatch)
    answer = (
        "Gingivitis is inflammation of the gums, usually caused by plaque "
        "build-up along the gum line. It is common and often reversible with "
        "good cleaning, but a dentist should assess any individual case."
    )
    calls = []

    def _llm(system, user):
        calls.append((system, user))
        if "query interpreter" in system:
            return json.dumps({"intent": "general_dental"})
        return answer

    monkeypatch.setattr(llm_service, "_call_llm", _llm)
    data = _ask("What is gingivitis?")

    assert data["intent"] == "general_dental"
    assert data["fallback_reason"] is None      # really served by the provider
    assert answer in data["message"]
    assert "not a diagnosis" in data["message"].lower()
    assert "not based on any patient's report" in data["message"].lower()
    assert RADIOLOGY_FALLBACK not in data["message"]

    assert len(calls) == 2                      # classified, then answered
    education_system, education_user = calls[1]
    assert education_user == "What is gingivitis?"
    assert "NOT a diagnosis" in education_system
    assert "personalized treatment advice" in education_system
    assert "patient" in education_system        # told not to touch patient records


def test_interpreter_prompt_teaches_every_supported_intent(monkeypatch):
    """Gemini is the classifier, so its prompt must describe every intent —
    including the ones that used to be lumped into 'unsupported'."""
    llm_service = _force_live_mode(monkeypatch)
    seen = {}

    def _spy(system, user):
        seen["system"] = system
        return json.dumps({"intent": "greeting"})

    monkeypatch.setattr(llm_service, "_call_llm", _spy)
    client.post("/query", json={"text": "hi"})

    system = seen["system"]
    for intent in (
        "greeting", "assistant_identity", "assistant_capabilities", "about_app",
        "small_talk", "general_dental", "retrieve_report", "patient_info",
        "dictation", "follow_up", "unavailable_action", "unsupported",
    ):
        assert intent in system, f"prompt never mentions intent {intent}"
    assert "never classify it as retrieve_report or follow_up" in system
    assert "not retrieve_report" in system.lower()   # unimplemented actions


# ── 3. Harmless conversation is conversation ────────────────────────────────

def test_jokes_and_small_talk_are_not_radiology_requests():
    for text in ("Tell me a joke", "how are you?", "thanks a lot!", "bye!"):
        data = _ask(text)
        assert data["intent"] == "small_talk", f"{text!r} -> {data['intent']}"
        assert data["report"] is None, text
        assert data["matched_patients"] == [], text
        msg = data["message"].lower()
        assert RADIOLOGY_FALLBACK not in msg, text
        assert "no active report" not in msg, text
        assert "couldn't reach the ai service" not in msg, text
    # …and the reply is friendly rather than an error
    assert "happy to chat" in _ask("Tell me a joke")["message"].lower()


def test_greetings_are_answered_as_greetings():
    for text in ("Hi", "Hello there!", "Hey, good morning!", "good morning everyone"):
        data = _ask(text)
        assert data["intent"] == "greeting", f"{text!r} -> {data['intent']}"
        assert RADIOLOGY_FALLBACK not in data["message"], text
        assert "Hi!" in data["message"], text
        assert data["report"] is None, text


def test_greeting_glued_to_a_task_is_still_the_task():
    data = _ask("hi, open Priya Sharma's latest OPG report")
    assert data["intent"] == "retrieve_report"
    assert data["report"] is not None
    assert data["report"]["patient_id"] == "P001"


# ── 4. Unimplemented actions are stated honestly ─────────────────────────────

def test_booking_a_scan_states_that_it_is_not_implemented():
    for text in (
        "Book a scan for Priya",
        "Schedule an appointment for Priya",
        "Cancel my appointment",
        "book priya sharma opg",
        "delete Priya Sharma report",
    ):
        data = _ask(text)
        assert data["intent"] == "unavailable_action", f"{text!r} -> {data['intent']}"
        assert data["report"] is None, text           # nothing was retrieved…
        assert data["matched_patients"] == [], text   # …and no patient was opened
        msg = data["message"].lower()
        assert "not an implemented capability" in msg, text
        assert "nothing was booked" in msg, text      # never claims it succeeded
        assert "won't say an action succeeded" in msg, text
        assert RADIOLOGY_FALLBACK not in msg, text
    # the reply quotes the request back, so it is clearly about THIS action
    assert "book a scan for priya" in _ask("Book a scan for Priya")["message"].lower()


def test_unimplemented_action_never_carries_a_patient_name():
    data = _ask("Book a scan for Priya")
    assert data["intent"] == "unavailable_action"
    assert data["patient_name"] is None


# ── 5. Report-related intents are preserved ──────────────────────────────────

def test_report_retrieval_still_resolves():
    data = _ask("Open Priya Sharma's latest OPG report")
    assert data["intent"] == "retrieve_report"
    assert data["ambiguous"] is False
    assert data["report"] is not None
    assert data["report"]["patient_id"] == "P001"
    assert data["report"]["type"] == "OPG"


def test_ambiguous_retrieval_still_asks_the_user():
    data = _ask("Open Sharma's OPG")
    assert data["intent"] == "retrieve_report"
    assert data["ambiguous"] is True
    assert data["report"] is None
    assert "select the correct patient" in data["message"].lower()


def test_unknown_patient_is_never_invented():
    data = _ask("Rahul report")
    assert data["intent"] == "retrieve_report"
    assert data["patient_name"] == "Rahul"
    assert data["report"] is None
    assert "no patient" in data["message"].lower()


def test_report_grounded_question_uses_the_open_report():
    for text in (
        "What are the key findings?",
        "Summarize her report",
        "Summarize this report please",
        "What does this OPG show?",
        "Give me a summary",
    ):
        data = _ask(text, {"active_report_id": "R001"})
        assert data["intent"] == "follow_up", f"{text!r} -> {data['intent']}"
        assert data["report"] is not None, text
        assert data["report"]["id"] == "R001", text

    fu = client.post(
        "/follow-up",
        json={"question": "What are the key findings?", "report_id": "R001"},
    )
    assert fu.status_code == 200
    body = fu.json()
    assert body["answer"]
    assert body["source_report_id"] == "R001"
    assert body["evidence"]                     # evidence citations still flow


def test_report_question_without_an_open_report_guides_instead_of_inventing():
    data = _ask("Summarize her report")
    assert data["intent"] == "follow_up"
    assert data["report"] is None
    assert "open a report" in data["message"].lower()


def test_patient_contact_lookup_is_preserved():
    data = _ask("what is Rohan Verma's phone number?")
    assert data["intent"] == "patient_info"
    assert "+91-77700-33333" in data["message"]


def test_dictation_and_unsupported_intents_are_preserved():
    data = _ask("Note: mild bone loss at 46")
    assert data["intent"] == "dictation"
    assert "dictation" in data["message"].lower()

    data = _ask("Write me a poem about the seaside")
    assert data["intent"] in ("unsupported", "unknown")
    assert "report" in data["message"].lower()


# ── 6. Gemini classifies the new intents (stubbed, no paid calls) ────────────

def test_gemini_classified_conversation_intents_end_to_end(monkeypatch):
    llm_service = _force_live_mode(monkeypatch)
    replies = {
        "Who are you?": {"intent": "assistant_identity"},
        "What can you do?": {"intent": "assistant_capabilities"},
        "What is Viscan AI?": {"intent": "about_app"},
        "Tell me a joke": {"intent": "small_talk"},
        "What is gingivitis?": {"intent": "general_dental"},
        "Book a scan for Priya": {"intent": "unavailable_action", "patient_name": "Priya"},
    }
    for text, payload in replies.items():
        monkeypatch.setattr(
            llm_service, "_call_llm",
            lambda *a, _payload=payload, **k: json.dumps(_payload),
        )
        data = _ask(text)
        assert data["intent"] == payload["intent"], f"{text!r} -> {data['intent']}"
        assert data["report"] is None, text
        assert data["matched_patients"] == [], text
        assert RADIOLOGY_FALLBACK not in data["message"], text
        # conversation never targets a record, whatever the model saw
        assert data["patient_name"] is None, text


def test_provider_outage_explains_itself_for_every_conversation_intent(monkeypatch):
    llm_service = _force_live_mode(monkeypatch)

    def _boom(*args, **kwargs):
        raise ConnectionError("provider down")

    monkeypatch.setattr(llm_service, "_call_llm", _boom)

    expectations = {
        "What can you do?": "assistant_capabilities",
        "What is Viscan AI?": "about_app",
        "Tell me a joke": "small_talk",
        "What is gingivitis?": "general_dental",
        "Book a scan for Priya": "unavailable_action",
        "Who are you?": "assistant_identity",
        "Hi": "greeting",
    }
    for text, intent in expectations.items():
        data = _ask(text)
        assert data["intent"] == intent, f"{text!r} -> {data['intent']}"
        msg = data["message"].lower()
        assert RADIOLOGY_FALLBACK not in msg, text
        assert "didn't understand" not in msg, text
    # the dental question explains the outage instead of inventing an answer
    dental = _ask("What is gingivitis?")
    assert dental["fallback_reason"] is not None
    assert "couldn't reach the ai service" in dental["message"].lower()
    assert "not a diagnosis" in dental["message"].lower()


def test_provider_outage_still_retrieves_real_reports(monkeypatch):
    llm_service = _force_live_mode(monkeypatch)

    def _boom(*args, **kwargs):
        raise ConnectionError("provider down")

    monkeypatch.setattr(llm_service, "_call_llm", _boom)
    data = _ask("open priya sharma's opg")
    assert data["intent"] == "retrieve_report"
    assert data["report"] is not None
    assert data["report"]["patient_id"] == "P001"
    assert data["fallback_reason"] is not None


# ── 7. Deterministic classifier generalises over phrasing ────────────────────

def test_demo_classifier_separates_conversation_from_report_work():
    for text in ("Tell me a joke", "how are you?", "thanks!", "bye"):
        assert _demo_interpret_query(text).intent == "small_talk", text
    for text in (
        "What is gingivitis?",
        "Why do my gums bleed when I brush?",
        "Is a root canal painful?",
        "Should I call a dentist?",
    ):
        assert _demo_interpret_query(text).intent == "general_dental", text
    for text in (
        "Book a scan for Priya",
        "Schedule an appointment",
        "delete the report",
        "When is my appointment?",
    ):
        assert _demo_interpret_query(text).intent == "unavailable_action", text
    # …while genuine report requests are untouched
    for text in ("open priya", "Rahul report", "Priya", "Show Rohan Verma's bitewing report"):
        assert _demo_interpret_query(text).intent == "retrieve_report", text


def test_demo_classifier_contextual_follow_up_needs_no_report_wording():
    ctx = {"active_report_id": "R001"}
    assert _demo_interpret_query("Is this normal?", ctx).intent == "follow_up"
    # …but the same sentence without context is not a report request
    assert _demo_interpret_query("Is this normal?").intent in (
        "unsupported", "unknown"
    )


def test_conversation_intents_never_carry_a_patient_name():
    for text in (
        "tell me a joke about Priya",
        "What can you do with Priya?",
        "Book a scan for Priya",
        "Are you Gemini?",
    ):
        result = _demo_interpret_query(text)
        assert result.patient_name is None, text
        assert result.report_type is None or result.intent == "unavailable_action", text
