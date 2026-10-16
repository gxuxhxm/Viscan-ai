from __future__ import annotations
import re
from contextlib import asynccontextmanager
from datetime import date
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from models import (
    QueryRequest, QueryResponse,
    DictationRequest, DictationResponse,
    FollowUpRequest, FollowUpResponse,
    PatientSummary, ReportDetail, FindingItem, ReportSummary,
    SaveReportRequest,
)
from services.retrieval import (
    resolve_patient_name,
    get_patient_by_id,
    get_latest_report_for_patient,
    get_reports_for_patient,
    get_report_by_id,
    get_patient_by_name,
    add_patient,
    add_report_if_new,
    find_identical_patient,
)
from services.llm_service import (
    interpret_query,
    structure_dictation,
    answer_follow_up,
    answer_general_dental,
    DEMO_MODE,
    LLM_PROVIDER,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Viscan AI Backend",
    description="Dental reporting AI API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:4173",  # vite preview
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

# Canned assistant replies. These are deterministic on purpose: a greeting
# answer and the truthful assistant-identity answer must never be paraphrased
# (or worse, misattributed) by a language model. Identity/capability replies
# describe Viscan AI and never name the underlying AI provider.
GREETING_MESSAGE = (
    "Hi! I can help retrieve reports, structure clinical dictation, and summarize reports."
)
ASSISTANT_IDENTITY_MESSAGE = (
    "I'm Viscan AI, a dental reporting assistant. I can retrieve a patient's "
    "latest radiology report by name (OPG, periapical, bitewing, CBCT), answer "
    "questions about the report you have open, look up patient contact details "
    "on file, and structure your clinical dictation into a review-ready draft. "
    "How can I help?"
)
UNSUPPORTED_MESSAGE = (
    "I can help with radiology reports and clinical dictation, but I couldn't turn "
    "that into a radiology request. Try something like \"Open Priya Sharma's latest "
    "OPG report\", \"Show Rohan Verma's bitewing report\", or dictate findings in "
    "the Dictation tab."
)
LLM_FALLBACK_MESSAGE = (
    "I couldn't reach the AI service right now, so I could only try simple report "
    "commands (like \"Open Priya Sharma's latest OPG report\"). Please try again in "
    "a moment."
)
# "What can you do?" — the implemented feature set, kept separate from the
# identity answer so the two questions never get one identical reply.
CAPABILITIES_MESSAGE = (
    "Here's what I can do in Viscan AI:\n"
    "• Retrieve a patient's latest radiology report by name (OPG, periapical, "
    "bitewing, CBCT), resolving ambiguous names before opening anything\n"
    "• Answer questions about the report you have open, quoting that report as "
    "evidence\n"
    "• Look up the contact and demographic details a patient has on file\n"
    "• Structure your clinical dictation into a review-ready draft report\n"
    "• Explain general dental topics as educational information — never a diagnosis\n"
    "Booking scans or appointments, sending messages and similar app actions "
    "aren't implemented here; I'll say so rather than pretend they ran. What "
    "would you like to start with?"
)
# "What is Viscan AI?" — about the application, not about the assistant's identity.
ABOUT_APP_MESSAGE = (
    "Viscan AI is a dental radiology reporting app for clinics. A clinician can "
    "retrieve a patient's radiology reports (OPG, periapical, bitewing, CBCT), "
    "dictate findings that are structured into a review-ready draft report, and "
    "ask questions about the report that is open — with quotes from that report "
    "as evidence. Its AI features (interpreting your messages, structuring "
    "dictation, answering report questions) work alongside record retrieval, "
    "evidence citations and safety checks that run in the app's own code."
)
# Harmless conversation (jokes, thanks, farewells): friendly, never a
# radiology request.
SMALL_TALK_MESSAGE = (
    "Happy to chat! Here's one from the clinic: why did the dentist make a great "
    "gardener? Because he was an expert at root canals. When you're ready, I can "
    "retrieve a patient's radiology report, answer questions about the report you "
    "have open, look up patient details, or structure a dictation."
)
# "Book a scan…" / "email this report…" — state plainly that the capability is
# NOT implemented, echo the request, and never imply anything was done.
UNAVAILABLE_ACTION_MESSAGE = (
    "I can't do that: \"{action}\" is not an implemented capability of this "
    "version of Viscan AI, so nothing was booked, scheduled, sent or deleted — "
    "I won't say an action succeeded when the app cannot perform it. What Viscan "
    "AI does implement: retrieving a patient's existing radiology reports (OPG, "
    "periapical, bitewing, CBCT), answering questions about the report you have "
    "open, looking up patient contact details on file, structuring clinical "
    "dictation into a draft, and exporting the draft you are editing. Ask me for "
    "one of those instead."
)


def _build_report_detail(report: dict) -> ReportDetail:
    return ReportDetail(
        id=report["id"],
        patient_id=report["patient_id"],
        type=report["type"],
        date=report["date"],
        radiologist=report["radiologist"],
        status=report["status"],
        findings=[FindingItem(**f) for f in report.get("findings", [])],
        impression=report["impression"],
        draft_report=report["draft_report"],
        tooth_number=report.get("tooth_number"),
        encounter_type=report.get("encounter_type"),
        procedure_date=report.get("procedure_date"),
        billing_codes=report.get("billing_codes"),
        anatomical_site=report.get("anatomical_site"),
        clinician_confirmed=report.get("clinician_confirmed"),
        source_transcript=report.get("source_transcript"),
        provenance=report.get("provenance"),
    )


# ── Save-report helpers ──────────────────────────────────────────────────────

_IMPRESSION_RE = re.compile(
    r"^[ \t]*IMPRESSION[ \t]*:?[ \t]*(?P<inline>.*)$", re.IGNORECASE | re.MULTILINE
)
_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 /-]+:")

_REPORT_TYPE_HINTS = (
    ("BITEWING", re.compile(r"\bbitewings?\b", re.IGNORECASE)),
    ("CBCT", re.compile(r"\bcbct\b|cone[- ]beam", re.IGNORECASE)),
    ("PERIAPICAL", re.compile(r"\bperiapical (?:radiographs?|films?)\b", re.IGNORECASE)),
    ("OPG", re.compile(r"\bopg\b|orthopantomogram|panoramic", re.IGNORECASE)),
)


def _truncate(text: str, limit: int = 600) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _derive_impression(
    draft_report: str, findings: list[dict], anatomical_site: str | None = None
) -> str:
    """
    Impression for a dictation-derived report, taken only from content the
    clinician supplied: the draft's own IMPRESSION section when present,
    otherwise the dictated finding texts. Never invents a clinical conclusion.
    """
    match = _IMPRESSION_RE.search(draft_report or "")
    if match:
        parts = [match.group("inline").strip()] if match.group("inline").strip() else []
        for line in (draft_report[match.end():] or "").splitlines():
            line = line.strip()
            if not line or _HEADING_RE.match(line):
                if parts:
                    break  # blank line or the next heading (REPORT BY:, …) ends the section
                continue
            parts.append(line.lstrip("-*• ").strip())
        section = _truncate(" ".join(p for p in parts if p))
        if section:
            return section

    observations = [str(f.get("observation", "")).strip() for f in findings]
    observations = [o for o in observations if o]
    if observations:
        return _truncate(" ".join(observations))

    if anatomical_site:
        return _truncate(f"Dictated findings for {anatomical_site}: see draft report.")
    return "See draft report."


def _detect_report_type(*texts: str | None) -> str:
    """
    Modality literally stated in the dictated text, else OPG (the default
    record type in this clinic's data). The UI always sends an explicit
    report_type, so this only covers direct API use.
    """
    blob = " ".join(t for t in texts if t)
    for name, pattern in _REPORT_TYPE_HINTS:
        if pattern.search(blob):
            return name
    return "OPG"


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "demo_mode": DEMO_MODE, "llm_provider": LLM_PROVIDER}


# ── Patients ─────────────────────────────────────────────────────────────────

def _patient_report_counts() -> dict[str, int]:
    """Reports-on-file per patient id: one read of the report store."""
    from services.retrieval import _load_reports
    counts: dict[str, int] = {}
    for report in _load_reports():
        pid = report["patient_id"]
        counts[pid] = counts.get(pid, 0) + 1
    return counts


@app.get("/patients", response_model=list[PatientSummary])
def list_patients():
    from services.retrieval import _load_patients
    counts = _patient_report_counts()
    return [
        PatientSummary(**{**p, "report_count": counts.get(p["id"], 0)})
        for p in _load_patients()
    ]


@app.get("/patients/{patient_id}/reports", response_model=list[ReportSummary])
def get_patient_reports(patient_id: str):
    patient = get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")
    reports = get_reports_for_patient(patient_id)
    return [
        ReportSummary(
            id=r["id"],
            patient_id=r["patient_id"],
            type=r["type"],
            date=r["date"],
            radiologist=r["radiologist"],
            status=r["status"],
            impression=r["impression"],
            tooth_number=r.get("tooth_number"),
            encounter_type=r.get("encounter_type"),
            procedure_date=r.get("procedure_date"),
            billing_codes=r.get("billing_codes"),
            clinician_confirmed=r.get("clinician_confirmed"),
        )
        for r in reports
    ]


@app.get("/reports/{report_id}", response_model=ReportDetail)
def get_report(report_id: str):
    report = get_report_by_id(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")
    return _build_report_detail(report)


@app.post("/reports", response_model=ReportDetail)
def save_report(req: SaveReportRequest):
    """
    Persist an edited dictation draft as a report record.

    Reuses the existing report schema and id sequence (R###); a new patient is
    appended to patients.json alongside the existing ones (P###). Clinician
    confirmation is stored as an explicit status (no clinician identity or
    signature is ever invented), and the original dictated transcript plus the
    dictation provenance are preserved on the record. An identical repeat
    submission returns the already-stored report instead of duplicating it.
    """
    # ── Patient: an existing one … ──
    if req.patient_id:
        patient = get_patient_by_id(req.patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient '{req.patient_id}' not found")
        patient_id = patient["id"]
    else:
        # ── … or a brand-new one, following the existing patient schema ──
        try:
            candidate = req.patient.model_dump()
            # Retrying a failed save (or entering the same person twice) must
            # not append a duplicate: an exact identity match reuses the id
            # already on disk; anything less is a distinct patient.
            existing = find_identical_patient(candidate)
            patient_id = existing["id"] if existing else add_patient(candidate)["id"]
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=500, detail=f"Could not persist patient: {exc}")

    findings = [f.model_dump() for f in req.findings]
    impression = req.impression or _derive_impression(
        req.draft_report, findings, req.anatomical_site
    )

    record = {
        "patient_id": patient_id,
        "type": req.report_type or _detect_report_type(req.transcript, req.draft_report),
        "date": date.today().isoformat(),
        # No clinician identity is invented; unassigned until a real one is recorded.
        "radiologist": req.radiologist or "Unassigned",
        "status": "final" if req.clinician_confirmed else "draft",
        "tooth_number": req.tooth_number,
        # Primary region dictated for this report; additive, older records omit it
        "anatomical_site": req.anatomical_site,
        "encounter_type": None,
        "procedure_date": None,
        "billing_codes": None,
        "findings": findings,
        "impression": impression,
        "draft_report": req.draft_report,
        # Additive dictation-save fields (existing records simply omit them)
        "clinician_confirmed": req.clinician_confirmed,
        "source_transcript": req.transcript,
        "provenance": {
            "source": "dictation",
            "demo_mode": req.demo_mode,
            "fallback_reason": req.fallback_reason,
            "metadata": req.dictation_metadata or {},
        },
    }

    try:
        report, _created = add_report_if_new(record)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not persist report: {exc}")

    return _build_report_detail(report)


# ── Patient info (contact / demographic lookup) ─────────────────────────────

# Details the patient schema can answer. Anything else a user asks for is
# named back honestly instead of being guessed at.
_PATIENT_INFO_LABELS = {
    "contact": "contact", "dob": "date of birth", "gender": "gender/age",
    "address": "address",
}

# Details this schema does not store (patients.json holds name, dob, gender,
# phone, email only). They are surfaced to the user as "not recorded".
_UNSTORED_LABELS = {
    "blood group": "blood group", "blood type": "blood type", "allergies": "allergies",
    "allergy": "allergy", "medications": "medications", "medication": "medication",
    "medical history": "medical history", "insurance": "insurance",
    "emergency contact": "emergency contact", "next of kin": "next of kin",
    "occupation": "occupation", "weight": "weight", "height": "height",
    "blood pressure": "blood pressure", "notes": "notes", "prescription": "prescription",
}

# The interpreter prompt tells the LLM to use the USER'S OWN WORDING for
# info_fields ("phone number", "e-mail", "how old"), so the labels reaching
# _format_patient_info are free text, not schema keys. Normalising them here
# keeps a stored value from being reported as missing just because the label
# differed ("phone number" ≠ "contact"): every recognised synonym maps onto the
# canonical field that can actually answer it. Unrecognised labels pass through
# unchanged and are answered honestly as "not recorded".
_INFO_FIELD_ALIASES = {
    # how to reach the patient → the contact block (phone + email)
    "contact": "contact", "contact details": "contact", "contact information": "contact",
    "contact info": "contact", "phone": "contact", "phone number": "contact",
    "telephone": "contact", "tel": "contact", "mobile": "contact",
    "mobile number": "contact", "cell": "contact", "cell phone": "contact",
    "email": "contact", "e-mail": "contact", "email address": "contact",
    "e-mail address": "contact", "mail": "contact", "reach": "contact",
    # demographics the record holds
    "address": "address", "home address": "address", "mailing address": "address",
    "residential address": "address", "postal address": "address",
    "dob": "dob", "date of birth": "dob", "d.o.b.": "dob", "birth date": "dob",
    "birthday": "dob",
    # age is derived from the stored dob and rendered with gender (demo parity)
    "gender": "gender", "sex": "gender", "age": "gender", "how old": "gender",
}


def _normalize_info_fields(info_fields: Optional[list[str]]) -> list[str]:
    """Map free-form detail labels onto canonical, record-answerable fields.
    Case/whitespace-insensitive; duplicates collapse; unknown labels (the
    user's wording for something the schema never stores) pass through so the
    caller can answer 'not recorded' instead of guessing."""
    normalized: list[str] = []
    for field in info_fields or []:
        if not isinstance(field, str) or not field.strip():
            continue
        key = " ".join(field.casefold().split())
        canonical = _INFO_FIELD_ALIASES.get(key)
        if canonical is None and key not in _UNSTORED_LABELS and "emergency" not in key:
            # Wordy variants the exact table may miss ("phone no.", "email id",
            # "postal address"). Details the schema never stores are excluded
            # above, so this can only ever route to a field that can answer.
            if re.search(r"\b(?:address|postal)\b", key):
                canonical = "address"
            elif re.search(r"\b(?:d\.?o\.?b\.?|birthday|birth)\b", key):
                canonical = "dob"
            elif re.search(r"\b(?:gender|sex|age|how\s+old)\b", key):
                canonical = "gender"
            elif re.search(r"\b(?:phone|mobile|cell|telephone|email|e-mail|mail|contact)\b", key):
                canonical = "contact"
        label = canonical if canonical is not None else field
        if label not in normalized:
            normalized.append(label)
    return normalized


_INFO_SUMMARY_FIELDS = ("id", "name", "dob", "gender", "phone", "email", "address")


def _info_summary(patient: dict) -> PatientSummary:
    """Build a PatientSummary carrying the contact/demographic fields (the
    /patients listing omits them; patient_info needs them surfaced)."""
    return PatientSummary(
        **{k: v for k, v in patient.items() if k in _INFO_SUMMARY_FIELDS},
        report_count=0,
    )


def _resolve_patient_for_info(
    req: QueryRequest, patient_name: Optional[str]
) -> tuple[Optional[dict], Optional[list[PatientSummary]], bool, str]:
    """
    Resolve which patient a patient_info request is about.

    Priority: the open report's patient (unambiguous, no name needed) → an
    explicitly named patient resolved via the real records. Returns
    (patient, matched_summaries, ambiguous, message). Exactly one of these is
    set; ``message`` is only populated when the lookup failed or was ambiguous.
    """
    # 1. Active report in the session → its patient. This makes
    #    "give me contact details of this person" work with no name at all.
    active_id = (req.session_context or {}).get("active_report_id")
    if active_id:
        report = get_report_by_id(active_id)
        if report:
            patient = get_patient_by_id(report["patient_id"])
            if patient:
                return patient, [_info_summary(patient)], False, ""

    # 2. A name was given → resolve it against real records (never invent).
    if patient_name:
        status, matches = resolve_patient_name(patient_name)
        if status == "match":
            patient = matches[0]
            return patient, [_info_summary(patient)], False, ""
        if status == "ambiguous":
            return None, [_info_summary(m) for m in matches[:4]], True, (
                f"Multiple patients match '{patient_name}'. Please select the correct patient."
            )
        return None, None, False, (
            f"I couldn't identify which patient you mean: no patient matched '{patient_name}'. "
            "Please open their report first or give the full name."
        )

    # 3. No report open and no name given.
    return None, None, False, (
        "I'd need to know which patient, so please open their report first, "
        "or give me their name (e.g. \"contact details for Priya Sharma\")."
    )


def _format_patient_info(patient: dict, info_fields: Optional[list[str]]) -> str:
    """Render the requested on-file details. Only ever shows what the record
    actually holds; a field with no value is named as not recorded rather than
    being substituted by a different detail or guessed at."""
    requested = _normalize_info_fields(info_fields) or ["contact"]

    lines = [f"{patient['name']} ({patient['id']}): details on file:"]
    shown = False
    unstored: list[str] = []
    for field in requested:
        if field == "contact":
            phone, email = patient.get("phone"), patient.get("email")
            if phone:
                lines.append(f"• Phone: {phone}")
                shown = True
            if email:
                lines.append(f"• Email: {email}")
                shown = True
        elif field == "address":
            address = patient.get("address")
            if address:
                lines.append(f"• Address: {address}")
                shown = True
        elif field == "dob":
            dob = patient.get("dob")
            if dob:
                lines.append(f"• Date of birth: {dob}")
                shown = True
        elif field == "gender":
            gender = patient.get("gender")
            if gender:
                lines.append(f"• Gender: {gender}")
                shown = True
            # Age is derived from the stored date of birth, never invented.
            dob = patient.get("dob")
            if dob:
                try:
                    y, m, _ = dob.split("-")
                    today = date.today()
                    age = today.year - int(y) - (
                        (today.month, today.day) < (int(m), int(_))
                    )
                    lines.append(f"• Age: {age}")
                    shown = True
                except ValueError:
                    pass
        elif field in _UNSTORED_LABELS:
            unstored.append(_UNSTORED_LABELS[field])
        else:
            unstored.append(field)

    if unstored:
        labels = ", ".join(unstored)
        lines.append(
            f"{patient['name']} ({patient['id']}) has no {labels} recorded in "
            "this system, so I can't provide it — I won't guess."
        )

    if not shown and not unstored:
        labels = "/".join(_PATIENT_INFO_LABELS.get(f, f) for f in requested)
        return (
            f"{patient['name']} ({patient['id']}) has no {labels} details on file."
        )
    return "\n".join(lines)


# ── Query (NLU + Retrieval) ──────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """
    Accept a free-text query. Use LLM to interpret intent, then deterministically
    retrieve the correct record. Handles ambiguity and missing records gracefully.
    """
    try:
        interpreted = interpret_query(req.text, req.session_context)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM service error: {exc}")

    intent = interpreted.intent
    patient_name = interpreted.patient_name
    report_type = interpreted.report_type
    fallback_reason = interpreted.fallback_reason

    matched_patients: list[PatientSummary] = []
    report_detail: ReportDetail | None = None
    ambiguous = False
    message = ""
    # True when `message` itself already explains the provider outage, so the
    # transparency note at the end would only repeat it.
    explains_outage = intent in ("unsupported", "unknown")

    if intent == "greeting":
        message = GREETING_MESSAGE
    elif intent == "assistant_identity":
        # Truthful, canned answer. Replies never name the underlying AI provider.
        message = ASSISTANT_IDENTITY_MESSAGE
    elif intent == "assistant_capabilities":
        # "What can you do?" gets the implemented feature list — its own answer,
        # never the identity reply.
        message = CAPABILITIES_MESSAGE
    elif intent == "about_app":
        # "What is Viscan AI?" is about the application, not about the assistant.
        message = ABOUT_APP_MESSAGE
    elif intent == "small_talk":
        # Harmless conversation (jokes, thanks, farewells) stays conversation:
        # it is never dressed up as a radiology request.
        message = SMALL_TALK_MESSAGE
    elif intent == "general_dental":
        # General dental knowledge: educational information produced by the
        # configured AI, clearly separated from patient-specific findings and
        # never a diagnosis.
        message = answer_general_dental(req.text, provider_error=fallback_reason)
        explains_outage = explains_outage or bool(fallback_reason)
    elif intent == "unavailable_action":
        # Honest capability report: say the action is not implemented and that
        # nothing happened — never claim an action was completed.
        message = UNAVAILABLE_ACTION_MESSAGE.format(action=_truncate(req.text, 160))
    elif intent in ("unsupported", "unknown"):
        # A recognised request the app deliberately does not serve (a safety
        # refusal, or a read-only lookup the records cannot answer) is answered
        # verbatim; otherwise the LLM outage is explained instead of pretending
        # the message was an unrecognised report command.
        if interpreted.limitation:
            message = interpreted.limitation
            explains_outage = True
        else:
            message = LLM_FALLBACK_MESSAGE if fallback_reason else UNSUPPORTED_MESSAGE
    elif intent == "report_count":
        # Read-only count of a patient's reports, computed from the records.
        # Never a report-opening request: nothing is fetched, opened or implied.
        if not patient_name:
            message = (
                "I need a patient name to count their reports. Try something like "
                "\"How many reports does Priya Sharma have?\"."
            )
        else:
            status, matches = resolve_patient_name(patient_name)

            def _summaries(cs: list[dict]) -> list[PatientSummary]:
                counts = _patient_report_counts()
                summaries = []
                for c in cs:
                    clean = {k: v for k, v in c.items() if not k.startswith("_")}
                    summaries.append(
                        PatientSummary(**{**clean, "report_count": counts.get(clean.get("id"), 0)})
                    )
                return summaries

            if status == "none":
                from services.retrieval import _load_patients
                known = ", ".join(p["name"] for p in _load_patients())
                message = (
                    f"No patient found matching '{patient_name}' in this clinic's records. "
                    f"I only count patients that actually exist, so I won't invent one. "
                    f"Known patients: {known}. Please check the name and try again."
                )
            elif status == "ambiguous":
                ambiguous = True
                matched_patients = _summaries(matches[:4])
                message = (
                    f"Multiple patients match '{patient_name}'. "
                    "Please select the correct patient from the list."
                )
            else:
                patient = matches[0]
                matched_patients = _summaries([patient])
                total = len(get_reports_for_patient(patient["id"], report_type))
                if report_type:
                    noun = f"{report_type} report" if report_type else "report"
                    if total == 0:
                        message = f"{patient['name']} ({patient['id']}) has no {noun}s on file."
                    else:
                        message = f"{patient['name']} ({patient['id']}) has {total} {noun}s on file."
                else:
                    if total == 0:
                        message = f"{patient['name']} ({patient['id']}) has no reports on file."
                    else:
                        message = f"{patient['name']} ({patient['id']}) has {total} reports on file."
    elif intent == "retrieve_report":
        if not patient_name:
            message = "I couldn't identify a patient name in your request. Please specify the patient's name."
        else:
            # The interpreter's extracted name is the search intent; the
            # resolver only validates/resolves it against real records
            # (exact full name → exact first name → exact last name → fuzzy).
            status, matches = resolve_patient_name(patient_name)

            def _summaries(cs: list[dict]) -> list[PatientSummary]:
                counts = _patient_report_counts()
                summaries = []
                for c in cs:
                    clean = {k: v for k, v in c.items() if not k.startswith("_")}
                    summaries.append(
                        PatientSummary(**{**clean, "report_count": counts.get(clean.get("id"), 0)})
                    )
                return summaries

            if status == "none":
                from services.retrieval import _load_patients
                known = ", ".join(p["name"] for p in _load_patients())
                message = (
                    f"No patient found matching '{patient_name}' in this clinic's records. "
                    f"I only retrieve patients that actually exist, so I won't invent one. "
                    f"Known patients: {known}. Please check the name and try again."
                )
            elif status == "ambiguous":
                # Only reached when one matching priority genuinely matched 2+
                # patients, never a silent pick, never a wrong-patient guess.
                ambiguous = True
                matched_patients = _summaries(matches[:4])
                message = (
                    f"Multiple patients match '{patient_name}'. "
                    "Please select the correct patient from the list."
                )
            else:
                # Unambiguous match at the highest matching priority
                patient = matches[0]
                matched_patients = _summaries([patient])
                report = get_latest_report_for_patient(patient["id"], report_type)
                if report is None:
                    rtype_str = f" of type '{report_type}'" if report_type else ""
                    message = f"No reports{rtype_str} found for {patient['name']}."
                else:
                    report_detail = _build_report_detail(report)
                    message = (
                        f"Showing the latest {report_detail.type} report for {patient['name']} "
                        f"dated {report_detail.date}."
                    )

    elif intent == "patient_info":
        patient, summaries, is_ambiguous, info_msg = _resolve_patient_for_info(
            req, patient_name
        )
        if patient:
            message = _format_patient_info(patient, interpreted.info_fields)
        else:
            ambiguous = is_ambiguous
            matched_patients = summaries or []
            message = info_msg

    elif intent == "dictation":
        message = "Dictation mode: please submit your transcript via the /dictation endpoint."

    elif intent == "follow_up":
        if not req.session_context or not req.session_context.get("active_report_id"):
            message = "No active report in session. Please open a report first before asking follow-up questions."
        else:
            report = get_report_by_id(req.session_context["active_report_id"])
            if report:
                report_detail = _build_report_detail(report)
                message = "Use /follow-up to ask questions about the active report."
            else:
                message = "Session report not found. Please open a report first."
    else:
        # Defensive: the intent validator guarantees one of the branches above
        # matched; never surface a raw intent at the user.
        message = UNSUPPORTED_MESSAGE

    # Transparency note, except where the message already explains the outage.
    if fallback_reason and not explains_outage:
        message = f"{message} (Note: {fallback_reason})"

    return QueryResponse(
        intent=intent,
        patient_name=patient_name,
        report_type=report_type,
        matched_patients=matched_patients,
        report=report_detail,
        ambiguous=ambiguous,
        demo_mode=DEMO_MODE,
        message=message,
        fallback_reason=fallback_reason,
        limitation=interpreted.limitation,
    )


# ── Dictation ────────────────────────────────────────────────────────────────

@app.post("/dictation", response_model=DictationResponse)
def dictation_endpoint(req: DictationRequest):
    """
    Accept a clinical dictation transcript and structure it.
    Never invents findings. Always requires clinician review.
    """
    try:
        structured = structure_dictation(req.transcript)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Dictation service error: {exc}")

    return DictationResponse(
        findings=structured.findings,
        anatomical_site=structured.anatomical_site,
        draft_report=structured.draft_report,
        uncertainties=structured.uncertainties,
        requires_review=True,
        demo_mode=DEMO_MODE,
        fallback_reason=structured.fallback_reason,
        metadata=structured.metadata,
        transcript=structured.transcript,
        usable=structured.usable,
    )


# ── Follow-up ────────────────────────────────────────────────────────────────

@app.post("/follow-up", response_model=FollowUpResponse)
def follow_up_endpoint(req: FollowUpRequest):
    """
    Answer a follow-up question about a specific report using session context.
    Always shows source evidence.
    """
    report = get_report_by_id(req.report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report '{req.report_id}' not found")

    try:
        result = answer_follow_up(req.question, report)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Follow-up service error: {exc}")

    answer = result.get("answer", "")
    evidence = result.get("evidence", [])
    # Defensive validation of model output before it reaches the response schema
    if not isinstance(answer, str):
        answer = str(answer)
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    evidence = [str(e) for e in evidence]

    return FollowUpResponse(
        answer=answer,
        source_report_id=result.get("source_report_id", req.report_id),
        evidence=evidence,
        demo_mode=DEMO_MODE,
    )
