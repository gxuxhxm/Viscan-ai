from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from models import (
    QueryRequest, QueryResponse,
    DictationRequest, DictationResponse,
    FollowUpRequest, FollowUpResponse,
    PatientSummary, ReportDetail, FindingItem, ReportSummary,
)
from services.retrieval import (
    search_patients_by_name,
    get_patient_by_id,
    get_latest_report_for_patient,
    get_reports_for_patient,
    get_report_by_id,
)
from services.llm_service import (
    interpret_query,
    structure_dictation,
    answer_follow_up,
    DEMO_MODE,
    LLM_PROVIDER,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Viscan AI Backend",
    description="Dental radiology assistant API — take-home POC",
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
    )


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "demo_mode": DEMO_MODE, "llm_provider": LLM_PROVIDER}


# ── Patients ─────────────────────────────────────────────────────────────────

@app.get("/patients", response_model=list[PatientSummary])
def list_patients():
    from services.retrieval import _load_patients
    return [PatientSummary(**p) for p in _load_patients()]


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
        )
        for r in reports
    ]


@app.get("/reports/{report_id}", response_model=ReportDetail)
def get_report(report_id: str):
    report = get_report_by_id(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")
    return _build_report_detail(report)


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

    matched_patients: list[PatientSummary] = []
    report_detail: ReportDetail | None = None
    ambiguous = False
    message = ""

    if intent == "retrieve_report":
        if not patient_name:
            message = "I couldn't identify a patient name in your request. Please specify the patient's name."
        else:
            candidates = search_patients_by_name(patient_name)
            exact_matches = [c for c in candidates if c.get("_exact")]

            def _summaries(cs: list[dict]) -> list[PatientSummary]:
                return [
                    PatientSummary(**{k: v for k, v in c.items() if not k.startswith("_")})
                    for c in cs
                ]

            if len(candidates) == 0:
                message = f"No patient found matching '{patient_name}'. Please check the name and try again."
            elif len(exact_matches) == 1:
                # Deterministic business rule: an exact full-name match resolves
                # to that patient, even when a similar name also scores high.
                patient = exact_matches[0]
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
            elif len(exact_matches) > 1 or (
                # Ambiguous if: top score is not a clear winner (gap < 0.20) AND multiple match
                len(candidates) >= 2
                and candidates[0]["_score"] - candidates[1]["_score"] < 0.20
            ):
                ambiguous = True
                matched_patients = _summaries(candidates[:4])
                message = (
                    f"Multiple patients match '{patient_name}'. "
                    "Please select the correct patient from the list."
                )
            else:
                # Unambiguous match
                patient = candidates[0]
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
        message = "I didn't understand your request. Try: 'Open Priya's latest OPG report' or 'Show Rohan's reports'."

    if interpreted.fallback_reason:
        message = f"{message} (Note: {interpreted.fallback_reason})"

    return QueryResponse(
        intent=intent,
        patient_name=patient_name,
        report_type=report_type,
        matched_patients=matched_patients,
        report=report_detail,
        ambiguous=ambiguous,
        demo_mode=DEMO_MODE,
        message=message,
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
