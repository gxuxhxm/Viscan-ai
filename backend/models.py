from __future__ import annotations
from typing import Optional, Any
from pydantic import BaseModel, field_validator


# ── Request Models ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    session_context: Optional[dict] = None

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query text cannot be empty")
        if len(v) > 2000:
            raise ValueError("Query text too long (max 2000 chars)")
        return v


class DictationRequest(BaseModel):
    transcript: str
    patient_id: Optional[str] = None
    report_id: Optional[str] = None

    @field_validator("transcript")
    @classmethod
    def transcript_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Transcript cannot be empty")
        if len(v) > 10000:
            raise ValueError("Transcript too long (max 10000 chars)")
        return v


class FollowUpRequest(BaseModel):
    question: str
    report_id: str
    session_id: Optional[str] = None

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        return v

    @field_validator("report_id")
    @classmethod
    def report_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("report_id required for follow-up")
        return v


# ── Response Models ─────────────────────────────────────────────────────────

class PatientSummary(BaseModel):
    id: str
    name: str
    dob: str
    gender: str

class ReportSummary(BaseModel):
    id: str
    patient_id: str
    type: str
    date: str
    radiologist: str
    status: str
    impression: str

class FindingItem(BaseModel):
    site: str
    observation: str
    severity: str

class ReportDetail(BaseModel):
    id: str
    patient_id: str
    type: str
    date: str
    radiologist: str
    status: str
    findings: list[FindingItem]
    impression: str
    draft_report: str

class QueryResponse(BaseModel):
    intent: str
    patient_name: Optional[str]
    report_type: Optional[str]
    matched_patients: list[PatientSummary]
    report: Optional[ReportDetail]
    ambiguous: bool
    demo_mode: bool
    message: str

class DictationResponse(BaseModel):
    findings: list[Any]
    anatomical_site: str
    draft_report: str
    uncertainties: list[str]
    requires_review: bool
    demo_mode: bool

class FollowUpResponse(BaseModel):
    answer: str
    source_report_id: Optional[str]
    evidence: list[str]
    demo_mode: bool
