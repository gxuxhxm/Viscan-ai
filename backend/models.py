from __future__ import annotations
import re
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# Report modalities used across the existing reports.json records.
REPORT_TYPES = ("OPG", "PERIAPICAL", "BITEWING", "CBCT")


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
    # Total reports on file for the patient (0 for a newly created patient).
    report_count: int = 0
    # Contact / demographic details. Optional (older records stay valid) and
    # None when the record genuinely has no value on file; never invented.
    phone: Optional[str] = None
    email: Optional[str] = None

class BillingCodes(BaseModel):
    """Proposed (NOT verified) billing codes attached to a synthetic report."""
    icd_10: Optional[str] = None
    cdt_proposed: Optional[str] = None

class ReportSummary(BaseModel):
    id: str
    patient_id: str
    type: str
    date: str
    radiologist: str
    status: str
    impression: str
    # Additive synthetic metadata (all optional; existing records stay valid)
    tooth_number: Optional[str] = None
    encounter_type: Optional[str] = None
    procedure_date: Optional[str] = None
    billing_codes: Optional[BillingCodes] = None
    clinician_confirmed: Optional[bool] = None   # explicit review status (None → not set)

class FindingItem(BaseModel):
    site: str
    observation: str
    severity: str
    # Optional additive fields. Report findings may carry them; dictation
    # findings (transient, clinician-dictated) typically omit them.
    complication_detected: Optional[bool] = None
    category: Optional[str] = None
    resolution_plan: Optional[str] = None
    # "historical" | "current", set by follow-up dictations so prior-study
    # findings stay separate from findings about the current examination.
    context: Optional[str] = None

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
    # Additive synthetic metadata (all optional)
    tooth_number: Optional[str] = None
    encounter_type: Optional[str] = None
    procedure_date: Optional[str] = None
    billing_codes: Optional[BillingCodes] = None
    # Primary region of the dictation ("tooth 26", "right maxillary sinus").
    # Additive optional field: records written before it existed stay valid.
    anatomical_site: Optional[str] = None
    # Additive dictation-save fields (all optional; existing records stay valid)
    clinician_confirmed: Optional[bool] = None   # explicit review status; no clinician identity stored
    source_transcript: Optional[str] = None      # original dictated transcript, preserved verbatim
    provenance: Optional[dict[str, Any]] = None  # how the draft was produced (dictation demo/fallback/metadata)

class QueryResponse(BaseModel):
    intent: str
    patient_name: Optional[str]
    report_type: Optional[str]
    matched_patients: list[PatientSummary]
    report: Optional[ReportDetail]
    ambiguous: bool
    demo_mode: bool
    message: str
    # Transparency / provenance (additive): None → interpretation came from
    # the LLM (or demo heuristics); set → the LLM failed and deterministic
    # fallback interpretation was used.
    fallback_reason: Optional[str] = None

class DictationResponse(BaseModel):
    findings: list[Any]
    anatomical_site: str
    draft_report: str
    uncertainties: list[str]
    requires_review: bool
    demo_mode: bool
    # Transparency / provenance (additive; review gate is never bypassed)
    fallback_reason: Optional[str] = None   # None → AI-structured; set → deterministic fallback
    metadata: dict[str, str] = {}           # header lines stripped before structuring
    transcript: str = ""                    # original transcript, always echoed back
    # False → the transcript held no usable clinical content: no draft was
    # produced (draft_report is empty) and the client must not offer a save.
    usable: bool = True

class FollowUpResponse(BaseModel):
    answer: str
    source_report_id: Optional[str]
    evidence: list[str]
    demo_mode: bool


# ── Report save (edited dictation draft → persisted record) ────────────────

class NewPatientInput(BaseModel):
    """
    New patient created alongside a saved report. Mirrors the existing
    patients.json schema, where ``id`` is generated server-side.

    Unknown keys are rejected instead of dropped silently: a caller-supplied
    ``id`` (or any field this schema does not carry) must never be mistaken
    for stored patient data.
    """
    model_config = ConfigDict(extra="forbid")

    name: str
    dob: str                      # YYYY-MM-DD, as stored for existing patients
    gender: str
    phone: Optional[str] = None
    email: Optional[str] = None

    @field_validator("name", "dob", "gender")
    @classmethod
    def required_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("cannot be empty")
        return v

    @field_validator("dob")
    @classmethod
    def dob_iso_date(cls, v: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError("dob must be an ISO date (YYYY-MM-DD)")
        return v

    @field_validator("phone", "email")
    @classmethod
    def optional_text(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None


class SaveReportRequest(BaseModel):
    """
    An edited dictation draft to persist as a report record, in the same
    JSON shape as the existing reports.json entries.

    Exactly one of ``patient_id`` (existing patient) or ``patient`` (new)
    must be given. ``clinician_confirmed`` is the explicit confirmation
    status; the API never invents a clinician identity or signature.
    """
    draft_report: str
    patient_id: Optional[str] = None
    patient: Optional[NewPatientInput] = None
    report_type: Optional[str] = None                 # defaults to a type detected in the text
    findings: list[FindingItem] = []
    impression: Optional[str] = None                  # derived from the draft when omitted
    transcript: Optional[str] = None                  # original dictated transcript
    anatomical_site: Optional[str] = None
    tooth_number: Optional[str] = None
    radiologist: Optional[str] = None
    clinician_confirmed: bool = False                 # must be explicitly requested by the caller
    # Dictation provenance (pass-through of the /dictation response)
    demo_mode: Optional[bool] = None
    fallback_reason: Optional[str] = None
    dictation_metadata: Optional[dict[str, str]] = None

    @field_validator("draft_report")
    @classmethod
    def draft_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("draft_report cannot be empty")
        if len(v) > 100_000:
            raise ValueError("draft_report too long (max 100000 chars)")
        return v

    @field_validator("patient_id", "tooth_number", "radiologist", "anatomical_site", "transcript", "impression")
    @classmethod
    def blank_to_none(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None

    @field_validator("report_type")
    @classmethod
    def known_report_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().upper()
        if v not in REPORT_TYPES:
            raise ValueError(f"report_type must be one of {', '.join(REPORT_TYPES)}")
        return v

    @model_validator(mode="after")
    def exactly_one_patient(self):
        if bool(self.patient_id) == bool(self.patient):
            raise ValueError("Provide exactly one of patient_id (existing) or patient (new)")
        return self
