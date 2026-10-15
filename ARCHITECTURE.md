# Viscan AI: Architecture Document

## System Overview

Viscan AI is a dental reporting AI. It separates LLM interpretation from deterministic data access, validates all AI outputs with Pydantic, deterministically re-checks model output against the source text/report, and provides a fully functional demo mode without API keys. Natural-language understanding, dictation structuring and follow-up summaries run on **Google Gemini** when `GEMINI_API_KEY` is configured.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER (React + Vite)                   │
│                                                                 │
│  ┌───────────┐    ┌──────────────────┐    ┌─────────────────┐  │
│  │  Patient  │    │  Report Viewer   │    │  Conversation   │  │
│  │  Sidebar  │    │  + Dictation     │    │  Panel (AI)     │  │
│  └─────┬─────┘    └────────┬─────────┘    └────────┬────────┘  │
│        │                   │                        │           │
│        └───────────────────┼────────────────────────┘           │
│                            │  api.js (fetch)                    │
└────────────────────────────┼────────────────────────────────────┘
                             │ HTTP (CORS)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Python)                      │
│                                                                 │
│   /query          /dictation      /follow-up /patients /reports │
│       │                │               │            │           │
│       ▼                ▼               ▼            ▼           │
│  ┌────────────────────────────────┐  ┌─────────────────────┐   │
│  │       llm_service.py           │  │    retrieval.py      │   │
│  │                                │  │                      │   │
│  │  interpret_query()             │  │  search_patients_    │   │
│  │  structure_dictation()         │  │  by_name()           │   │
│  │  answer_follow_up()            │  │  get_latest_report() │   │
│  │                                │  │  get_report_by_id()  │   │
│  │  [Demo mode: heuristics]       │  │                      │   │
│  │  [Live mode: Gemini API]       │  │  [Pure Python]       │   │
│  └────────────┬───────────────────┘  └──────────┬──────────┘   │
│               │                                  │              │
│               │  Pydantic validation              │              │
│               ▼                                  ▼              │
│         models.py                         data/ (JSON)          │
│    (InterpretedQuery,                   patients.json           │
│     StructuredDictation,               reports.json            │
│     QueryResponse, etc.)                                        │
└─────────────────────────────────────────────────────────────────┘
                             │
                      ┌──────┴──────┐
                      │  Gemini API │  (when GEMINI_API_KEY set)
                      │  flash-tier │
                      └─────────────┘
```

---

## Data Flow

### 1. Patient Report Retrieval

```
User types: "Open Priya's latest OPG report"
     │
     ▼ POST /query
llm_service.interpret_query(text)
     │
     ├─ Live: Gemini → JSON intent (greeting / assistant_identity /
     │         assistant_capabilities / about_app / small_talk /
     │         general_dental / patient_info / unavailable_action /
     │         retrieve_report / dictation / follow_up / unsupported)
     └─ Demo: regex heuristic → intent
     │
     ▼ InterpretedQuery validated by Pydantic
     │  { intent: "retrieve_report", patient_name: "Priya", report_type: "OPG" }
     │
     ▼ retrieval.resolve_patient_name("Priya")
     │  priority order, the first tier with a hit decides:
     │   1. exact full name  → no match for "Priya"
     │   2. exact first name → Priya Sharma (P001), unique → wins;
     │      "Priyanka Sharma" only shares the prefix, it is not a match
     │   3. exact last name  → not reached (tier 2 already decided)
     │   4. fuzzy/partial    → search_patients_by_name(), score gap < 0.20
     │
     ├─ one patient at that tier → get_latest_report_for_patient(P001, "OPG")
     ├─ 2+ patients at that tier → return ambiguity flag + candidate list
     └─ no match                 → return helpful error message
     │
     ▼ QueryResponse → frontend
```

### 2. Clinical Dictation

```
Clinician types transcript
     │
     ▼ POST /dictation
llm_service.structure_dictation(transcript)
     │
     ├─ header/metadata lines split off first
     ├─ no clinical vocabulary in the text (noise / unrelated question)
     │    → unusable result: no findings, draft_report = "", usable: false
     │      (also returned when the model itself answers "usable": false)
     ├─ Live: Gemini → structured JSON
     └─ Demo: minimal passthrough
     │
     ▼ StructuredDictation validated by Pydantic
     │  { findings: [...], anatomical_site: ..., draft_report: ...,
     │    uncertainties: [...], usable: ..., requires_review: true }  ← always forced
     │
     ▼ DictationResponse → frontend (editable draft, review banner)
        usable: true  → structured findings + editable draft + Save Report
        usable: false → "Nothing to structure" notice, no draft, no Save button
```

### 3. Conversational Follow-up

```
User asks: "Summarize her report"   (a report is open in the UI)
     │
     ▼ POST /query { question, session_context: { active_report_id } }
     │  interpret_query() classifies intent = follow_up
     │  retrieval.get_report_by_id(active_report_id)  ← deterministic, no LLM
     ▼ QueryResponse { intent: follow_up, report: <active report> }
     │
     ▼ Frontend then calls POST /follow-up { question, report_id }
retrieval.get_report_by_id(report_id)  ← deterministic, no LLM
     │
     ▼ llm_service.answer_follow_up(question, report)
     │  System prompt includes ONLY the report content
     │  LLM constrained to answer only from that context
     │
     ▼ FollowUpResponse { answer, evidence: [...], source_report_id }
     │  evidence list shown in UI alongside answer
     │
     ▼ Frontend displays answer + collapsible evidence block

No open report → /query returns intent=follow_up with guidance to open a
report first (no context is invented).
```

### 4. Saving an Edited Dictation Report

```
Clinician edits the draft, clicks Save Report, confirms clinical review
     │
     ▼ POST /reports { draft_report, patient_id | patient, findings,
     │                 transcript, clinician_confirmed: true, … }
main.save_report() → Pydantic validates the request
     │
     ├─ patient: an existing id is verified …
     ├─ …or a new patient (manual form or validated JSON input in the save
     │    dialog — same NewPatientInput schema) is appended to patients.json
     │    (P###); an exact identity match reuses the id already on disk, so a
     │    retry never appends a duplicate patient, and a caller-supplied id is
     │    rejected rather than trusted
     ├─ report record built in the existing schema (R###, next id):
     │    clinician_confirmed: explicit status, no clinician identity invented
     │    status: final when confirmed, draft otherwise
     │    anatomical_site: the dictated primary region (additive, optional)
     │    source_transcript + dictation provenance preserved
     ├─ identical repeat submission → returns the already-stored report
     └─ retrieval.add_report_if_new(): locked, atomic, format-preserving append
     │
     ▼ ReportDetail → retrievable via /reports/{id}, /patients/{id}/reports
        and conversational /query retrieval
```

---

## Key Safety Properties

| Property | Mechanism |
|---|---|
| No fabricated findings | LLM system prompt explicitly forbids invention; Pydantic validates output |
| No drafts from unusable input | Deterministic clinical-vocabulary gate runs before any LLM call; the model can also declare a transcript `usable: false`. Either verdict returns empty findings, an empty `draft_report` and `usable: false`, which the UI renders as a "Nothing to structure" notice with no Save action |
| Always requires review | `requires_review` validator always returns `True` in StructuredDictation |
| No silent confirmation | A dictation report is only marked clinician-confirmed when the user explicitly picks "Yes, Confirm & Save"; the confirmation stores a status field only; no clinician identity or signature is ever invented |
| Source evidence shown | Every follow-up answer includes the exact finding text used |
| Graceful LLM failure | Provider errors and malformed/non-JSON LLM output fall back to deterministic parsers (surfaced via `fallback_reason` / uncertainties), never a 500 |
| Provider-free identity | Greeting and identity replies ("Are you Gemini?", "Who are you?") are canned deterministic text that describe Viscan AI and never name the underlying AI provider |
| Natural phrasing via the configured AI | Greetings, identity questions, retrieval requests and contextual follow-ups are classified by the configured AI model, not a hardcoded keyword list; deterministic code is only used for real report retrieval, safety checks, and a minimal demo/fallback path |
| Ambiguity surfaced | The AI-extracted name is resolved by priority (exact full name → exact first name → exact last name → fuzzy). Ambiguity is raised only when one tier genuinely matches 2+ patients (shared first or last name, or fuzzy candidates within a 0.20 gap), never a silent selection, never a prefix collision like "Priya" vs "Priyanka" |
| No real patient data | All 12 reports are fully synthetic; no PII involved |

---

## Component Responsibilities

| Component | Responsibility |
|---|---|
| `retrieval.py` | Read JSON, resolve the AI-extracted patient name by priority (exact full → first → last → fuzzy, `resolve_patient_name`), return typed dicts; append new patients/reports with locked, atomic, format-preserving writes. **No LLM.** |
| `llm_service.py` | Build prompts, call OpenAI, validate output, deterministic fallbacks on provider/malformed-output failure. **No data access.** |
| `main.py` | Route HTTP requests, orchestrate retrieval + LLM, return validated responses |
| `models.py` | Pydantic schemas for all inputs/outputs |
| `App.jsx` | 3-panel UI state machine; calls `api.js`, renders results |
| `api.js` | Thin fetch wrapper; no business logic |

---

## Future Extension Points

- **Audio/STT**: Replace textarea in dictation with `MediaRecorder` → Whisper API → same `/dictation` endpoint
- **DICOM viewer**: Embed `cornerstone.js` in report detail panel
- **Auth**: Add FastAPI `Depends(verify_token)` to all routes
- **Database**: Replace JSON files with PostgreSQL + SQLAlchemy
- **Streaming**: Switch follow-up to SSE/WebSocket for token streaming
