# Viscan AI — Architecture Document

## System Overview

Viscan AI is a dental radiology assistant POC. It separates LLM interpretation from deterministic data access, validates all AI outputs with Pydantic, deterministically re-checks model output against the source text/report, and provides a fully functional demo mode without API keys. Natural-language understanding, dictation structuring and follow-up summaries run on **Google Gemini** when `GEMINI_API_KEY` is configured.

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
│   /query          /dictation      /follow-up    /patients       │
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
     │         retrieve_report / dictation / follow_up / unsupported)
     └─ Demo: regex heuristic → intent
     │
     ▼ InterpretedQuery validated by Pydantic
     │  { intent: "retrieve_report", patient_name: "Priya", report_type: "OPG" }
     │
     ▼ retrieval.search_patients_by_name("Priya")
     │  token/prefix similarity → [Priya Sharma (0.856), Priyanka Sharma (0.825)]
     │  exact full-name matches are flagged (_exact=True)
     │
     ├─ exact full-name match (1) → get_latest_report_for_patient(P001, "OPG")
     ├─ 1 clear unambiguous match  → get_latest_report_for_patient(...)
     ├─ 2+ matches within 0.20 gap → return ambiguity flag + candidate list
     └─ 0 matches → return helpful error message
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
     ├─ Live: Gemini → structured JSON
     └─ Demo: minimal passthrough
     │
     ▼ StructuredDictation validated by Pydantic
     │  { findings: [...], anatomical_site: ..., draft_report: ...,
     │    uncertainties: [...], requires_review: true }  ← always forced
     │
     ▼ DictationResponse → frontend (editable draft, review banner)
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

---

## Key Safety Properties

| Property | Mechanism |
|---|---|
| No fabricated findings | LLM system prompt explicitly forbids invention; Pydantic validates output |
| Always requires review | `requires_review` validator always returns `True` in StructuredDictation |
| Source evidence shown | Every follow-up answer includes the exact finding text used |
| Graceful LLM failure | Provider errors and malformed/non-JSON LLM output fall back to deterministic parsers (surfaced via `fallback_reason` / uncertainties), never a 500 |
| Honest assistant identity | Greeting and "Are you Gemini?" replies are canned deterministic text — the app states it *uses* the Gemini API and never claims to be Gemini |
| Natural phrasing via Gemini | Greetings, identity questions, retrieval requests and contextual follow-ups are classified by Gemini, not a hardcoded keyword list; deterministic code is only used for real report retrieval, safety checks, and a minimal demo/fallback path |
| Ambiguity surfaced | Exact full names resolve deterministically; partial names matching 2+ patients within a 0.20 gap trigger disambiguation, not silent selection |
| No real patient data | All 10 reports are fully synthetic; no PII involved |

---

## Component Responsibilities

| Component | Responsibility |
|---|---|
| `retrieval.py` | Read JSON, fuzzy-match names (exact full-name matches flagged), return typed dicts. **No LLM.** |
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
