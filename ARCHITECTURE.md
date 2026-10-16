# Viscan AI — Architecture

## Overview

Viscan AI is a three-panel dental reporting app (React + Vite) talking to a FastAPI backend, engineered around a single pipeline: **LLM interpretation → Pydantic schema validation → deterministic retrieval & business rules → grounded response.** Google Gemini (OpenAI as the configured alternative) does the interpreting — intent recognition, patient-name/report-type entity extraction, clinical-dictation structuring and report-aware question answering — while deterministic code owns everything that must be exact.

The data boundary is explicit: **the LLM never has direct access to the data store.** Every record read/write goes through deterministic Python functions over local JSON (`services/retrieval.py`), every model output is Pydantic-validated and then re-checked against the source text, and the only report content that leaves the backend is prompt text the backend itself builds — for grounded follow-up answers, the selected report's relevant content (ids, findings, impression), with the answer's evidence re-verified against the stored report before it reaches the UI. With no API key configured the same pipeline stages are served by deterministic parsers (reported as `demo_mode: true`) — a resilience and testing path that never claims AI output.

Every stage is covered by automated tests: **210 total** — 183 backend (181 offline tests that can never spend a paid API call, plus **2 live Gemini API-call tests** that exercise the real provider end to end) and 27 frontend tests.

## System diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                          BROWSER (React + Vite)                        │
│                                                                        │
│   ┌─────────────┐    ┌────────────────────┐    ┌────────────────────┐  │
│   │   Patient   │    │   Report Viewer    │    │   Conversation     │  │
│   │   Sidebar   │    │   + Dictation      │    │   Panel (AI)       │  │
│   └──────┬──────┘    └─────────┬──────────┘    └─────────┬──────────┘  │
│          │                     │                          │            │
│          └─────────────────────┼──────────────────────────┘            │
│                                │  api.js (fetch) · Web Speech API      │
└────────────────────────────────┼───────────────────────────────────────┘
                                 │ HTTP (CORS) → localhost:8000
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend (Python)                       │
│                                                                        │
│     /query          /dictation       /follow-up      /patients·reports │
│        │                │                 │                │           │
│        ▼                ▼                 ▼                ▼           │
│   ┌───────────────────────────────┐   ┌───────────────────────────┐    │
│   │  AI Orchestration &           │   │        retrieval.py       │    │
│   │  Structured Generation        │   │  (deterministic tools)    │    │
│   │      (llm_service.py)         │   │                           │    │
│   │                               │   │  resolve_patient_name()   │    │
│   │  interpret_query()            │   │  get_latest_report()      │    │
│   │  structure_dictation()        │   │  get_report_by_id()       │    │
│   │  answer_follow_up()           │   │  add_report_if_new()      │    │
│   │                               │   │                           │    │
│   │  live: Gemini ▸ OpenAI        │   │  Pure Python · no LLM     │    │
│   │  offline: deterministic       │   │  JSON read/write only     │    │
│   │             parsers           │   │                           │    │
│   └──────────────┬────────────────┘   └─────────────┬─────────────┘    │
│                  │ structured JSON                  │ JSON r/w         │
│                  ▼                                  ▼                  │
│             models.py                         data/ (JSON)             │
│        (InterpretedQuery,                    patients.json             │
│         StructuredDictation,                 reports.json              │
│         QueryResponse, etc.)                                           │
│                  │ Pydantic-validated model output                     │
│                  ▼ into grounding & safety validation                  │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │  GROUNDING & SAFETY VALIDATION (deterministic · always on)     │   │
│   │                                                                │   │
│   │  transcript grounding (token overlap) · severity downgrade     │   │
│   │  follow-up evidence re-check · invented-name drop · noise gate │   │
│   │  unsafe-request refusal · provider failure → offline fallback  │   │
│   │  requires_review = true · validated, grounded output → the UI  │   │
│   │                                                                │   │
│   └──────────────┴─────────────────────────────────────────────────┘   │
│                  │ grounded, evidence-labelled response                │
│                  ▲ (HTTP 200 → back to the React UI)                   │
└─────────────────────────────────┴──────────────────────────────────────┘
                                  │
                         ┌────────┴────────┐
                         │    Gemini API   │  (when GEMINI_API_KEY is set)
                         │    flash-tier   │
                         └─────────────────┘
```

The provider receives **prompt text only** — the user's message, or, for `/follow-up`, the selected report's relevant content that the backend chooses to put in the prompt. It has no credentials for, and no path to, `data/*.json`; all record I/O stays in `retrieval.py`.

Between model output and the UI sit two explicit stages: **Pydantic schema validation** (`models.py`) and the **Grounding & Safety Validation** box — transcript grounding (token overlap), severity downgrade, follow-up evidence re-check, invented-name and unsafe-request guards, provider-failure fallback, and the forced review gate. Only validated, grounded, labelled output is returned to the browser; this is verification of what the model produced, not a guarantee that a model can never be wrong.

## Flows

### 1. Patient report retrieval

```
User types/speaks: "Open Priya's latest OPG report"
     │
     ▼ POST /query
llm_service.interpret_query(text)
     ├─ live: Gemini ▸ OpenAI → JSON intent + patient_name + report_type
     └─ offline: deterministic parser   (provider error / malformed output →
          deterministic fallback, surfaced as fallback_reason — never a 500)
     ▼ Pydantic InterpretedQuery  (invalid intent counts as malformed output)
     │   guards: invented patient name dropped · unsafe request → unsupported
     ▼ retrieval.resolve_patient_name("Priya")        [pure Python, no LLM]
        1. exact full name  2. exact first  3. exact last  4. fuzzy (0.45)
     ├─ one patient at that tier → get_latest_report_for_patient(P001, "OPG")
     ├─ 2+ patients at that tier → ambiguous: true + candidates (UI asks)
     └─ no match                 → helpful message, never a fabricated record
     ▼ QueryResponse { intent, report, ambiguous, demo_mode, fallback_reason }
```

### 2. Voice / STT

```
Mic button (retrieval or dictation panel)
     ▼
useSpeechRecognition.js — Web Speech API, Chrome/Edge, en-IN, continuous
     ▼ speech ends → final transcript auto-submits (typed input always there)
POST /query  or  POST /dictation           ← same pipeline as typed text
     │
     └─ the backend never receives or stores audio — only transcript text
       (unsupported browsers simply get the typed-input path)
```

### 3. Clinical dictation → report generation

```
Clinician dictates or pastes a transcript
     │
     ▼ POST /dictation
structure_dictation(transcript)
     ├─ header/metadata lines split off first
     ├─ no clinical vocabulary → no findings, draft_report = "", usable: false
     │    (also when the model itself answers "usable": false)
     └─ live: Gemini · offline: deterministic parser → structured JSON
     ▼ Pydantic StructuredDictation { findings[], anatomical_site, draft_report,
     │    uncertainties[] }  —  requires_review is ALWAYS forced true
     ▼ grounding: findings are re-checked against the transcript (token
        overlap — unsupported ones dropped); unsupported severities are
        scrubbed, dropped findings flagged
     ▼ frontend:  usable: true  → editable draft + review banner + Save Report
                  usable: false → "nothing to structure", no draft, no Save
     ▼ clinician edits, explicitly confirms, POST /reports
        → reports.json (R###, status final|draft, source_transcript kept);
          a new patient gets P### — identity-checked, de-duplicated, atomic
```

### 4. Conversational follow-up

```
User asks: "Summarize her report"   (a report is open in the UI)
     │
     ▼ POST /query { question, session_context: { active_report_id } }
       intent = follow_up · get_report_by_id()  ← deterministic, no LLM
     ▼ frontend calls POST /follow-up { question, report_id }
       retrieval.get_report_by_id(report_id)  ← deterministic, no LLM
     ▼ llm_service.answer_follow_up(question, report)
       system prompt contains ONLY that report's content; answer only from it
       (that content is sent to the provider as prompt text — the LLM never
        reads the store itself; returned evidence is re-checked after)
     ▼ FollowUpResponse { answer, evidence: [...], source_report_id }
       evidence re-checked against the report → shown next to the answer
     ▼ no open report → guided reply, no invented context
```

## Safety properties

| Property | Mechanism |
|---|---|
| LLM has no data-store access | All record reads/writes stay in `retrieval.py`; the provider receives prompt text only (the user's message, or the selected report's relevant content for a grounded follow-up) |
| No fabricated findings | Prompt forbids invention + transcript grounding check (token overlap) drops unsupported findings and downgrades severities the dictation never stated |
| No drafts from noise input | Deterministic clinical gate before any LLM call; `usable: false` removes draft + Save UI |
| Always requires review | `requires_review` validator returns `true` unconditionally; save needs explicit confirmation |
| Evidence shown | Every follow-up answer lists the exact report text it used |
| Safe identity handling | Priority-tier name resolution; ambiguity surfaced, never guessed; invented names dropped |
| Graceful failure | Provider/malformed-output → deterministic fallback with `fallback_reason`, never a 500 |
| No real patient data | Fully synthetic fixtures; local-only reads and writes |

## Components

| Component | Responsibility |
|---|---|
| `backend/main.py` | Routes, orchestration, CORS, save-report validation |
| `backend/models.py` | Pydantic schemas for every input/output |
| `backend/services/llm_service.py` | Prompts, Gemini/OpenAI calls, offline deterministic fallbacks, grounding checks — **no data-store access** (it sends only prompt text to the provider) |
| `backend/services/retrieval.py` | JSON reads/writes, name resolution — **no LLM** |
| `frontend/src/App.jsx` | 3-panel UI state machine |
| `frontend/src/api.js` · `useSpeechRecognition.js` | Fetch client (`localhost:8000`) · Web Speech API input |
