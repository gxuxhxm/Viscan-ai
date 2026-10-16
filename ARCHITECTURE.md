# Viscan AI — Architecture

## Overview

Viscan AI is a three-panel dental reporting app (React + Vite) talking to a FastAPI backend. Its rule: **the LLM only interprets and structures text — it never touches patient data.** All record reads/writes go through deterministic Python functions over local JSON, every model output is Pydantic-validated and then re-checked against the source text, and with no API key the same pipeline runs on deterministic heuristics (demo mode).

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
│   │         llm_service.py        │   │        retrieval.py       │    │
│   │                               │   │                           │    │
│   │  interpret_query()            │   │  resolve_patient_name()   │    │
│   │  structure_dictation()        │   │  get_latest_report()      │    │
│   │  answer_follow_up()           │   │  get_report_by_id()       │    │
│   │                               │   │  add_report_if_new()      │    │
│   │  live: Gemini ▸ OpenAI        │   │  Pure Python · no LLM     │    │
│   │  demo: regex heuristics       │   │                           │    │
│   └──────────────┬────────────────┘   └─────────────┬─────────────┘    │
│                  │ Pydantic validation              │ JSON r/w         │
│                  ▼                                  ▼                  │
│             models.py                         data/ (JSON)             │
│        (InterpretedQuery,                    patients.json             │
│         StructuredDictation,                 reports.json              │
│         QueryResponse, etc.)                                           │
└─────────────────────────────────┴──────────────────────────────────────┘
                                  │
                         ┌────────┴────────┐
                         │    Gemini API   │  (when GEMINI_API_KEY is set)
                         │    flash-tier   │
                         └─────────────────┘
```

## Flows

### 1. Patient report retrieval

```
User types/speaks: "Open Priya's latest OPG report"
     │
     ▼ POST /query
llm_service.interpret_query(text)
     ├─ live: Gemini ▸ OpenAI → JSON intent + patient_name + report_type
     └─ demo: regex heuristic   (provider error / malformed output →
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
     └─ live: Gemini ▸ demo heuristics → structured JSON
     ▼ Pydantic StructuredDictation { findings[], anatomical_site, draft_report,
     │    uncertainties[] }  —  requires_review is ALWAYS forced true
     ▼ grounding: only findings verbatim in the transcript survive; unsupported
        severities are scrubbed, dropped findings flagged
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
     ▼ FollowUpResponse { answer, evidence: [...], source_report_id }
       evidence re-checked against the report → shown next to the answer
     ▼ no open report → guided reply, no invented context
```

## Safety properties

| Property | Mechanism |
|---|---|
| No fabricated findings | Prompt forbids invention + transcript grounding check drops anything not verbatim present |
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
| `backend/services/llm_service.py` | Prompts, Gemini/OpenAI calls, demo fallbacks, grounding checks — **no data access** |
| `backend/services/retrieval.py` | JSON reads/writes, name resolution — **no LLM** |
| `frontend/src/App.jsx` | 3-panel UI state machine |
| `frontend/src/api.js` · `useSpeechRecognition.js` | Fetch client (`localhost:8000`) · Web Speech API input |
