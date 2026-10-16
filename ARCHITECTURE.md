# Viscan AI — Architecture (one page)

## Overview

Viscan AI is a three-panel dental reporting app (React + Vite) talking to a FastAPI backend. Its rule: **the LLM only interprets and structures text — it never touches patient data.** All record reads/writes go through deterministic Python functions over local JSON, every model output is Pydantic-validated and then re-checked against the source text, and with no API key the same pipeline runs on deterministic heuristics (demo mode), so the app is fully functional and testable with zero keys.

```
VOICE (browser)                    TEXT
Mic ─ Web Speech API ─► transcript ┘        ┌─ typed input ─┐
 (Chrome/Edge; typed fallback)              └───────────────┘
        │  POST /query · /dictation · /follow-up · /reports
        ▼
┌────────────────────────── FastAPI (localhost:8000) ──────────────────────────┐
│                                                                              │
│  LLM layer — services/llm_service.py  (text → JSON only, no data access)     │
│   interpret_query()   structure_dictation()   answer_follow_up()             │
│   live: Gemini gemini-3.5-flash-lite ▸ OpenAI gpt-4o-mini ▸ demo heuristics   │
│   ─► Pydantic schema (models.py) ─► grounding checks ─► fallback_reason      │
│                                                                              │
│  Tool/API layer — services/retrieval.py  (pure Python, no LLM)               │
│   resolve_patient_name()  get_latest_report_for_patient()                    │
│   get_report_by_id()      add_report_if_new()  (locked, atomic append)       │
│                                                                              │
│  Store: data/patients.json · data/reports.json  (synthetic)                  │
└──────────────────────────────────────────────────────────────────────────────┘
        │ validated responses
        ▼
Browser: Patient sidebar · Report viewer + editable draft · Conversation panel
         (every answer shows its source evidence; demo/live badge)
```

## Flows

### 1. Voice / STT
Mic button → `useSpeechRecognition.js` (Web Speech API, `en-IN`, continuous) → on speech end the final transcript **auto-submits into the same handler as typed input** → `POST /query` (retrieval) or `POST /dictation` (dictation). No audio is stored or uploaded by the backend; unsupported browsers get the typed-input path.

### 2. LLM interpretation
`POST /query {text}` → `interpret_query()` → Gemini returns a JSON intent (retrieve_report / dictation / follow_up / patient_info / report_count / greeting / assistant_identity / capabilities / small_talk / general_dental / unavailable_action / unsupported …) with `patient_name`, `report_type`, `question`. Output is Pydantic-validated (an out-of-vocabulary intent counts as malformed output), then re-checked in Python: a patient name the model invented — absent from the user's text — is dropped, conversation intents can never target a record, and a request to fabricate clinical content is refused (`intent=unsupported`) regardless of what the model decided. Provider error or malformed JSON → deterministic parser, surfaced as `fallback_reason`, never a 500.

### 3. Tool / API (deterministic retrieval)
`retrieval.resolve_patient_name()` resolves the extracted name by priority — exact full name → exact first name → exact last name → fuzzy (0.45 threshold; a 0.20 score gap between candidates means ambiguity). One hit → `get_latest_report_for_patient()` / `get_report_by_id()` returns the report; 2+ hits → `ambiguous: true` + candidate list (UI shows a disambiguation card, never a silent pick); no hit → helpful message, never a fabricated record. **No LLM is involved in any read.**

### 4. Report generation (dictation → saved report)
`POST /dictation {transcript}` → header/metadata lines split off → clinical-vocabulary gate: no clinical signal ⇒ `usable: false`, empty `draft_report`, no Save action (a model `"usable": false` verdict does the same) → otherwise LLM structures `{findings[], anatomical_site, draft_report, uncertainties[]}` → Pydantic validates with **`requires_review` forced `true`** → grounding pass keeps only findings verbatim present in the transcript and strips unsupported severities → `DictationResponse` renders an editable draft with a review banner. The clinician edits it, explicitly confirms, and `POST /reports` appends `{R###, findings, source_transcript, status: final|draft, clinician_confirmed}` to `reports.json` (existing patient by id, or a new `P###` patient — identity-verified, de-duplicated, atomic write), after which the record is retrievable through every read endpoint.

### 5. Conversational follow-up (report-aware chat)
With a report open, `POST /query {question, session_context:{active_report_id}}` classifies `intent=follow_up` and returns that report (fetched deterministically); the frontend then calls `POST /follow-up {question, report_id}`, where the system prompt contains **only that report's content** and the model is constrained to answer from it. Evidence strings are re-checked against the report (fabricated evidence is replaced by the report's own findings) and displayed with the answer. No open report → guidance to open one, no invented context.

## Safety properties

| Property | Mechanism |
|---|---|
| No fabricated findings | Prompt forbids invention + transcript grounding check drops anything not verbatim present |
| No drafts from noise input | Deterministic clinical gate before any LLM call; `usable: false` removes draft + Save UI |
| Always requires review | `requires_review` validator returns `true` unconditionally; save needs explicit confirmation |
| Evidence shown | Every follow-up answer lists the exact report text it used |
| Safe identity handling | Priority-tier name resolution; ambiguity surfaced, never guessed; invented names dropped |
| Graceful failure | Provider/malformed-output → deterministic fallback with `fallback_reason`, never a 500 |
| Provider-free identity | Greeting/identity answers are canned text — the app never names its underlying AI provider |
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
