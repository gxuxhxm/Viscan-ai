# Viscan AI — Dental Radiology Assistant (Take-Home POC)

A proof-of-concept dental radiology AI assistant with patient report retrieval, clinical dictation structuring, and conversational follow-up — built with React + Vite + Tailwind CSS v4 (frontend) and Python + FastAPI + Pydantic (backend).

---

## Quick Start

### 1. Backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# add GEMINI_API_KEY to .env to enable Gemini (optional — demo mode works without it)
uvicorn main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install   # already installed if node_modules exists
npm run dev   # starts on http://localhost:5173
```

Open **http://localhost:5173** in your browser.

---

## Features

| Feature | Description |
|---|---|
| **Patient report retrieval** | Type or speak "Open Priya's latest OPG" - Gemini interprets intent, deterministic functions retrieve the record |
| **Ambiguous identity handling** | "Priya" matches both Priya Sharma & Priyanka Sharma - UI shows a disambiguation card |
| **Clinical dictation** | Type or dictate a clinical transcript, get grounded structured findings + editable draft report - always requires clinician review |
| **Conversational follow-up** | Ask "Summarise her report" - answered from the open report with source evidence displayed |
| **Voice input** | Mic buttons for retrieval and dictation use the browser Web Speech API - speech ends → transcript auto-submits; typed input is always available |
| **Demo mode** | Fully functional without an API key using deterministic heuristics |

---

## Architecture

```
VISCAN AI/
├── backend/
│   ├── main.py              # FastAPI app, all endpoints
│   ├── models.py            # Pydantic request/response schemas
│   ├── services/
│   │   ├── retrieval.py     # Deterministic data lookup (no LLM)
│   │   └── llm_service.py   # LLM calls + demo-mode heuristics
│   ├── data/
│   │   ├── patients.json    # 5 synthetic patients
│   │   └── reports.json     # 10 synthetic reports (2+ per patient)
│   ├── tests/
│   │   ├── test_api.py      # FastAPI integration tests
│   │   └── test_services.py # Unit tests (no HTTP)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main 3-panel application
│   │   ├── api.js           # Fetch-based API client
│   │   ├── index.css        # DESIGN.md token system + Tailwind v4
│   │   └── main.jsx
│   └── package.json
└── ARCHITECTURE.md
```

### Key Design Decisions

1. **LLM ↔ Retrieval separation**: `llm_service.py` only interprets intent and structures text. All record lookup goes through `retrieval.py`, which is pure Python with no LLM involvement.

2. **Pydantic validation**: Every LLM output is validated against a Pydantic schema before use. If validation fails, a safe deterministic fallback is returned.

3. **Demo mode**: if no `GEMINI_API_KEY` (or `OPENAI_API_KEY`) is set, all features remain functional via regex/heuristic fallbacks. The UI clearly labels demo-mode responses.

4. **Dictation safety**: `requires_review` is always forced to `True` in Pydantic validators, regardless of LLM output. The UI prominently shows a clinician review warning.

5. **Deterministic output grounding**: live-model output is re-checked in Python before reaching the client — dictation findings must appear verbatim in the transcript (others are discarded and flagged), follow-up evidence must be quoted from the report (fabricated evidence is replaced by the report's own findings), and a patient name the model invents (absent from the user's text) is dropped.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check + demo mode status |
| GET | `/patients` | List all patients |
| GET | `/patients/{id}/reports` | List reports for a patient |
| GET | `/reports/{id}` | Get full report detail |
| POST | `/query` | NLU query → intent + retrieval |
| POST | `/dictation` | Structure clinical transcript |
| POST | `/follow-up` | Q&A on a specific report |

### Sample: POST /query

```json
// Request (exact full name → resolves)
{ "text": "Open Priya Sharma's latest OPG report" }

// Response
{
  "intent": "retrieve_report",
  "patient_name": "Priya Sharma",
  "report_type": "OPG",
  "matched_patients": [{ "id": "P001", "name": "Priya Sharma", ... }],
  "report": { "id": "R001", "type": "OPG", "date": "2025-11-10", "findings": [...], ... },
  "ambiguous": false,
  "demo_mode": true,
  "message": "Showing the latest OPG report for Priya Sharma dated 2025-11-10."
}
```

### Sample: Ambiguous query

```json
// Request
{ "text": "Open Sharma's OPG" }

// Response
{
  "ambiguous": true,
  "matched_patients": [
    { "id": "P001", "name": "Priya Sharma" },
    { "id": "P002", "name": "Priyanka Sharma" }
  ],
  "report": null,
  "message": "Multiple patients match 'Sharma'. Please select the correct patient."
}
```

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

All tests run without any paid API calls (demo mode forced via blank keys in the test environment — no Gemini/OpenAI call is ever made).

Tests cover:
- ✅ Correct report retrieval (exact full name, + type-filtered/latest)
- ✅ Ambiguous patient identity (Priya vs Priyanka Sharma — partial names ask, exact names resolve)
- ✅ Missing records (unknown patients/reports fail safely with helpful messages)
- ✅ Dictation does not fabricate clinical findings
- ✅ `requires_review` is always `True`
- ✅ Empty inputs are rejected (422)
- ✅ Unknown patients return 404/helpful message
- ✅ All 5 patients each have ≥2 reports
- ✅ Provider failures & malformed LLM output fall back to deterministic parsers (simulated in tests)
- ✅ Contextual follow-up uses session context when present, and guides the user when absent

---

## Privacy & Limitations

### Privacy
- All patient data is **entirely synthetic** — no real patient records used.
- No data is persisted beyond the in-memory session; JSON files are read-only fixtures.
- No authentication or authorisation is implemented. **Do not deploy publicly or use with real patient data.**

### Authorisation
- This POC has no role-based access control, audit logging, or session management.
- In production, every endpoint would require clinician authentication and request-level audit trails.

### Latency & Model Costs
- With Gemini enabled (flash-lite tier): observed 2-6s per request when healthy; during Google-side capacity pressure requests took 5-18s (bounded retries of 429/5xx add up to ~3s), after which the deterministic fallback answers instead. Flash-lite is a low-cost tier - POC-sized requests cost fractions of a cent (see Google's current pricing).
- Demo mode: all responses are sub-millisecond, zero cost.

### Known Limitations
1. **Demo-mode NLU is heuristic**: regex-based intent detection will miss complex phrasings.
2. **Fuzzy name matching**: token/prefix similarity with a 0.45 inclusion threshold. Exact
   full-name matches always resolve to that patient; partial names matching 2+ patients
   within a 0.20 score gap require clarification. Very short or misspelled names may still
   be missed or over-matched.
3. **Follow-up context is the open report**: a follow-up question is always answered from
   the currently open report — pronouns like "her report" are not resolved across patients.
4. **Voice input is browser-native**: the mic buttons use the Web Speech API - best supported in Chrome/Edge (which may send audio to Google's servers for transcription, so a network connection is required); Firefox and some Safari versions don't support it and show a typed-input fallback. Speech recognition isn't covered by automated tests - mic flows need a real browser + permission grant.
5. **Gemini model availability varies by account**: the configured `GEMINI_MODEL` may return 404 (unavailable to your account) or 503 (high demand). The backend retries transient errors then falls back to deterministic parsing (surfaced in the response message) - set `GEMINI_MODEL` in `backend/.env` if needed.
6. **Dictation draft prose is not deterministically verified**: the structured findings list is grounding-checked against the transcript, but the free-text draft around it is model-written prose - hence the always-on review gate.
7. **No report images**: report viewer shows structured text, not DICOM images.
8. **No persistent sessions**: conversation context lives in React state only; page refresh clears it.
9. **Single-user**: no multi-tenancy or concurrent session management.
10. **Not for clinical use**: this is a take-home POC demonstrating architecture patterns only.

---

## Environment Variables

Copy `backend/.env.example` to `backend/.env`:

```
GEMINI_API_KEY=       # Optional - Google Gemini key; leave blank for demo mode
GEMINI_MODEL=gemini-3.5-flash-lite   # Optional - model override (availability varies by account)
OPENAI_API_KEY=       # Optional legacy fallback when no Gemini key is set
```

`.env` is excluded from Git via `.gitignore`.
