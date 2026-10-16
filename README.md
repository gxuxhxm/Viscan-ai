# Viscan AI — Dental Reporting AI

An AI dental reporting assistant that covers the three mandated workflows — **patient report retrieval**, **clinical dictation → draft report**, and **context-dependent conversational follow-up** — plus a bonus **report-aware chatbot**. Voice input and a full no-key demo mode are built in.

**Stack:** React 19 + Vite + Tailwind CSS v4 (frontend) · Python + FastAPI + Pydantic (backend) · Google Gemini / OpenAI (optional) · deterministic fallback (no key needed).

> Architecture: see [ARCHITECTURE.md](./ARCHITECTURE.md) · Design tokens: [DESIGN.md](./DESIGN.md)
>
> Not a certified clinical product. All patient data is synthetic. See [Privacy, Cost & Authorization](#8-privacy-cost--authorization).

---

## 1. Delivered features (requirement → status)

### A. Patient report retrieval — ✅ delivered (mandatory)
- Typed **or spoken** request such as *"Open Priya's latest OPG report"* → intent + patient name extracted, mock patient/report data queried, correct report returned.
- **Ambiguity is never guessed**: *"Sharma"* matches Priya Sharma and Priyanka Sharma → the UI shows a disambiguation card; fuzzy candidates within a 0.20 score gap also ask instead of picking.
- **Missing data fails safely**: unknown patients and missing reports return a helpful message, never a fabricated record.

### B. Clinical dictation to draft report — ✅ delivered (mandatory)
- Audio (browser speech-to-text) **or** a pasted transcript such as *"Lower right third molar is horizontally impacted"* → **editable structured JSON**: findings, anatomical site, severity, and a readable draft report.
- **No invented observations**: every finding must appear in the transcript (re-verified in Python; unsupported findings are dropped and flagged), uncertainties are listed, and `requires_review` is forced to `true`.
- Non-clinical/noise transcripts return `usable: false` → no draft and no Save action.
- The clinician edits the draft, explicitly confirms it, and saves it (existing or new patient) back into the JSON store.

### C. Conversational follow-up — ✅ delivered (mandatory)
- One context-dependent follow-up such as *"Summarise her report"* is answered from the **currently open report** — the patient is not asked for again when the reference is unambiguous.
- **Every answer shows its source**: the exact report evidence (finding text) used for the reply is displayed alongside it.
- No open report → guided reply instead of an invented answer.

### Bonus — Report-aware chatbot — ✅ delivered
- Grounded question answering over the selected report (`POST /follow-up`), plus greeting/identity/capability/small-talk routing (`POST /query`). Live vs demo-mode responses are always labelled in the UI.

### Also delivered
- **Voice input** for retrieval and dictation (mic → transcript auto-submits on speech end), with typed-input fallback everywhere.
- **Demo mode**: every feature works with zero API keys.
- **208 automated tests** (181 backend + 27 frontend), none of which make a paid API call.

---

## 2. Setup

**Prerequisites:** Python 3.10+, Node 18+.

### Backend — `http://localhost:8000`

```powershell
cd backend
python -m venv .venv                                     # first run only
.\.venv\Scripts\pip.exe install -r requirements.txt      # first run only
copy .env.example .env                                   # first run only; keys optional
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

Same thing with an activated virtualenv:

```powershell
cd backend
.\.venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

Sanity check: `http://localhost:8000/health` → `{"status":"ok","demo_mode":true,"llm_provider":null}`

### Frontend — `http://localhost:5173`

```powershell
cd frontend
npm install        # first run only
npm run            # list the available scripts
npm run dev        # → http://localhost:5173
```

Open **http://localhost:5173** in your browser.

### Ports (docs = code)

| Service | URL | Where it is defined |
|---|---|---|
| Backend API | `http://localhost:8000` | `--port 8000` above; `frontend/src/api.js` → `VITE_API_URL \|\| 'http://localhost:8000'` |
| Frontend dev server | `http://localhost:5173` | Vite default port (no override in `vite.config.js`); allowed by backend CORS in `backend/main.py` |

To use another API port, set `VITE_API_URL` in `frontend/.env` **and** add that origin to the CORS list in `backend/main.py`.

---

## 3. Demo instructions (no API key required)

1. Leave `GEMINI_API_KEY` empty (or skip creating `backend/.env` entirely).
2. Start the backend and frontend as above. `GET /health` reports `"demo_mode": true` and the UI shows the **Demo Mode** banner.
3. Try these in the UI:
   - `Open Priya's latest OPG report` → correct report retrieved
   - `Open Sharma's OPG` → ambiguity card (two matching patients)
   - Dictate or paste `Lower right third molar is horizontally impacted` → **Create AI Report** → edit → **Save Report** (explicit confirmation)
   - With a report open: `Summarise her report` → answer + evidence

Demo mode uses deterministic regex/heuristic parsers: sub-millisecond responses, zero cost, fully covered by the test suite. Adding a key switches intent interpretation, dictation structuring and follow-up to the live model without any code change.

---

## 4. API instructions

Base URL: `http://localhost:8000` · Interactive docs: `http://localhost:8000/docs`

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health + `demo_mode` + active `llm_provider` |
| GET | `/patients` | List patients |
| GET | `/patients/{id}/reports` | Reports for a patient |
| GET | `/reports/{id}` | Full report detail |
| POST | `/query` | NLU: intent + patient-name resolution + retrieval |
| POST | `/dictation` | Transcript → structured findings + editable draft |
| POST | `/follow-up` | Grounded question answering on one report |
| POST | `/reports` | Save an edited draft (creates a new patient if needed) |

### Examples

> On Windows PowerShell, call `curl.exe` (plain `curl` there is an alias for `Invoke-WebRequest`).

```bash
# A. Patient report retrieval
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"text": "Open Priya latest OPG report"}'

# B. Clinical dictation → structured draft
curl -X POST http://localhost:8000/dictation \
  -H 'Content-Type: application/json' \
  -d '{"transcript": "Lower right third molar is horizontally impacted"}'

# C. Conversational follow-up (report_id from a previous /query or /reports call)
curl -X POST http://localhost:8000/follow-up \
  -H 'Content-Type: application/json' \
  -d '{"question": "Summarise the findings", "report_id": "R001"}'
```

### Sample response — retrieval

```json
{
  "intent": "retrieve_report",
  "patient_name": "Priya Sharma",
  "report_type": "OPG",
  "matched_patients": [{ "id": "P001", "name": "Priya Sharma" }],
  "report": {
    "id": "R001",
    "type": "OPG",
    "date": "2025-11-10",
    "status": "final",
    "findings": [
      { "site": "Upper left quadrant",
        "observation": "Periapical radiolucency at 26 (upper left first molar)",
        "severity": "moderate" }
    ],
    "impression": "Periapical pathology at 26 requiring endodontic evaluation. …"
  },
  "ambiguous": false,
  "demo_mode": true,
  "message": "Showing the latest OPG report for Priya Sharma dated 2025-11-10."
}
```

An ambiguous request returns `"ambiguous": true`, the candidate list in `matched_patients`, `"report": null`, and a message asking the user to choose. Live-model responses may additionally carry `fallback_reason` / `limitation` explaining any deterministic downgrade.

---

## 5. Model & provider choices

| Stage | Choice | Why |
|---|---|---|
| **Voice / STT** | Browser Web Speech API in the frontend (`frontend/src/useSpeechRecognition.js`, `en-IN`), mic transcript auto-submits; typed input always available | No audio files uploaded, no server-side STT key or cost; transcript enters the exact same pipeline as typed text |
| **LLM — primary** | **Google Gemini**, default `gemini-3.5-flash-lite` via `GEMINI_API_KEY` (`GEMINI_MODEL` to override) | Flash tier: cheapest/lowest-latency option for short JSON-structured clinical prompts |
| **LLM — fallback** | **OpenAI `gpt-4o-mini`** via `OPENAI_API_KEY`, used only when the Gemini key is blank | Second provider for regions/accounts without Gemini access |
| **LLM — no key** | Deterministic regex/heuristic parsers in `backend/services/llm_service.py` (demo mode) | App stays fully usable, testable and free with zero keys |
| **Tools / data access** | Pure Python functions in `backend/services/retrieval.py` over `backend/data/*.json` — **never an LLM** | Record lookup must be exact, auditable and instant |
| **Output validation** | Pydantic schemas in `backend/models.py` on every model response | Malformed or invented model output never reaches the UI |

Provider errors (404 model unavailable, 429/503 capacity, malformed JSON) are retried briefly and then answered deterministically, with the reason surfaced as `fallback_reason` — never a 500.

---

## 6. Environment variables (sample)

Copy `backend/.env.example` → `backend/.env`:

```dotenv
# Google Gemini (primary). Leave blank → demo mode, or OpenAI fallback below.
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.5-flash-lite

# OpenAI (optional legacy fallback; used only when GEMINI_API_KEY is blank)
OPENAI_API_KEY=
```

Optional `frontend/.env`:

```dotenv
VITE_API_URL=http://localhost:8000
```

**Secrets handling:** no API key exists anywhere in the committed code — only the empty `.env.example` template is tracked. `.env` (backend and frontend) is listed in `.gitignore` and is not, and has never been, committed; verify any time with:

```powershell
git check-ignore backend/.env     # → matches .gitignore
git ls-files backend/.env         # → empty (nothing to push)
```

---

## 7. Running tests (brief)

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest tests/ -q     # 181 passed (~4 s)

cd ..\frontend
npm test                                           # 27 passed (~5 s)
```

- Keys are blanked in the test environment, so **no paid API call is ever made**.
- Backend covers retrieval + name resolution + ambiguity, missing records, no fabricated findings, unusable transcripts, `requires_review` always true, empty-input 422s, provider-failure fallbacks, follow-up context, and save/confirm/new-patient flows.
- Frontend (Vitest + Testing Library) covers the ambiguity panel, unusable-dictation state and the new-patient entry dialog.

---

## 8. Privacy, cost & authorization

### Privacy
- All patient data is **entirely synthetic** (6 patients, 12 reports) — no real records, no PII.
- Everything stays local: reads and writes (saving an edited draft, creating a patient) append to `backend/data/*.json` on your machine. Nothing is uploaded anywhere except the prompt text sent to the configured LLM when a key is set.
- `backend/.env` with your key is git-ignored and never pushed. Restore pristine fixtures with `git checkout backend/data`.

### Cost & latency
- **Demo mode:** zero cost, sub-millisecond responses.
- **Live mode (flash-lite tier):** fractions of a cent per request; observed 2–6 s healthy, 5–18 s during provider-side pressure (bounded retries add ~3 s) before the deterministic fallback answers. See your provider's current pricing for exact rates.

### Authorization
- **No authentication or authorization is implemented** — there is no login, role-based access control, audit logging or session management.
- **Do not deploy publicly or use with real patient data.** In production every endpoint would need clinician authentication, request-level audit trails, and encrypted storage.

---

## 9. Known limitations

1. **Demo-mode NLU is heuristic** — regex-based intent detection misses complex phrasings (live mode handles them).
2. **Fuzzy name matching** — token/prefix similarity with a 0.45 threshold; exact full names always resolve, partial names matching 2+ patients within a 0.20 score gap require clarification. Very short or misspelled names may still be missed or over-matched.
3. **Follow-up context = the open report** — pronouns such as "her report" are not resolved across patients.
4. **Voice input is browser-native** — the Web Speech API works best in Chrome/Edge (which may send audio to the speech provider, so a network connection is needed); Firefox/older Safari fall back to typed input. Mic flows are not covered by automated tests.
5. **Model availability varies by account** — `GEMINI_MODEL` can return 404/503; the backend retries, then falls back deterministically (`fallback_reason` in the response). Override `GEMINI_MODEL` in `backend/.env`.
6. **Draft prose is not fully grounding-checked** — findings are verified verbatim against the transcript; the surrounding free text is model-written, hence the always-on review gate.
7. **No report images** — the viewer shows structured text, not DICOM.
8. **No persistent sessions** — conversation context lives in React state; a refresh clears it.
9. **Single-user** — no multi-tenancy or concurrent session handling.
10. **Not for clinical use** — an architecture demonstration, not a certified clinical product.

---

## 10. Repository layout

```
VISCAN AI/
├── backend/
│   ├── main.py                 # FastAPI app + all endpoints + CORS
│   ├── models.py               # Pydantic request/response schemas
│   ├── services/
│   │   ├── retrieval.py        # deterministic data lookup (no LLM) + safe writes
│   │   └── llm_service.py      # Gemini/OpenAI calls, prompts, demo fallbacks, grounding checks
│   ├── data/                   # synthetic patients.json / reports.json
│   ├── tests/                  # 181 pytest tests (no paid API calls)
│   ├── .env.example            # environment template (real .env is git-ignored)
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx             # 3-panel application
│       ├── api.js              # fetch client → http://localhost:8000
│       ├── useSpeechRecognition.js  # Web Speech API voice input
│       └── __tests__/          # 27 vitest tests
├── ARCHITECTURE.md             # one-page architecture (voice/STT → LLM → tools → report)
├── DESIGN.md                   # design tokens
└── README.md
```
