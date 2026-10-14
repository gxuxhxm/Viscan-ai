from __future__ import annotations
import os
import json
import re
import time
from typing import Optional
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Output schemas (validated after LLM call)
# ---------------------------------------------------------------------------

class InterpretedQuery(BaseModel):
    intent: str          # "retrieve_report" | "dictation" | "follow_up" | "unknown"
    patient_name: Optional[str] = None
    report_type: Optional[str] = None   # OPG / PERIAPICAL / BITEWING / CBCT
    free_text: Optional[str] = None
    # Set when the LLM was unavailable/malformed and a deterministic
    # fallback parser was used instead (surfaced to the user).
    fallback_reason: Optional[str] = None

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        allowed = {"retrieve_report", "dictation", "follow_up", "unknown"}
        if v not in allowed:
            raise ValueError(f"Invalid intent '{v}'. Must be one of {allowed}")
        return v

    @field_validator("report_type")
    @classmethod
    def normalise_report_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        mapping = {
            "opg": "OPG", "orthopantomogram": "OPG", "panoramic": "OPG",
            "pano": "OPG",
            "periapical": "PERIAPICAL", "pa": "PERIAPICAL",
            "bitewing": "BITEWING", "bw": "BITEWING",
            "cbct": "CBCT", "cone beam": "CBCT", "cone-beam": "CBCT",
        }
        return mapping.get(v.lower().strip(), v.upper())


class StructuredDictation(BaseModel):
    findings: list[dict]
    anatomical_site: str
    draft_report: str
    uncertainties: list[str]
    requires_review: bool = True

    @field_validator("findings")
    @classmethod
    def validate_findings(cls, v: list) -> list:
        for item in v:
            if not isinstance(item, dict):
                raise ValueError("Each finding must be a dict")
            if "observation" not in item:
                raise ValueError("Each finding must have 'observation'")
        return v

    @field_validator("requires_review")
    @classmethod
    def always_review(cls, v: bool) -> bool:
        return True  # Always force clinician review


# ---------------------------------------------------------------------------
# LLM Service
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Which provider (if any) is configured. None → deterministic demo mode.
LLM_PROVIDER = "gemini" if GEMINI_API_KEY else ("openai" if OPENAI_API_KEY else None)
DEMO_MODE = LLM_PROVIDER is None


def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    """
    Call Google Gemini's generateContent REST endpoint (no extra SDK
    dependency) and return the plain-text completion.
    Raises on network/HTTP/malformed-response errors so callers can fall back.
    """
    try:
        import httpx  # type: ignore
    except ImportError:
        raise RuntimeError("httpx package not installed")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024},
    }
    response = None
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = httpx.post(
                url, params={"key": GEMINI_API_KEY}, json=payload, timeout=30.0
            )
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            last_exc = exc
            # Transient capacity/rate-limit errors → bounded retry with backoff
            if status in {429, 500, 502, 503, 529} and attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise
    else:
        raise last_exc if last_exc else RuntimeError("Gemini request failed")

    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call the configured provider. Returns raw text content."""
    if GEMINI_API_KEY:
        return _call_gemini(system_prompt, user_prompt)

    try:
        import openai  # type: ignore
    except ImportError:
        raise RuntimeError("openai package not installed")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def _parse_llm_json(raw: str) -> dict:
    """
    Parse a JSON object from an LLM response, tolerating markdown code fences
    (```json … ```) that Gemini sometimes wraps around its output.
    Raises json/ValueError on anything that is not a JSON object.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM output must be a JSON object")
    return data


# ---------------------------------------------------------------------------
# Deterministic fallback parsers (used in demo mode)
# ---------------------------------------------------------------------------

REPORT_TYPE_KEYWORDS = {
    "opg": "OPG",
    "orthopantomogram": "OPG",
    "panoramic": "OPG",
    "periapical": "PERIAPICAL",
    "pa": "PERIAPICAL",
    "bitewing": "BITEWING",
    "bw": "BITEWING",
    "cbct": "CBCT",
    "cone beam": "CBCT",
    "cone-beam": "CBCT",
}

REPORT_TYPE_RE = re.compile(
    r"\b(opg|orthopantomogram|panoramic|periapical|bitewing|bw|pa|cbct|cone[\s-]beam)\b",
    re.IGNORECASE,
)

# Command verbs / prepositions that anchor a following patient name.
_NAME_ANCHORS = r"open|show|get|fetch|retrieve|view|find|load|display|pull|for|of|from|patient"

# Words that are never a patient name (guards against "Show me…", "the report…",
# "open the opg report…").
_NAME_STOPWORDS = {
    "me", "the", "a", "an", "this", "that", "these", "those", "her", "his", "their",
    "my", "our", "your", "its", "it", "all", "any", "latest", "newest", "old",
    "report", "reports", "patient", "patients", "file", "files", "image", "images",
    "scan", "scans", "record", "records", "document", "documents", "chart", "name",
    "list", "summary", "details", "info", "information", "doctor", "dr", "clinic",
    # radiology / request vocabulary
    "opg", "cbct", "bitewing", "periapical", "panoramic", "recent", "recently",
    "please", "full", "most", "findings", "finding", "result", "results",
}

# Single-word title-case name: "Priya Sharma"
_NAME_TITLE = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"


def _valid_name(candidate: str) -> Optional[str]:
    """Validate a captured name candidate; returns a normalised name or None."""
    candidate = " ".join(candidate.split())
    if not candidate:
        return None
    # Strip leading generic words and command verbs ("Open Priya" → "Priya",
    # "the report" → "" → rejected)
    words = candidate.split()
    anchors = set(_NAME_ANCHORS.split("|"))
    while words and (
        words[0].casefold() in _NAME_STOPWORDS or words[0].casefold() in anchors
    ):
        words = words[1:]
    candidate = " ".join(words)
    if len(candidate) < 2:
        return None
    if candidate.islower():
        candidate = candidate.title()
    return candidate


def _extract_patient_name(text: str) -> Optional[str]:
    """
    Deterministically extract a patient name from a free-text query.
    Never lets command verbs ("Open", "Show") become part of the name.
    """
    # 1. After a command verb / preposition (verb is case-insensitive, the name
    #    must be title-cased): "Open Priya Sharma's …", "report for Priya"
    for m in re.finditer(rf"(?i:(?:{_NAME_ANCHORS})\s+)({_NAME_TITLE})", text):
        name = _valid_name(m.group(1))
        if name:
            return name

    # 1b. Same anchors, fully-lowercase names: "open priya sharma opg"
    for m in re.finditer(
        rf"(?i:(?:{_NAME_ANCHORS})\s+)([a-z][a-z]+(?:\s+[a-z][a-z]+)?)", text
    ):
        name = _valid_name(m.group(1))
        if name:
            return name

    # 2. Title-case possessive: "Priya Sharma's latest report"
    for m in re.finditer(rf"\b({_NAME_TITLE})'s\b", text):
        name = _valid_name(m.group(1))
        if name:
            return name

    # 3. Lowercase possessive: "open priya's opg" → "Priya"
    for m in re.finditer(r"\b([a-z][a-z]+(?:\s+[a-z][a-z]+)?)'s\b", text):
        name = _valid_name(m.group(1))
        if name:
            return name

    # 4. Any two consecutive title-case words (last resort), skipping a leading verb
    for m in re.finditer(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+)\b", text):
        words = m.group(1).split()
        # Never start the name with a command verb ("Open Priya" → "Priya")
        while words and words[0].casefold() in _NAME_ANCHORS.split("|"):
            words = words[1:]
        name = _valid_name(" ".join(words))
        if name:
            return name
    return None


_RETRIEVE_VERB_RE = re.compile(
    r"\b(?:open|show|get|fetch|retrieve|view|find|load|display|pull)\b", re.IGNORECASE
)
_QUESTION_RE = re.compile(r"\?|^\s*(?:what|why|how|when|where|who|which|is|are|was|were|"
                          r"can|could|should|would|do|does|did|explain|tell|describe|"
                          r"summar\w*|compare|review|any)\b", re.IGNORECASE)
_DICTATION_RE = re.compile(
    r"\b(?:transcribe|transcription|dictate|dictation|record|note|notes|document)\b",
    re.IGNORECASE,
)


def _demo_interpret_query(text: str) -> InterpretedQuery:
    """Heuristic intent parsing — no LLM, used in demo mode."""
    # Detect report type (word-bounded, case-insensitive)
    report_type = None
    m = REPORT_TYPE_RE.search(text)
    if m:
        key = m.group(1).lower()
        report_type = REPORT_TYPE_KEYWORDS.get(key) or REPORT_TYPE_KEYWORDS.get(key.replace("-", " "))

    # Detect intent. Order matters:
    #   1. explicit retrieval verbs  ("Open Priya's OPG report")
    #   2. questions                 ("What are the key findings?", "Summarise her report")
    #   3. dictation cues            ("Note: mild bone loss at 46")
    #   4. a patient name without a verb  ("Priya's latest report")
    if _RETRIEVE_VERB_RE.search(text):
        intent = "retrieve_report"
    elif _QUESTION_RE.search(text):
        intent = "follow_up"
    elif _DICTATION_RE.search(text):
        intent = "dictation"
    else:
        intent = "unknown"

    patient_name = _extract_patient_name(text)
    if intent == "unknown" and patient_name:
        intent = "retrieve_report"

    return InterpretedQuery(
        intent=intent,
        patient_name=patient_name,
        report_type=report_type,
        free_text=text,
    )


# ── Deterministic dictation structuring (demo mode) ──────────────────────────
# These rules only MATCH text that is literally present in the transcript.
# They never add clinical content: the observation stays verbatim, and
# severity/site are reported only when stated (otherwise "unspecified").

_SEVERITY_RULES = [
    (re.compile(r"\b(severe|severely)\b", re.I), "severe"),
    (re.compile(r"\b(moderate|moderately)\b", re.I), "moderate"),
    (re.compile(r"\b(mild|mildly|slight|slightly)\b", re.I), "mild"),
    (re.compile(
        r"\b(within normal limits|no significant (?:findings|abnormalities|pathology)"
        r"|appears? normal|is normal|no pathology|no periapical patholog\w*"
        r"|no caries|no abnormalit\w*|unremarkable)\b",
        re.I,
    ), "none"),
]

_QUADRANT_RULES = [
    (re.compile(r"\bupper[-\s]?left\b", re.I), "upper left"),
    (re.compile(r"\bupper[-\s]?right\b", re.I), "upper right"),
    (re.compile(r"\blower[-\s]?left\b", re.I), "lower left"),
    (re.compile(r"\blower[-\s]?right\b", re.I), "lower right"),
    (re.compile(r"\bupper anterior(?: teeth)?\b", re.I), "upper anterior"),
    (re.compile(r"\blower anterior(?: teeth)?\b", re.I), "lower anterior"),
]

_STRUCTURE_RULES = [
    (re.compile(r"\bthird molar(?:s)?\b|\bwisdom teeth?\b", re.I), "third molar"),
    (re.compile(r"\bmolar(?:s)?\b", re.I), "molar"),
    (re.compile(r"\bpremolar(?:s)?\b", re.I), "premolar"),
    (re.compile(r"\bincisor(?:s)?\b", re.I), "incisor"),
    (re.compile(r"\bcanine(?:s)?\b", re.I), "canine"),
    (re.compile(r"\bsinus(?:es)?\b", re.I), "sinus"),
    (re.compile(r"\btemporomandibular\b|\btmj\b", re.I), "TMJ"),
    (re.compile(r"\bjaw\b", re.I), "jaw"),
]

_TOOTH_NUM_RE = re.compile(r"\b(?:tooth|teeth)\s*(?:number\s*|#\s*)?(\d{1,2})\b", re.I)


def _detect_severity(sentence: str) -> str:
    """Return a severity label only when literally stated in the sentence."""
    for pattern, label in _SEVERITY_RULES:
        m = pattern.search(sentence)
        if not m:
            continue
        # Ignore negated mentions: "no mild …", "without moderate …",
        # "not within normal limits …"
        preceding = sentence[max(0, m.start() - 20):m.start()].lower().split()
        if preceding and preceding[-1] in {"no", "not", "without", "denies", "denied"}:
            continue
        return label
    return "unspecified"


def _detect_site(text: str) -> Optional[str]:
    """Return a comma-joined site description from words actually present."""
    parts: list[str] = []
    for pattern, label in _QUADRANT_RULES + _STRUCTURE_RULES:
        if pattern.search(text) and label not in parts:
            parts.append(label)
    for m in _TOOTH_NUM_RE.finditer(text):
        tooth = f"tooth {m.group(1)}"
        if tooth not in parts:
            parts.append(tooth)
    return ", ".join(parts) if parts else None


def _split_sentences(text: str) -> list[str]:
    """Split into sentences while KEEPING the terminating punctuation, so
    each observation can stay byte-for-byte verbatim from the transcript."""
    parts = re.split(r"([.;!?]+|\n+)", text)  # [chunk, delim, chunk, …, chunk]
    sentences: list[str] = []
    for i in range(0, len(parts) - 1, 2):
        chunk, delim = parts[i], parts[i + 1]
        if chunk.strip():
            sentences.append((chunk + delim).strip())
        elif sentences:
            sentences[-1] += delim  # consecutive delimiters → keep with previous
    if parts[-1].strip():
        sentences.append(parts[-1].strip())
    return sentences or [text.strip()]


def _demo_structure_dictation(text: str) -> StructuredDictation:
    """
    Deterministic dictation structuring — demo mode only.
    Splits the transcript into findings, echoing each sentence verbatim and
    deriving site/severity only from words that are literally present.
    Never invents clinical observations.
    """
    text = text.strip()
    sentences = _split_sentences(text)
    overall_site = _detect_site(text)

    findings = []
    for sentence in sentences:
        findings.append({
            "site": _detect_site(sentence) or overall_site or "unspecified",
            "observation": sentence,
            "severity": _detect_severity(sentence),
        })

    uncertainties = [
        "Demo mode: deterministic parsing only — no AI interpretation performed",
        "All fields unverified — clinician confirmation required",
    ]
    if not overall_site:
        uncertainties.append("Anatomical site not stated in the transcript")
    if any(f["severity"] == "unspecified" for f in findings):
        uncertainties.append("Severity not stated for some findings")

    anatomical_site = overall_site or "unspecified — please review"

    findings_block = "\n".join(
        f"{i}. [{f['site']}] (severity: {f['severity']}) — {f['observation']}"
        for i, f in enumerate(findings, start=1)
    )

    draft_report = (
        "CLINICAL DICTATION DRAFT\n\n"
        "[DEMO MODE — deterministic parser, no AI interpretation]\n"
        "Nothing below was inferred: observations are verbatim from the transcript.\n\n"
        f"TRANSCRIPT:\n{text}\n\n"
        f"FINDINGS:\n{findings_block}\n\n"
        f"ANATOMICAL SITE: {anatomical_site}\n\n"
        "REVIEW STATUS: Requires clinician confirmation. "
        "This draft is not finalised and is not saved automatically."
    )

    return StructuredDictation(
        findings=findings,
        anatomical_site=anatomical_site,
        draft_report=draft_report,
        uncertainties=uncertainties,
        requires_review=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def interpret_query(text: str, session_context: Optional[dict] = None) -> InterpretedQuery:
    """
    Use LLM to parse user intent. Falls back to heuristics in demo mode.
    Never fabricates patient data.
    """
    if DEMO_MODE:
        return _demo_interpret_query(text)

    ctx_str = ""
    if session_context:
        ctx_str = f"\n\nCurrent session context:\n{json.dumps(session_context, indent=2)}"

    system = (
        "You are a dental-imaging assistant query interpreter. "
        "Given a user's free-text query, output ONLY a JSON object with these fields:\n"
        '  "intent": one of ["retrieve_report","dictation","follow_up","unknown"]\n'
        '  "patient_name": string or null\n'
        '  "report_type": one of ["OPG","PERIAPICAL","BITEWING","CBCT"] or null\n'
        '  "free_text": the original user text\n'
        "Rules:\n"
        "- retrieve_report: user wants to open/view a radiology report\n"
        "- dictation: user is providing clinical findings to be structured\n"
        "- follow_up: user is asking a question about a previously retrieved report\n"
        "- unknown: none of the above\n"
        "Output ONLY valid JSON, no explanation."
        + ctx_str
    )

    try:
        raw = _call_llm(system, text)
    except Exception as exc:
        # Provider error (network, auth, rate limit…) — deterministic fallback
        result = _demo_interpret_query(text)
        result.free_text = text
        result.fallback_reason = f"AI provider unavailable ({type(exc).__name__}); used deterministic interpretation"
        return result

    # Parse and validate
    try:
        data = _parse_llm_json(raw)
        parsed = InterpretedQuery(**data)
        if not parsed.free_text:
            parsed.free_text = text
        # Deterministic grounding guards (same rules the demo parser applies):
        # - a patient name the model invented (not present in the user's text)
        #   is dropped, so retrieval can never target a fabricated identity
        if parsed.patient_name and not _name_in_text(parsed.patient_name, text):
            parsed.patient_name = None
        # - a name with no explicit verb is a retrieval request
        if parsed.intent == "unknown" and parsed.patient_name:
            parsed.intent = "retrieve_report"
        return parsed
    except Exception:
        # LLM returned invalid output — fall back deterministically
        result = _demo_interpret_query(text)
        result.free_text = text
        result.fallback_reason = "AI returned malformed output; used deterministic interpretation"
        return result


def _name_in_text(name: str, text: str) -> bool:
    """True when any word of the name (≥3 chars) appears in the user's text."""
    text_cf = text.casefold()
    return any(tok in text_cf for tok in name.casefold().split() if len(tok) >= 3)


def _fallback_dictation(transcript: str, reason: str) -> StructuredDictation:
    """
    Safe deterministic fallback when the LLM is unavailable or its output is
    malformed. Never invents findings: the transcript itself is the only
    'observation', and clinician review is always required.
    """
    return StructuredDictation(
        findings=[{"observation": transcript[:500], "site": "see transcript", "severity": "unspecified"}],
        anatomical_site="Could not parse — review required",
        draft_report=(
            "CLINICAL DICTATION DRAFT\n\n"
            f"[{reason} — manual transcription required]\n\n"
            f"Original transcript:\n{transcript}"
        ),
        uncertainties=[reason, "Manual review required"],
        requires_review=True,
    )


def _ground_findings(transcript: str, findings: list) -> tuple[list, int]:
    """
    Deterministic no-invention guard: keep only findings whose observation is
    literally present in the transcript (case/whitespace-insensitive).
    Returns (kept_findings, dropped_count).
    """
    norm = " ".join(transcript.casefold().split())
    kept: list = []
    dropped = 0
    for item in findings:
        obs = " ".join(str(item.get("observation", "")).casefold().split()).rstrip(".,;:!?")
        if obs and obs in norm:
            kept.append(item)
        else:
            dropped += 1
    return kept, dropped


def structure_dictation(transcript: str) -> StructuredDictation:
    """
    Use LLM to convert a clinical dictation transcript into structured JSON.
    Never invents clinical observations beyond what is in the transcript.
    """
    if DEMO_MODE:
        return _demo_structure_dictation(transcript)

    system = (
        "You are a dental radiology assistant. A clinician has dictated findings. "
        "Your task: extract and structure ONLY what was stated. "
        "NEVER invent or infer clinical observations not explicitly mentioned.\n\n"
        "Output ONLY a JSON object with:\n"
        '  "findings": list of {"site": str, "observation": str, "severity": "none"|"mild"|"moderate"|"severe"}\n'
        '  "anatomical_site": string summarising the primary region\n'
        '  "draft_report": a professional radiological report draft\n'
        '  "uncertainties": list of strings where the dictation was unclear or incomplete\n'
        '  "requires_review": true (always set this to true)\n\n'
        "Rules:\n"
        "- Copy each observation VERBATIM from the transcript; do not paraphrase it.\n"
        "- If the dictation is unclear, add the unclear point to uncertainties.\n"
        "- Do not add clinical conclusions not present in the dictation.\n"
        "- The draft_report must begin with the phrase 'CLINICAL DICTATION DRAFT'.\n"
        "- Write the draft as a short professional report with a FINDINGS heading\n"
        "  listing the observations, and end with a 'REVIEW STATUS:' line stating\n"
        "  that clinician confirmation is required before use.\n"
        "- Output ONLY valid JSON, no explanation."
    )

    try:
        raw = _call_llm(system, transcript)
    except Exception as exc:
        return _fallback_dictation(
            transcript, f"AI provider unavailable ({type(exc).__name__})"
        )

    try:
        data = _parse_llm_json(raw)
        structured = StructuredDictation(**data)
    except Exception:
        # Validation failed — return safe fallback
        return _fallback_dictation(transcript, "AI output failed validation")

    # ── Deterministic safety pass (LLM output never reaches the client raw) ──
    kept, dropped = _ground_findings(transcript, structured.findings)
    if dropped:
        if not kept:
            return _fallback_dictation(
                transcript, "AI output contained findings not present in the transcript"
            )
        structured.uncertainties = list(structured.uncertainties) + [
            f"{dropped} AI-generated finding(s) discarded: not present in the transcript"
        ]
        structured.findings = kept
    for item in structured.findings:
        item.setdefault("site", "unspecified")
        item.setdefault("severity", "unspecified")
    if not structured.draft_report.startswith("CLINICAL DICTATION DRAFT"):
        structured.draft_report = "CLINICAL DICTATION DRAFT\n\n" + structured.draft_report
    if "review" not in structured.draft_report.lower():
        # Deterministic review gate: the artefact itself always says so,
        # regardless of what the model chose to write.
        structured.draft_report = (
            f"{structured.draft_report.rstrip()}\n\n"
            "REVIEW STATUS: Requires clinician confirmation before use. "
            "Not finalised — do not save automatically."
        )
    if not structured.uncertainties:
        structured.uncertainties = ["AI-structured output — clinician verification required"]
    return structured


def _ground_evidence(report: dict, evidence: list) -> list[str]:
    """
    Deterministic grounding guard for follow-up evidence: keep only evidence
    strings that literally appear in the report record (findings or impression).
    """
    parts = [
        str(f.get("observation", "")) for f in report.get("findings", [])
    ]
    parts.append(str(report.get("impression", "")))
    blob = " ".join(" ".join(p.casefold().split()) for p in parts)
    grounded = []
    for e in evidence:
        e_cf = " ".join(str(e).casefold().split()).rstrip(".,;:!?")
        if e_cf and e_cf in blob:
            grounded.append(str(e))
    return grounded


def answer_follow_up(question: str, report: dict, session_context: Optional[dict] = None) -> dict:
    """
    Answer a follow-up question about a report. Returns answer + source evidence.
    Never answers beyond what the report contains.
    """
    if DEMO_MODE:
        return {
            "answer": (
                f"[DEMO MODE] Answering based on report {report.get('id','?')} "
                f"for patient {report.get('patient_id','?')}.\n\n"
                f"Report impression: {report.get('impression', 'N/A')}"
            ),
            "source_report_id": report.get("id"),
            "evidence": [f["observation"] for f in report.get("findings", [])],
            "demo_mode": True,
        }

    findings_text = "\n".join(
        f"- [{f.get('site','')}] {f.get('observation','')} (severity: {f.get('severity','')})"
        for f in report.get("findings", [])
    )

    system = (
        "You are a dental radiology assistant answering a clinician's question about a report. "
        "Answer ONLY using the report content provided. Do not fabricate findings. "
        "If the answer cannot be determined from the report, say so clearly.\n\n"
        f"Report ID: {report.get('id')}\n"
        f"Date: {report.get('date')}\n"
        f"Type: {report.get('type')}\n"
        f"Findings:\n{findings_text}\n"
        f"Impression: {report.get('impression','')}\n\n"
        "Respond with a JSON object:\n"
        '  "answer": string\n'
        '  "evidence": list of strings (verbatim finding texts used)\n'
        '  "source_report_id": string\n'
        "Output ONLY valid JSON."
    )

    try:
        raw = _call_llm(system, question)
    except Exception as exc:
        return {
            "answer": (
                f"Unable to reach the AI provider ({type(exc).__name__}). "
                f"Report impression: {report.get('impression', '')}"
            ),
            "source_report_id": report.get("id"),
            "evidence": [f.get("observation", "") for f in report.get("findings", [])],
            "error": "llm_provider_error",
        }

    try:
        data = _parse_llm_json(raw)
        answer = data.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("missing/invalid answer")
        evidence = data.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = [evidence]
        # Deterministic grounding: evidence must be quoted from the report.
        # If the model quoted nothing verifiable, use the report's own findings.
        grounded = _ground_evidence(report, [str(e) for e in evidence])
        if not grounded:
            grounded = [f.get("observation", "") for f in report.get("findings", [])]
        return {
            "answer": answer,
            "evidence": grounded,
            "source_report_id": report.get("id"),
        }
    except Exception:
        return {
            "answer": f"Unable to process follow-up (AI validation failed). Report impression: {report.get('impression','')}",
            "source_report_id": report.get("id"),
            "evidence": [f.get("observation", "") for f in report.get("findings", [])],
            "error": "llm_output_invalid",
        }
