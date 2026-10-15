from __future__ import annotations
import os
import json
import logging
import re
import time
from typing import Optional
from pydantic import BaseModel, field_validator

logger = logging.getLogger("viscan.llm")

# ---------------------------------------------------------------------------
# Output schemas (validated after LLM call)
# ---------------------------------------------------------------------------

class InterpretedQuery(BaseModel):
    intent: str          # "greeting" | "assistant_identity" | "retrieve_report" |
                         # "dictation" | "follow_up" | "unsupported" | "unknown"
    patient_name: Optional[str] = None
    report_type: Optional[str] = None   # OPG / PERIAPICAL / BITEWING / CBCT
    free_text: Optional[str] = None
    # Set when the LLM was unavailable/malformed and a deterministic
    # fallback parser was used instead (surfaced to the user).
    fallback_reason: Optional[str] = None

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        allowed = {
            "greeting", "assistant_identity", "retrieve_report",
            "dictation", "follow_up", "unsupported", "unknown",
        }
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
    # Provenance / transparency (additive). fallback_reason is None only when
    # the output was genuinely AI-structured; otherwise it names the reason a
    # deterministic result was produced instead.
    fallback_reason: Optional[str] = None
    metadata: dict[str, str] = {}    # header lines stripped before structuring
    transcript: str = ""             # original transcript, echoed back verbatim

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


def _provider_name() -> str:
    """Name of the currently-configured LLM provider. Evaluated per call so
    logs always reflect which provider (if any) served the request."""
    if GEMINI_API_KEY:
        return "gemini"
    if OPENAI_API_KEY:
        return "openai"
    return "none"


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
# Transcript metadata split (header lines vs clinical text)
# ---------------------------------------------------------------------------

# Header keys a clinician may dictate as `key: value` lines before the
# clinical narrative. These are NEVER sent to the model and never parsed as
# clinical findings — they are echoed back verbatim as `metadata`.
_METADATA_KEYS = ("name", "patient", "dob", "date", "tooth", "age", "sex", "id")
_METADATA_LINE_RE = re.compile(
    r"^\s*(" + "|".join(_METADATA_KEYS) + r")\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE,
)


def _is_metadata_value(value: str) -> bool:
    """A metadata value is short and carries no sentence punctuation. This
    guards against swallowing clinical content that merely begins with a
    key-like word on a single dictated line (e.g. 'name: Priya. Bone loss…')."""
    if not value or len(value) > 60:
        return False
    return not re.search(r"[.!?]", value)


def _split_metadata(transcript: str) -> tuple[str, dict[str, str]]:
    """
    Split a dictation transcript into (clinical_text, metadata).

    Lines shaped like `name: …`, `dob: …`, `tooth: 26`, etc. are lifted out
    as metadata; everything else is clinical text that will be structured.
    The first occurrence of each key wins; later duplicates stay clinical.
    """
    metadata: dict[str, str] = {}
    clinical_lines: list[str] = []
    for line in transcript.splitlines():
        m = _METADATA_LINE_RE.match(line)
        if m:
            key, value = m.group(1).lower(), m.group(2).strip()
            if key not in metadata and _is_metadata_value(value):
                metadata[key] = value
                continue
        clinical_lines.append(line)
    return "\n".join(clinical_lines).strip(), metadata


def _is_metadata_finding(observation: str) -> bool:
    """True when an observation is *purely* a metadata header line (e.g.
    'name: Priya Sharma'). Belt-and-braces guard so a model-echoed header
    can never surface as a clinical finding."""
    m = _METADATA_LINE_RE.match(str(observation).strip())
    return bool(m) and _is_metadata_value(m.group(2).strip())


def _describe_llm_error(exc: Exception) -> str:
    """Human-readable provider-failure detail. HTTP errors carry their status
    (e.g. 'HTTP 429 rate limited'); everything else falls back to the type."""
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    if status is not None:
        if status == 429:
            return "HTTP 429 rate limited"
        reason = (getattr(resp, "reason_phrase", "") or "").strip()
        return f"HTTP {status} {reason}".strip()
    return type(exc).__name__


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
    # Strip trailing request vocabulary ("Rahul report" → "Rahul",
    # "priya opg" → "Priya") so report-type words never join the name.
    while words and words[-1].casefold() in _NAME_STOPWORDS:
        words = words[:-1]
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

    # 3b. Bare "<Name> report" shorthand: "Rahul report", "Priya Sharma report",
    #     "open rahul report" (possessives/commands are handled above).
    for m in re.finditer(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:latest\s+|newest\s+|old(?:est)?\s+)?"
        r"(?:opg|cbct|bitewing|periapical|panoramic)?\s*reports?\b",
        text,
    ):
        name = _valid_name(m.group(1))
        if name:
            return name
    for m in re.finditer(
        r"\b([a-z][a-z]+)\s+(?:latest\s+|newest\s+|old(?:est)?\s+)?"
        r"(?:opg|cbct|bitewing|periapical|panoramic)?\s*reports?\b",
        text,
    ):
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

# ── Minimal conversational fallbacks ────────────────────────────────────────
# Deliberately tiny and deterministic: they recognise a BARE salutation or an
# explicit "who/what are you?" identity question, so the assistant can answer
# honestly even without the LLM (demo mode / provider failure). They are NOT a
# conversational parser — all other natural phrasing is the LLM's job.

_GREETING_RE = re.compile(
    r"(?i)^\s*(?:hi|hello|hey|howdy|hiya|greetings|good\s+(?:morning|afternoon|evening))"
    r"(?:\s+there)?[!.,\s]*$"
)

_ASSISTANT_IDENTITY_RE = re.compile(
    r"(?i)\b(?:who|what)\s+are\s+you\b"
    r"|\bare\s+you\s+(?:really\s+|actually\s+)?(?:a\s+|an\s+|the\s+)?"
    r"(?:gemini|chatgpt|gpt(?:[\s-]?[\d.]+)?|claude|copilot|llm|ai|bot)\b"
    r"|\byou\s+(?:really\s+|actually\s+)?(?:a\s+|an\s+|the\s+)?"
    r"(?:gemini|chatgpt|gpt(?:[\s-]?[\d.]+)?|claude|copilot|llm|ai|bot)\b"
)


def _demo_interpret_query(text: str) -> InterpretedQuery:
    """Heuristic intent parsing — no LLM. Used in demo mode and as the
    deterministic fallback when the LLM provider is unavailable.

    Deliberately conservative: it handles report retrieval, dictation cues,
    follow-up questions, bare salutations and explicit identity questions.
    It does NOT attempt to understand arbitrary conversation — that is the
    LLM's job.
    """
    # Detect report type (word-bounded, case-insensitive)
    report_type = None
    m = REPORT_TYPE_RE.search(text)
    if m:
        key = m.group(1).lower()
        report_type = REPORT_TYPE_KEYWORDS.get(key) or REPORT_TYPE_KEYWORDS.get(key.replace("-", " "))

    # Detect intent. Order matters:
    #   1. bare salutation        ("Hi", "Hello there!")
    #   2. identity question      ("Are you Gemini?", "Who are you?")
    #   3. explicit retrieval verbs  ("Open Priya's OPG report")
    #   4. questions                 ("What are the key findings?", "Summarise her report")
    #   5. dictation cues            ("Note: mild bone loss at 46")
    #   6. a patient name without a verb  ("Priya's latest report")
    if _GREETING_RE.match(text):
        intent = "greeting"
    elif _ASSISTANT_IDENTITY_RE.search(text):
        intent = "assistant_identity"
    elif _RETRIEVE_VERB_RE.search(text):
        intent = "retrieve_report"
    elif _QUESTION_RE.search(text):
        intent = "follow_up"
    elif _DICTATION_RE.search(text):
        intent = "dictation"
    else:
        intent = "unknown"

    patient_name = None
    if intent not in ("greeting", "assistant_identity"):
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
    Use the LLM (Gemini when configured) to classify and parse the user's
    message: greetings, assistant-identity questions, report retrieval,
    dictation, contextual follow-ups, or unsupported requests. Falls back to
    conservative heuristics in demo mode / on provider failure.
    Never fabricates patient data.
    """
    if DEMO_MODE:
        result = _demo_interpret_query(text)
        logger.info("chat: demo mode (no LLM call) — deterministic intent=%s", result.intent)
        return result

    ctx_str = ""
    if session_context:
        ctx_str = f"\n\nCurrent session context:\n{json.dumps(session_context, indent=2)}"

    system = (
        "You are the query interpreter for Viscan AI, a dental radiology assistant app. "
        "Given a user's free-text message, output ONLY a JSON object with these fields:\n"
        '  "intent": one of ["greeting","assistant_identity","retrieve_report",'
        '"dictation","follow_up","unsupported"]\n'
        '  "patient_name": string or null\n'
        '  "report_type": one of ["OPG","PERIAPICAL","BITEWING","CBCT"] or null\n'
        '  "free_text": the original user text\n'
        "Intent rules:\n"
        '- greeting: a salutation or small talk with no task request ("hi", "hello there")\n'
        '- assistant_identity: a question about who you are or which model powers you '
        '(e.g. "Are you Gemini?", "Who are you?")\n'
        "- retrieve_report: user wants to open/view/find a patient's radiology report "
        "(e.g. \"Rahul report\", \"open Rahul\")\n"
        "- dictation: user is dictating clinical findings to be structured\n"
        '- follow_up: user asks about the report currently open in the session '
        '("Give me a summary", "What are the key findings?")\n'
        '- unsupported: anything outside these tasks (general knowledge, coding help, chit-chat beyond greetings)\n'
        "Rules:\n"
        "- Extract patient_name ONLY when the user actually wrote it — never invent a name.\n"
        "- Use the session context to decide follow_up (a report is open) vs retrieve_report.\n"
        "Output ONLY valid JSON, no explanation."
        + ctx_str
    )

    provider = _provider_name()
    try:
        raw = _call_llm(system, text)
    except Exception as exc:
        # Provider error (network, auth, rate limit…) — deterministic fallback
        logger.warning(
            "chat: %s call failed (%s) — deterministic fallback used",
            provider, _describe_llm_error(exc),
        )
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
            logger.warning(
                "chat: %s returned a patient name absent from the user text — dropped",
                provider,
            )
            parsed.patient_name = None
        # - a name with no explicit verb is a retrieval request
        if parsed.intent == "unknown" and parsed.patient_name:
            parsed.intent = "retrieve_report"
        logger.info(
            "chat: classified via %s — intent=%s patient=%s report_type=%s",
            provider, parsed.intent, parsed.patient_name, parsed.report_type,
        )
        return parsed
    except Exception:
        # LLM returned invalid output — fall back deterministically
        logger.warning("chat: %s returned malformed output — deterministic fallback used", provider)
        result = _demo_interpret_query(text)
        result.free_text = text
        result.fallback_reason = "AI returned malformed output; used deterministic interpretation"
        return result


def _name_in_text(name: str, text: str) -> bool:
    """True when any word of the name (≥3 chars) appears in the user's text."""
    text_cf = text.casefold()
    return any(tok in text_cf for tok in name.casefold().split() if len(tok) >= 3)


def _fallback_dictation(
    clinical: str,
    reason: str,
    metadata: Optional[dict] = None,
    transcript: str = "",
) -> StructuredDictation:
    """
    Safe deterministic fallback when the LLM is unavailable or its output is
    malformed. Never invents findings: the clinical text itself is the only
    'observation', and clinician review is always required.
    `transcript` (the original, metadata included) is echoed for preservation.
    """
    return StructuredDictation(
        findings=[{"observation": clinical[:500], "site": "see transcript", "severity": "unspecified"}],
        anatomical_site="Could not parse — review required",
        draft_report=(
            "CLINICAL DICTATION DRAFT\n\n"
            f"[{reason} — manual transcription required]\n\n"
            f"Original transcript:\n{clinical}"
        ),
        uncertainties=[reason, "Manual review required"],
        requires_review=True,
        fallback_reason=reason,
        metadata=metadata or {},
        transcript=transcript or clinical,
    )


def _no_clinical_findings_result(
    transcript: str, metadata: dict
) -> StructuredDictation:
    """
    Deterministic result for a transcript that carried only header/metadata
    lines and no clinical narrative. Marked as NOT AI-structured.
    """
    return StructuredDictation(
        findings=[],
        anatomical_site="No clinical findings detected",
        draft_report=(
            "CLINICAL DICTATION DRAFT\n\n"
            "[No clinical findings detected in the transcript.]\n\n"
            "Only header/metadata lines were provided — no clinical narrative "
            "to structure. Dictate your observations and submit again."
        ),
        uncertainties=["No clinical findings detected — only header/metadata lines provided"],
        requires_review=True,
        fallback_reason="No clinical findings detected in the transcript (not AI-structured)",
        metadata=metadata,
        transcript=transcript,
    )


_GROUND_STOPWORDS = {
    # Pure function words + common report filler. Excluding these from a
    # finding's *significant* tokens lets reworded-but-supported findings
    # survive (the model often swaps prefab phrasing, not clinical nouns).
    "a", "an", "the", "and", "or", "of", "at", "in", "on", "to", "is", "are",
    "was", "were", "be", "been", "being", "with", "for", "from", "by", "as",
    "it", "its", "this", "that", "these", "those", "no", "not", "nor", "but",
    "if", "then", "than", "so", "such", "into", "onto", "over", "under",
    "also", "both", "all", "any", "there", "here", "without", "per",
    # Report filler / prefab verbs that get reworded freely
    "present", "presence", "noted", "note", "notes", "shows", "show", "shown",
    "appears", "appear", "appearing", "seen", "see", "visible", "visibility",
    "within", "normal", "limits", "limit", "demonstrated", "demonstrates",
    "demonstrate", "approximately", "about", "mm", "cm",
}


def _tokenise(text: str) -> set[str]:
    """Lowercased alphanumeric tokens (keeps tooth numbers like '26')."""
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _significant_tokens(text: str) -> list[str]:
    """Finding-side tokens worth matching: length ≥ 3 or a pure number,
    minus the filler stopwords."""
    return [
        t for t in re.findall(r"[a-z0-9]+", text.casefold())
        if (len(t) >= 3 or t.isdigit()) and t not in _GROUND_STOPWORDS
    ]


def _is_grounded(observation: str, transcript_tokens: set[str], transcript_norm: str) -> bool:
    """Token-overlap grounding: a finding is supported when a majority of its
    significant tokens appear in the transcript. Reworded-but-supported text
    survives; wholly-invented text does not."""
    obs_norm = " ".join(str(observation).casefold().split()).rstrip(".,;:!?")
    if not obs_norm:
        return False
    sig = _significant_tokens(observation)
    if not sig:
        # Degenerate case (all filler) — fall back to substring containment.
        return obs_norm in transcript_norm
    matched = sum(1 for t in sig if t in transcript_tokens)
    return matched >= 1 and (matched / len(sig)) >= 0.5


def _ground_findings(transcript: str, findings: list) -> tuple[list, int]:
    """
    Deterministic no-invention guard: keep only findings supported by the
    transcript (token-overlap, tolerant of rewording). Returns
    (kept_findings, dropped_count).
    """
    transcript_tokens = _tokenise(transcript)
    transcript_norm = " ".join(transcript.casefold().split())
    kept: list = []
    dropped = 0
    for item in findings:
        if _is_grounded(item.get("observation", ""), transcript_tokens, transcript_norm):
            kept.append(item)
        else:
            dropped += 1
    return kept, dropped


def _apply_metadata_finding_guard(result: StructuredDictation) -> StructuredDictation:
    """
    No-name-as-finding guard (belt and braces): drop any finding whose
    observation is *purely* a metadata header line (e.g. 'name: Priya Sharma'),
    so a patient name can never surface as a clinical finding. Noted in
    uncertainties whenever it fires.
    """
    kept: list = []
    dropped = 0
    for item in result.findings:
        if _is_metadata_finding(item.get("observation", "")):
            dropped += 1
        else:
            kept.append(item)
    if dropped:
        result.findings = kept
        result.uncertainties = list(result.uncertainties) + [
            f"{dropped} metadata line(s) discarded: patient header echoed as a finding"
        ]
    return result


def structure_dictation(transcript: str) -> StructuredDictation:
    """
    Use LLM to convert a clinical dictation transcript into structured JSON.
    Never invents clinical observations beyond what is in the transcript.

    Pipeline:
      1. Header/metadata lines (`name:`, `dob:`, `tooth:`, …) are split off
         FIRST — they are never sent to the model or the demo parser, and are
         echoed back as `metadata`.
      2. Metadata-only input short-circuits to an explicit "no clinical
         findings detected" result (marked as NOT AI-structured).
      3. The clinical text is structured (Gemini when configured, otherwise
         the deterministic demo parser).
      4. Deterministic safety passes: token-overlap grounding drops unsupported
         findings (visible in uncertainties), metadata-shaped findings are
         dropped, and the review gate is always forced on.
    """
    clinical, metadata = _split_metadata(transcript)

    # Metadata-only input → explicit, honest, non-AI result.
    if not clinical:
        logger.info("dictation: header/metadata-only transcript — explicit no-findings result (no LLM call)")
        return _no_clinical_findings_result(transcript, metadata)

    if DEMO_MODE:
        logger.info("dictation: demo mode (no LLM call) — deterministic structuring")
        result = _demo_structure_dictation(clinical)
        result.metadata = metadata
        result.transcript = transcript
        return _apply_metadata_finding_guard(result)

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
        "- Convey each observation faithfully using the words from the dictation; "
        "  you may tidy phrasing, but do NOT add, infer, or embellish clinical facts.\n"
        "- If the dictation is unclear, add the unclear point to uncertainties.\n"
        "- Do not add clinical conclusions not present in the dictation.\n"
        "- The draft_report must begin with the phrase 'CLINICAL DICTATION DRAFT'.\n"
        "- Write the draft as a short professional report with a FINDINGS heading\n"
        "  listing the observations, and end with a 'REVIEW STATUS:' line stating\n"
        "  that clinician confirmation is required before use.\n"
        "- Output ONLY valid JSON, no explanation."
    )

    provider = _provider_name()
    try:
        raw = _call_llm(system, clinical)
    except Exception as exc:
        logger.warning(
            "dictation: %s call failed (%s) — safe fallback used",
            provider, _describe_llm_error(exc),
        )
        reason = f"AI provider unavailable ({_describe_llm_error(exc)})"
        return _fallback_dictation(clinical, reason, metadata, transcript)

    try:
        data = _parse_llm_json(raw)
        structured = StructuredDictation(**data)
    except Exception:
        # Validation failed — return safe fallback
        logger.warning("dictation: %s output failed validation — safe fallback used", provider)
        return _fallback_dictation(
            clinical, "AI output failed validation", metadata, transcript
        )

    # ── Deterministic safety pass (LLM output never reaches the client raw) ──
    kept, dropped = _ground_findings(clinical, structured.findings)
    if dropped:
        logger.warning(
            "dictation: %d ungrounded finding(s) dropped from %s output", dropped, provider
        )
        if not kept:
            return _fallback_dictation(
                clinical,
                "AI output contained findings not present in the transcript",
                metadata,
                transcript,
            )
        structured.uncertainties = list(structured.uncertainties) + [
            f"{dropped} AI-generated finding(s) discarded: not present in the transcript"
        ]
        structured.findings = kept

    # No-name-as-finding: drop any metadata-shaped finding the model echoed.
    structured = _apply_metadata_finding_guard(structured)

    # Empty-findings safety net: never echo raw gibberish as a "draft".
    if not structured.findings:
        logger.warning("dictation: no grounded findings survived %s output — explicit no-findings result", provider)
        return _no_clinical_findings_result(transcript, metadata)

    logger.info("dictation: structured via %s — %d finding(s)", provider, len(structured.findings))

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
    # Provenance: genuinely AI-structured → no fallback_reason.
    structured.metadata = metadata
    structured.transcript = transcript
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
        logger.info("follow-up: demo mode (no LLM call) — deterministic answer for report %s", report.get("id"))
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

    provider = _provider_name()
    try:
        raw = _call_llm(system, question)
    except Exception as exc:
        logger.warning(
            "follow-up: %s call failed (%s) — deterministic evidence fallback used for report %s",
            provider, _describe_llm_error(exc), report.get("id"),
        )
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
            logger.warning(
                "follow-up: %s evidence not found in report %s — replaced with the report's own findings",
                provider, report.get("id"),
            )
            grounded = [f.get("observation", "") for f in report.get("findings", [])]
        logger.info("follow-up: answered via %s for report %s", provider, report.get("id"))
        return {
            "answer": answer,
            "evidence": grounded,
            "source_report_id": report.get("id"),
        }
    except Exception:
        logger.warning("follow-up: %s output failed validation — evidence fallback used for report %s", provider, report.get("id"))
        return {
            "answer": f"Unable to process follow-up (AI validation failed). Report impression: {report.get('impression','')}",
            "source_report_id": report.get("id"),
            "evidence": [f.get("observation", "") for f in report.get("findings", [])],
            "error": "llm_output_invalid",
        }
