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
    intent: str          # "greeting" | "assistant_identity" | "assistant_capabilities" |
                         # "about_app" | "small_talk" | "general_dental" |
                         # "retrieve_report" | "patient_info" | "dictation" |
                         # "follow_up" | "unavailable_action" | "unsupported" | "unknown"
    patient_name: Optional[str] = None
    report_type: Optional[str] = None   # OPG / PERIAPICAL / BITEWING / CBCT
    free_text: Optional[str] = None
    # Set when the LLM was unavailable/malformed and a deterministic
    # fallback parser was used instead (surfaced to the user).
    fallback_reason: Optional[str] = None
    # Fields relevant to a patient_info intent (which demographic detail the
    # user asked for: contact / dob / gender, or "all" for everything on file).
    info_fields: Optional[list[str]] = None

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        allowed = {
            "greeting", "assistant_identity", "assistant_capabilities", "about_app",
            "small_talk", "general_dental", "retrieve_report",
            "dictation", "follow_up", "patient_info", "unavailable_action",
            "unsupported", "unknown",
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
    # False → the transcript held no usable clinical content (gibberish,
    # unrelated question, stray voice input). No draft is produced for it and
    # the client must not offer a save (see _has_clinical_signal()).
    usable: bool = True

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
# clinical findings; they are echoed back verbatim as `metadata`.
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
# "open the opg report…", and generic references like "of this person").
_NAME_STOPWORDS = {
    "me", "the", "a", "an", "this", "that", "these", "those", "her", "his", "their",
    "my", "our", "your", "its", "it", "all", "any", "latest", "newest", "old",
    "report", "reports", "patient", "patients", "file", "files", "image", "images",
    "scan", "scans", "record", "records", "document", "documents", "chart", "name",
    "list", "summary", "details", "info", "information", "doctor", "dr", "clinic",
    # generic references to a human that are not names
    "person", "individual", "someone", "somebody", "anyone", "people", "gentleman",
    "lady", "individuals", "folks",
    # radiology / request vocabulary
    "opg", "cbct", "bitewing", "periapical", "panoramic", "recent", "recently",
    "please", "full", "most", "findings", "finding", "result", "results",
    # dental anatomy / clinical nouns are never patient names
    # ("invent findings for tooth 38" must never search a patient "Tooth")
    "tooth", "teeth", "molar", "molars", "premolar", "premolars", "incisor",
    "incisors", "canine", "canines", "wisdom", "root", "canal", "canals",
    "gum", "gums", "jaw", "sinus", "sinuses", "crown", "crowns", "filling",
    "fillings", "implant", "implants", "denture", "dentures", "nerve", "enamel",
    "plaque", "tartar", "calculus", "decay", "caries", "abscess", "lesion",
    "radiograph", "radiographs", "radiolucency", "radiopacity", "xray",
    # interrogatives / question scaffolding ("Who is Priya?", "Why?")
    "who", "what", "when", "where", "why", "which", "how", "is", "are", "was",
    "were", "do", "does", "did", "can", "could", "should", "would", "tell", "give",
    # prepositions / particle verbs that anchor or follow a name candidate
    # ("pull up the scan for rohan verma" → "Rohan Verma", never "Up")
    "up", "down", "out", "off", "back", "about", "into", "over", "under",
    "through", "again", "now", "here", "there", "then", "just", "only", "maybe",
}

# Single-word title-case name: "Priya Sharma"
_NAME_TITLE = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"


def _valid_name(candidate: str) -> Optional[str]:
    """Validate a captured name candidate; returns a normalised name or None."""
    candidate = " ".join(candidate.split())
    # Strip sentence punctuation first: "Why?" must never become a name.
    candidate = candidate.strip("?.!,;:()[]{}\"'“”‘’")
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

    # 5. A bare single word ("Priya"): a lone non-generic word is treated as a
    #    patient name. Generic vocabulary is rejected by the stopword/anchor
    #    lists inside _valid_name, so no patient-specific rule is needed.
    words = text.split()
    if len(words) == 1:
        return _valid_name(words[0])
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

# A patient-info request: the user wants a demographic or contact detail about
# a patient (phone, email, address, dob, age, gender) rather than a radiology
# report. Deliberately specific so ordinary report commands ("open Priya's OPG")
# are never re-routed here.
_PATIENT_INFO_RE = re.compile(
    r"\b(?:"
    r"contact(?:\s+(?:details?|info(?:rmation)?))?|phone(?:\s*(?:number|no\.?)?)?|mobile|cell"
    r"|call(?:\s+(?:him|her|them))?|email|e-mail|mailing\s+address|address"
    r"|date\s+of\s+birth|d\.?o\.?b\.?|birthday|age|gender|sex"
    r")\b",
    re.IGNORECASE,
)

# ── Conversational / non-report intents (deterministic side) ─────────────────
# Everything below recognises CLASSES of phrasing — any salutation-only
# sentence, any "what can you do" capability question, any dental knowledge
# question, any scheduling request — rather than individual phrases, so unseen
# wording routes the same way. The aim is that a non-report message never gets
# dressed up as a radiology request (and never inherits one canned answer).

# Intents that are pure conversation: they never carry a patient name and can
# never trigger a record lookup, whatever the text happens to contain.
NON_PATIENT_INTENTS = {
    "greeting", "assistant_identity", "assistant_capabilities", "about_app",
    "small_talk", "general_dental", "unavailable_action",
}

# Salutation-only sentences: every word must be greeting vocabulary, and at
# least one must be a real salutation ("hi team", "Hey, good morning!" ✓,
# "hi, open Priya's OPG report" ✗ — it contains task words).
_GREETING_CORE = {
    "hi", "hello", "hey", "howdy", "hiya", "greetings", "yo", "morning",
    "afternoon", "evening", "night", "gday", "howzit",
}
_GREETING_VOCAB = _GREETING_CORE | {
    "hiya", "hii", "helloo", "heyy", "good", "day", "there", "everyone", "all",
    "team", "folks", "doctor", "doc", "dr", "sir", "madam", "mam", "ma'am",
    "welcome", "again", "friend", "friends", "guys", "pal",
}


def _is_bare_greeting(text: str) -> bool:
    """True when the message is ONLY salutations/pleasantries, with no task."""
    words = re.findall(r"[a-z']+", text.casefold())
    if not words or not any(w in _GREETING_CORE for w in words):
        return False
    return all(w in _GREETING_VOCAB for w in words)


# "What can you do?" — capability/feature questions about the assistant.
# Deliberately requires a question word + "you", so report commands that use
# the word "can" ("Can you open Priya's OPG?") are never captured here.
_ASSISTANT_CAPABILITIES_RE = re.compile(
    r"(?i)\b(?:"
    r"(?:what|which|how)(?:'s|\s+(?:else|else's))?\s+(?:can|could|do|does|should)\s+you\b"
    r"|what\s+(?:can|could)\s+you\s+(?:do|help|handle|assist|support|answer|manage|process|offer)\b"
    r"|what\s+(?:do|does)\s+you\s+do\b"
    r"|what\s+are\s+your\s+(?:capabilities|abilities|features|skills|functions|options)\b"
    r"|what\s+are\s+you\s+(?:capable|able)\s+of\b"
    r"|how\s+(?:can|do)\s+you\s+(?:help|assist|be\s+of\s+help|support\s+me)\b"
    r"|(?:your|the)\s+(?:capabilities|abilities|feature\s+list)\b"
    r")"
)

# "Who/what are you?" — identity / which model powers you.
_ASSISTANT_IDENTITY_RE = re.compile(
    r"(?i)\b(?:who|what)\s+are\s+you\b"
    # "tell me who you are" / "who you are" — same identity question, reworded
    r"|(?:tell\s+me\s+)?(?:who|what)\s+you\s+are\b"
    # "Which model powers you?" / "what model are you running?"
    r"|\b(?:which|what)\s+model\b[^?.!]{0,30}\byou\b"
    r"|\bare\s+you\s+(?:really\s+|actually\s+)?(?:a\s+|an\s+|the\s+)?"
    r"(?:gemini|chatgpt|gpt(?:[\s-]?[\d.]+)?|claude|copilot|llm|ai|bot)\b"
    r"|\byou\s+(?:really\s+|actually\s+)?(?:a\s+|an\s+|the\s+)?"
    r"(?:gemini|chatgpt|gpt(?:[\s-]?[\d.]+)?|claude|copilot|llm|ai|bot)\b"
)

# "What is Viscan AI?" — questions about the app/product itself. Requires the
# app to be named or an "…app/application" object, so a clinical question
# ("What is this finding?") can never land here.
_ABOUT_APP_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:what|who|which|how|is|are|does|do|can|could|should|would|tell)\b"
    r"[^?.!]{0,60}\bviscan(?:\s+ai)?\b"
    r"|\btell\s+me\s+about\b[^?.!]{0,40}\b(?:viscan|app|application|software|platform|system)\b"
    r"|what\s+(?:is|are)\s+(?:the\s+|this\s+)?(?:app|application|software)\b"
    r"|what\s+does\s+(?:this|the\s+app|viscan(?:\s+ai)?)\b[^?.!]{0,20}\bdo\b"
    r"|how\s+(?:do|does)\s+(?:this|the\s+app|viscan(?:\s+ai)?)\b[^?.!]{0,20}\bwork"
    r"|\babout\s+(?:this|the)\s+(?:app|application|software|platform|system)\b"
    r")"
)

# Outbound actions the app cannot perform (mailing/sharing a report out).
# Checked before patient_info: "email this report to Dr Rao" is an action the
# app cannot run, not a request for a patient's email address.
_OUTBOUND_ACTION_RE = re.compile(
    r"(?i)\b(?:(?:email|e-?mail|mail|send|share|forward|post)\s+"
    r"(?:\w+\s+){0,3}(?:report|reports|file|files|scan|scans|image|images|"
    r"attachment|results?|summary|it|them)\b)"
)

# Scheduling / record-management actions that are NOT implemented in this app.
# Verbs (and the appointment nouns they imply): booking a scan, rescheduling,
# deleting a record, etc. Never treated as report retrieval.
_APP_ACTION_RE = re.compile(
    r"(?i)\b(?:"
    r"book(?:s|ed|ing)?|schedul(?:e|ed|es|ing)|reschedul(?:e|ed|es|ing)|"
    r"cancel(?:l?(?:ed|ing|s))?|arrange(?:d|s|ment|ments)?|set\s+up|make\s+(?:an?\s+)?appointment|"
    r"appointment[s]?|time\s+slots?|slots?|delet(?:e|es|ed|ing)|archiv(?:e|es|ed|ing)"
    r")\b"
)

# Harmless social conversation: jokes, thanks, farewells, "how are you".
# Recognised as a class (any joke request / pleasantry), never a report task.
_SMALL_TALK_RE = re.compile(
    r"(?i)\b(?:"
    r"joke[s]?|funny|laugh(?:ing|ed)?|lol|lmao|haha|hehe|pun\b"
    r"|how(?:'s|\s+is|\s+are|\s+does)?\s+(?:it\s+going|you(?:\s+doing|\s+guys|\s+all)?|things|life)\b"
    r"|thank(?:s|you|you\s+very\s+much)|appreciate|cheers\b"
    r"|goodbye|good\s*bye|bye+|see\s+ya|see\s+you|take\s+care|good\s+night|goodnight"
    r"|what(?:'s|\s+is)\s+new|how(?:'s|\s+is)\s+your\s+day"
    r"|are\s+we\s+friends|miss\s+me"
    r")"
)

# Words that unambiguously refer to a stored report / its content. Used for
# report-grounded questions ("What are the key findings?") and for the
# "<Name> report" retrieval shorthand ("Rahul report").
_REPORT_CUE_RE = re.compile(
    r"(?i)\b(?:"
    r"reports?|findings?|impression|summari[sz]\w*|summary|summation|"
    r"results?|radiographs?|x-?rays?|study|studies"
    r")\b"
)

# A retrieval verb needs a target (a patient, a report type, or a report
# noun), so "how do I get rid of plaque?" is never a retrieval attempt.
_RETRIEVAL_TARGET_RE = re.compile(
    r"(?i)\b(?:"
    r"reports?|records?|scans?|images?|files?|chart|radiographs?|x-?rays?|films?|"
    r"opg|orthopantomogram|panoram\w*|periapical|bitewings?|cbct|cone[\s-]beam|patients?"
    r")\b"
)

# A demonstrative/pronoun right next to a report noun: the user is pointing at
# the open report ("What does this OPG show?"), which is report-grounded.
_DEICTIC_REPORT_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:this|that|these|those|it|they|them|her|his)\b[\s\w]{0,30}\b"
    r"(?:reports?|scans?|images?|radiographs?|x-?rays?|findings?|studies|"
    r"opg|cbct|bitewings?|periapical|panoram\w*)\b"
    r"|\b(?:reports?|scans?|images?|radiographs?|findings?)\b[\s\w]{0,20}\b"
    r"(?:it|this|that|them)\b"
    r")"
)

# General dental / oral-health vocabulary for EDUCATIONAL questions. Deliberately
# excludes ultra-generic report words ("normal", "findings", "pain" alone), so
# "Is this normal?" about an open report stays a report question.
_GENERAL_DENTAL_RE = re.compile(
    r"(?i)\b(?:"
    r"teeth|tooth|toothbrush|toothpaste|gum[s]?|gingiv\w*|gingivitis|"
    r"cavit(?:y|ies)|caries|decay|plaque|tartar|calculus|floss(?:ing)?|brush(?:ing)?|"
    r"dentist[s]?|dental|orthodont\w*|braces|retainer[s]?|"
    r"wisdom\s+teeth|root\s+canal|canals?|pulp\b|enamel|dentin\b|"
    r"sensitivit\w+|sensitive|bleed(?:ing|s)?|swelling|abscess|"
    r"periodont\w*|halitosis|breath|"
    r"whitening|scaling|polishing|fillings?|crowns?|bridges?|implants?|dentures?|"
    r"extraction[s]?|root\s+planing|deep\s+cleaning|"
    r"tmj|jaw|mandible|maxilla|sinus(?:es)?|"
    r"x-?rays?|radiograph\w*|opg|cbct|bitewings?|periapical|panoram\w*|"
    r"impacted|impaction|erupt\w*|numbness|toothache|ache\b|"
    r"treatment|diagnos[ei]s|infection|disease"
    r")\b"
)

# Patient-info requests need a patient anchor (a name, a possessive, "the
# patient") or an unambiguous contact token, so general questions that merely
# contain "age", "sex" or "call" ("At what age do wisdom teeth come in?",
# "Should I call a dentist?") are never patient lookups.
_STRONG_CONTACT_RE = re.compile(
    r"(?i)\b(?:"
    r"contact(?:\s+(?:details?|info(?:rmation)?))?|phone(?:\s*(?:number|no\.?)?)?|mobile|cell|"
    r"e-?mail|mailing\s+address|address|date\s+of\s+birth|d\.?o\.?b\.?|birthday|"
    r"reach(?:\s+(?:him|her|them))?"
    r")\b"
)

_PATIENT_CONTEXT_RE = re.compile(
    r"(?i)\b(?:patient|person|him|her|them|his|they|this\s+one|one)\b|['’]s\b"
)


def _demo_patient_info_fields(text: str) -> list[str]:
    """
    Deterministically classify which patient detail(s) a request asks for.
    Returns a subset of {"contact","dob","gender"}; "contact" covers any
    reach-them detail (phone, mobile, email, address, "how do I reach X").
    """
    fields: list[str] = []
    if re.search(
        r"\b(?:contact|phone|mobile|cell|email|e-mail|mail|address|reach|call)\b",
        text, re.IGNORECASE,
    ):
        fields.append("contact")
    if re.search(r"\b(?:date\s+of\s+birth|d\.?o\.?b\.?|birthday)\b", text, re.IGNORECASE):
        fields.append("dob")
    if re.search(r"\b(?:gender|sex|age)\b", text, re.IGNORECASE):
        fields.append("gender")
    return fields


def _demo_interpret_query(
    text: str, session_context: Optional[dict] = None
) -> InterpretedQuery:
    """Heuristic intent parsing: no LLM. Used in demo mode and as the
    deterministic fallback when the LLM provider is unavailable.

    Conservative about facts (it never invents patient data) but broad about
    CLASSES of conversation: it separates greetings, identity, capabilities,
    app questions, small talk, general dental knowledge, unimplemented actions,
    report retrieval, report questions and dictation. A non-report message is
    therefore answered as a conversation instead of being pushed at a
    radiology-request handler. Anything it cannot place stays `unknown` and the
    caller answers with scope guidance.
    """
    # Detect report type (word-bounded, case-insensitive)
    report_type = None
    m = REPORT_TYPE_RE.search(text)
    if m:
        key = m.group(1).lower()
        report_type = REPORT_TYPE_KEYWORDS.get(key) or REPORT_TYPE_KEYWORDS.get(key.replace("-", " "))

    # Extracted once, up front: the retrieval rules need it, while every
    # conversation intent discards it (it never targets a record).
    patient_name = _extract_patient_name(text)
    active_report = bool(
        session_context and session_context.get("active_report_id")
    )
    is_question = bool(_QUESTION_RE.search(text))

    has_retrieval_verb = _RETRIEVE_VERB_RE.search(text) is not None
    # A verb inside a QUESTION needs a report noun/type to count as retrieval
    # ("could you pull up the latest scan for rohan verma?" ✓, but "how do I get
    # rid of plaque?" must stay a dental question). Outside a question the
    # extracted name is enough ("open priya").
    has_report_target = bool(
        report_type or _RETRIEVAL_TARGET_RE.search(text) or _REPORT_CUE_RE.search(text)
    )
    is_retrieval = (
        (has_retrieval_verb and (has_report_target or (bool(patient_name) and not is_question)))
        # "Rahul report" / "Priya Sharma report" — a name with a report noun
        or (patient_name and (report_type or _REPORT_CUE_RE.search(text)))
    )

    # Detect intent. Order matters:
    #   1. bare salutation          ("Hi", "Hey, good morning!")
    #   2. capabilities question    ("What can you do?")
    #   3. identity question        ("Are you Gemini?", "Who are you?")
    #   4. app question             ("What is Viscan AI?")
    #   5. unimplemented action     ("email this report to…", "book a scan…")
    #   6. patient details          ("Priya's phone number")
    #   7. report-pointing question ("What does this OPG show?")
    #   8. report retrieval         ("Open Priya's OPG report", "Rahul report")
    #   9. small talk               ("Tell me a joke", "thanks!")
    #  10. dictation cue            ("Note: mild bone loss at 46")
    #  11. report-grounded question ("What are the key findings?")
    #  12. other questions          → dental / patient / contextual / unsupported
    #  13. a patient name without a verb  ("Priya's latest report")
    if _is_bare_greeting(text):
        intent = "greeting"
    elif _ASSISTANT_CAPABILITIES_RE.search(text):
        intent = "assistant_capabilities"
    elif _ASSISTANT_IDENTITY_RE.search(text):
        intent = "assistant_identity"
    elif _ABOUT_APP_RE.search(text):
        intent = "about_app"
    elif _OUTBOUND_ACTION_RE.search(text):
        # Before patient_info: "email this report to Dr Rao" is an action the
        # app cannot run, not a request for a patient's email address.
        intent = "unavailable_action"
    elif _PATIENT_INFO_RE.search(text) and (
        _STRONG_CONTACT_RE.search(text)
        or _PATIENT_CONTEXT_RE.search(text)
        or patient_name
    ):
        # Must precede the retrieval-verb rule: "get me Priya's contact details"
        # contains "get", but the user wants her contact info, not a report.
        intent = "patient_info"
    elif _APP_ACTION_RE.search(text):
        # Booking/scheduling/deleting is not implemented, and it must never be
        # mistaken for retrieval just because a name follows it
        # ("book a scan for Priya").
        intent = "unavailable_action"
    elif is_question and _DEICTIC_REPORT_RE.search(text):
        # Pointing at a report ("What does this OPG show?") is a report-grounded
        # question, not a fresh retrieval — even though the sentence contains a
        # retrieval-looking verb such as "show".
        intent = "follow_up"
    elif is_retrieval:
        intent = "retrieve_report"
    elif _SMALL_TALK_RE.search(text):
        intent = "small_talk"
    elif _DICTATION_RE.search(text) and not is_question:
        intent = "dictation"
    elif _REPORT_CUE_RE.search(text):
        # Report wording with no patient name attached: a question about a
        # (typically open) report, not a new retrieval request.
        intent = "follow_up"
    elif is_question:
        if _GENERAL_DENTAL_RE.search(text):
            # General dental knowledge: educational, NOT a patient finding.
            # Checked before the name/report fallbacks so a dental question is
            # never turned into a record lookup.
            intent = "general_dental"
        elif patient_name:
            # The question names a patient → their record ("Who is Priya?")
            intent = "retrieve_report"
        elif active_report:
            # Contextual follow-up while a report is open
            intent = "follow_up"
        else:
            intent = "unsupported"
    elif _DICTATION_RE.search(text):
        intent = "dictation"
    elif patient_name:
        intent = "retrieve_report"
    else:
        intent = "unknown"

    # Conversation never targets a record: drop any name-shaped token the
    # extractor happened to pick up out of chit-chat.
    if intent in NON_PATIENT_INTENTS:
        patient_name = None

    return InterpretedQuery(
        intent=intent,
        patient_name=patient_name,
        report_type=report_type,
        free_text=text,
        info_fields=(
            _demo_patient_info_fields(text) if intent == "patient_info" else None
        ),
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

# Bare FDI tooth numbers: dictation often drops the word "tooth"
# ("bone loss at 46"), which would otherwise leave the finding labelled with a
# site it never mentioned. Restricted to the FDI range behind a dental cue and
# with measurement/percentage units excluded, so "at 46 mm" or "46%" is never
# mistaken for a tooth.
_BARE_TOOTH_RE = re.compile(
    r"(?i)(?:#|\b(?:at|in|on|of)\b)\s*#?\s*(1[1-8]|2[1-8]|3[1-8]|4[1-8])\b"
    r"(?!\s?(?:%|mm|cm|year|yr|month|day|age))"
)

# Vocabulary that only a genuine dental-radiology dictation contains. Used by
# the usability gate (_has_clinical_signal): nothing here is a generic word, so
# questions/chatter/typos ("what the dark report for 2:36@") match nothing.
_CLINICAL_SIGNAL_RE = re.compile(
    r"\b(?:"
    # imaging / modality
    r"radiograph\w*|radiolucen\w+|radiopaque\w*|opg|panoram\w+|cbct|bitewing|periapical|x-?rays?"
    # teeth, supporting structures and findings
    r"|caries|cavit(?:y|ies)|decay|crown|tooth|teeth|molar|premolar|incisor|canine|wisdom"
    r"|root|canal|pulp\w*|gingiv\w*|periodont\w+|alveolar|bone|calculus|plaque"
    r"|mobility|furcation|abscess|cyst|lesion|impacted|erupt\w*|resorption|fracture"
    r"|widened|sclerosis|apical|lucency|opacit\w+|trabecular|defect"
    r"|sinus|implant|prosthe\w+|restor\w+|filling|extraction|denture|orthodont\w+"
    r"|mandible|maxilla|jaw|quadrant|arch"
    # symptoms, findings, severity and measurement
    r"|pain|tender\w*|sensitivit\w+|sensitive|bleeding|swelling|symptom\w*"
    r"|findings?|abnormalit\w+|unremarkable|normal|within normal limits"
    r"|no (?:new |significant |other )?(?:issues|concerns|complaints|symptoms)"
    r"|mild|moderate|severe|slight|marked|significant|millimet\w+|mm"
    r")\b",
    re.I,
)


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
    for m in _BARE_TOOTH_RE.finditer(text):
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
    Deterministic dictation structuring, demo mode only.
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
        "Demo mode: deterministic parsing only, no AI interpretation performed",
        "All fields unverified, clinician confirmation required",
    ]
    if not overall_site:
        uncertainties.append("Anatomical site not stated in the transcript")
    if any(f["severity"] == "unspecified" for f in findings):
        uncertainties.append("Severity not stated for some findings")

    anatomical_site = overall_site or "unspecified (please review)"

    findings_block = "\n".join(
        f"{i}. [{f['site']}] (severity: {f['severity']}): {f['observation']}"
        for i, f in enumerate(findings, start=1)
    )

    draft_report = (
        "CLINICAL DICTATION DRAFT\n\n"
        "[DEMO MODE: deterministic parser, no AI interpretation]\n"
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
    message: greetings, assistant identity/capabilities, app questions, small
    talk, general dental knowledge, unimplemented actions, report retrieval,
    dictation, report-grounded and contextual follow-ups, or unsupported
    requests. Falls back to conservative heuristics in demo mode / on provider
    failure. Never fabricates patient data.
    """
    if DEMO_MODE:
        result = _demo_interpret_query(text, session_context)
        logger.info("chat: demo mode (no LLM call); deterministic intent=%s", result.intent)
        return result

    ctx_str = ""
    if session_context:
        ctx_str = f"\n\nCurrent session context:\n{json.dumps(session_context, indent=2)}"

    system = (
        "You are the query interpreter for Viscan AI, a dental radiology assistant app. "
        "Given a user's free-text message, output ONLY a JSON object with these fields:\n"
        '  "intent": one of ["greeting","assistant_identity","assistant_capabilities",'
        '"about_app","small_talk","general_dental","retrieve_report",'
        '"dictation","follow_up","patient_info","unavailable_action","unsupported"]\n'
        '  "patient_name": string or null\n'
        '  "report_type": one of ["OPG","PERIAPICAL","BITEWING","CBCT"] or null\n'
        '  "info_fields": list of ["contact","dob","gender"] or null\n'
        '  "free_text": the original user text\n'
        "Intent rules:\n"
        '- greeting: a salutation with no task request ("hi", "hello there", '
        '"good morning")\n'
        '- assistant_identity: a question about who you are or which model powers you '
        '("Who are you?", "Are you Gemini?", "Are you an AI?")\n'
        '- assistant_capabilities: a question about what you can do or which features '
        'exist ("What can you do?", "What else can you do?", "How do you help?")\n'
        '- about_app: a question about what Viscan AI / this application is or does '
        '("What is Viscan AI?", "Tell me about this app")\n'
        '- small_talk: harmless social chatter with no request for information or work '
        '("tell me a joke", "how are you?", "thanks!", "bye")\n'
        '- general_dental: general dental knowledge — how teeth, gums, conditions, '
        'treatments or dental terms work IN GENERAL ("What is gingivitis?", "Why do my '
        'gums bleed?", "Is a root canal painful?"). This is education about dentistry, '
        'NOT about a patient and NOT about the open report; never classify it as '
        'retrieve_report or follow_up.\n'
        "- retrieve_report: user wants to open/view/find a patient's radiology report "
        "(e.g. \"Rahul report\", \"open Rahul\")\n"
        "- patient_info: user asks for a patient's demographic or contact details, such as "
        "phone, email, address, date of birth, age or gender "
        "(e.g. \"give me the contact details of this person\", \"what is Priya's "
        "phone number?\"). This is NOT a request for a radiology report. Set "
        "\"info_fields\" to the requested details from this list: \"contact\" "
        "(phone/email/address/how to reach them), \"dob\" (date of birth/age) "
        "and/or \"gender\"; use [\"contact\"] when the user does not specify.\n"
        "- dictation: user is dictating clinical findings to be structured\n"
        '- follow_up: user asks about the report currently open in the session '
        '("Give me a summary", "What are the key findings?", "What does this OPG show?"). '
        'Use the session context to decide follow_up (a report is open and the question '
        'is about it) vs retrieve_report.\n'
        '- unavailable_action: user asks the app to PERFORM an action it cannot: booking '
        'or scheduling a scan/appointment, cancelling or rescheduling, sending or '
        'emailing a report, deleting records, referrals, payments. These are NOT '
        'retrieve_report requests, even when a patient name follows ("book a scan for '
        'Priya"). Retrieving reports, answering questions about an open report, '
        'patient contact lookup, dictation structuring and draft export ARE implemented '
        'and must not use this intent.\n'
        '- unsupported: requests outside these tasks (writing poems, code, essays, or '
        'general non-dental knowledge)\n'
        "Rules:\n"
        "- Extract patient_name ONLY when the user actually wrote it; never invent a name.\n"
        "- Never classify a greeting, joke, identity/capabilities/app question, general "
        "dental question, or unimplemented action as retrieve_report or follow_up.\n"
        "- Use the session context to decide follow_up (a report is open) vs retrieve_report.\n"
        "Output ONLY valid JSON, no explanation."
        + ctx_str
    )

    provider = _provider_name()
    try:
        raw = _call_llm(system, text)
    except Exception as exc:
        # Provider error (network, auth, rate limit…): deterministic fallback
        logger.warning(
            "chat: %s call failed (%s); deterministic fallback used",
            provider, _describe_llm_error(exc),
        )
        result = _demo_interpret_query(text, session_context)
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
                "chat: %s returned a patient name absent from the user text; dropped",
                provider,
            )
            parsed.patient_name = None
        # - conversation intents never target a record, whatever the model
        #   thought it saw in the text
        if parsed.intent in NON_PATIENT_INTENTS:
            parsed.patient_name = None
            parsed.info_fields = None
        # - a name with no explicit verb is a retrieval request
        if parsed.intent == "unknown" and parsed.patient_name:
            parsed.intent = "retrieve_report"
        logger.info(
            "chat: classified via %s; intent=%s patient=%s report_type=%s",
            provider, parsed.intent, parsed.patient_name, parsed.report_type,
        )
        return parsed
    except Exception:
        # LLM returned invalid output; fall back deterministically
        logger.warning("chat: %s returned malformed output; deterministic fallback used", provider)
        result = _demo_interpret_query(text, session_context)
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
        anatomical_site="Could not parse: review required",
        draft_report=(
            "CLINICAL DICTATION DRAFT\n\n"
            f"[{reason}: manual transcription required]\n\n"
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
            "Only header/metadata lines were provided; no clinical narrative "
            "to structure. Dictate your observations and submit again."
        ),
        uncertainties=["No clinical findings detected; only header/metadata lines provided"],
        requires_review=True,
        fallback_reason="No clinical findings detected in the transcript (not AI-structured)",
        metadata=metadata,
        transcript=transcript,
        usable=False,   # nothing clinical to structure → no savable draft
    )


def _has_clinical_signal(text: str) -> bool:
    """
    Deterministic usability gate for a dictation transcript.

    True only when the text carries vocabulary that genuinely belongs to a
    dental radiology dictation (site, tooth, finding, symptom, severity or
    imaging terms). A transcript without any of it (a stray question, typos,
    voice-input noise such as "what the dark report for 2:36@") is not a
    dictation and must never be turned into an editable, savable draft.
    """
    return _CLINICAL_SIGNAL_RE.search(text) is not None


def _unusable_transcript_result(
    transcript: str,
    metadata: dict[str, str],
    reason: str,
) -> StructuredDictation:
    """
    Honest result for a transcript with no usable clinical content: no
    findings, NO draft (so nothing can be edited or saved), and explicit
    guidance to dictate the observation and submit again.
    """
    return StructuredDictation(
        findings=[],
        anatomical_site="Not determinable: transcript not clinically usable",
        draft_report="",
        uncertainties=[
            reason,
            (
                "Dictate a clinical observation (site or tooth, finding, severity) "
                "and submit again. Nothing was drafted from this transcript."
            ),
        ],
        requires_review=True,
        fallback_reason=reason,
        usable=False,
        metadata=metadata,
        transcript=transcript,
    )


def _deterministic_structuring_fallback(
    clinical: str,
    metadata: dict,
    transcript: str,
    reason: str,
) -> StructuredDictation:
    """
    Honest last resort when the AI produced no usable findings for a transcript
    that DOES contain clinical narrative: structure that narrative
    deterministically (every observation stays verbatim from the transcript,
    nothing is invented) and label the result with the real reason it is not
    AI-structured. Unlike `_no_clinical_findings_result`, this never claims the
    input was header/metadata only, which would be false here.
    """
    result = _demo_structure_dictation(clinical)
    result = _apply_metadata_finding_guard(result)
    if not result.findings:
        # Nothing extractable at all (e.g. duplicate header lines only).
        return _no_clinical_findings_result(transcript, metadata)
    result.fallback_reason = reason
    result.uncertainties = [
        reason,
        "Deterministic (non-AI) structuring; observations are quoted verbatim "
        "from the transcript and were not interpreted",
        "All fields unverified, clinician confirmation required",
    ]
    result.draft_report = result.draft_report.replace(
        "[DEMO MODE: deterministic parser, no AI interpretation]",
        f"[{reason}]",
    )
    result.metadata = metadata
    result.transcript = transcript
    return result


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
        # Degenerate case (all filler): fall back to substring containment.
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
         FIRST: they are never sent to the model or the demo parser, and are
         echoed back as `metadata`.
      2. Metadata-only input short-circuits to an explicit "no clinical
         findings detected" result (marked as NOT AI-structured).
      3. A transcript with no clinical vocabulary at all (noise, an unrelated
         question) short-circuits to an unusable result: no findings, no draft,
         nothing savable; the same verdict can also come from the model
         itself via `"usable": false`.
      4. The clinical text is structured (Gemini when configured, otherwise
         the deterministic demo parser), with historical and current findings
         kept separate. An empty AI findings list triggers one corrective
         retry, then deterministic structuring of the same clinical text,
         never a bogus "no findings" result.
      5. Deterministic safety passes: token-overlap grounding drops unsupported
         findings (visible in uncertainties), metadata-shaped findings are
         dropped, and the review gate is always forced on.
    """
    clinical, metadata = _split_metadata(transcript)

    # Metadata-only input → explicit, honest, non-AI result.
    if not clinical:
        logger.info("dictation: header/metadata-only transcript; explicit no-findings result (no LLM call)")
        return _no_clinical_findings_result(transcript, metadata)

    # Usability gate (demo and live alike, before any LLM call): a transcript
    # with no clinical vocabulary at all is not a dictation, so it gets no
    # findings, no draft and nothing savable. Only guidance to try again.
    if not _has_clinical_signal(clinical):
        reason = "No clinical content detected in the transcript; nothing to structure"
        logger.info("dictation: transcript carries no clinical signals; no draft produced (no LLM call)")
        return _unusable_transcript_result(transcript, metadata, reason)

    if DEMO_MODE:
        logger.info("dictation: demo mode (no LLM call); deterministic structuring")
        result = _demo_structure_dictation(clinical)
        result.metadata = metadata
        result.transcript = transcript
        return _apply_metadata_finding_guard(result)

    system = (
        "You are a dental radiology assistant. A clinician has dictated findings. "
        "Your task: extract and structure EVERY clinical statement in the dictation. "
        "NEVER invent or infer clinical observations not explicitly mentioned.\n\n"
        "Output ONLY a JSON object with:\n"
        '  "findings": list of {"site": str, "observation": str, '
        '"severity": "none"|"mild"|"moderate"|"severe", '
        '"context": "historical"|"current"}\n'
        '  "anatomical_site": string summarising the primary region\n'
        '  "draft_report": a professional radiological report draft\n'
        '  "uncertainties": list of strings where the dictation was unclear or incomplete\n'
        '  "usable": true when the text is an actual clinical dictation, false otherwise\n'
        '  "requires_review": true (always set this to true)\n\n'
        "Rules:\n"
        '- Set "usable": false ONLY when the text contains no clinical observations at '
        '  all (gibberish, an unrelated question, small talk or stray voice input). In '
        "  that case return an empty \"findings\" list, an empty \"draft_report\" and "
        '  "usable": false; never turn non-clinical text into a report draft, however '
        "  uncertain it is.\n"
        "- Convey each observation faithfully using the words from the dictation; "
        "  you may tidy phrasing, but do NOT add, infer, or embellish clinical facts.\n"
        "- Structure historical findings too. When the dictation reports what a PREVIOUS "
        "  examination (e.g. a prior OPG or CBCT) documented, keep each prior finding as "
        '  its own entry with "context": "historical", naming the prior study/date in the '
        '  observation. Statements about the examination being performed now get '
        '"context": "current". Never merge a historical finding with a current one.\n'
        '- The findings list must NEVER be empty when the dictation contains clinical '
        "  statements. This includes dictations that only say no new findings are "
        '  available for the current exam: record that statement itself as a finding '
        'with severity "none" and context "current", alongside the historical findings.\n'
        "- Never claim a condition has improved, worsened, or remained unchanged unless "
        "  the dictation says so; never invent measurements, diagnoses, or treatment "
        "  recommendations.\n"
        "- Follow any instruction the dictation gives about how the report should be "
        "  written (e.g. separating historical from current findings, stating that "
        "  current findings require radiographic assessment and clinician review).\n"
        "- If the dictation is unclear, add the unclear point to uncertainties.\n"
        "- Do not add clinical conclusions not present in the dictation.\n"
        "- The draft_report must begin with the phrase 'CLINICAL DICTATION DRAFT'.\n"
        "- Write the draft as a short professional report with a FINDINGS heading that\n"
        "  separates HISTORICAL FINDINGS (prior studies) from CURRENT OBSERVATIONS,\n"
        "  and end with a 'REVIEW STATUS:' line stating that clinician confirmation is\n"
        "  required before use.\n"
        "- Output ONLY valid JSON, no explanation."
    )

    # Corrective nudge appended when the model answers with zero findings for a
    # transcript that plainly contains clinical narrative (common for follow-up
    # dictations that merely state "no new findings" while carrying prior-exam
    # findings).
    empty_nudge = (
        "\n\nREMINDER: the dictation above DOES contain clinical statements, so "
        '"findings" must not be empty. Structure every statement it contains: '
        'prior-examination findings as "historical", statements about the current '
        'examination as "current"; and when the dictation says no new findings are '
        'available, record that as a finding with severity "none" instead of '
        "returning an empty list."
    )

    provider = _provider_name()
    structured: Optional[StructuredDictation] = None
    for attempt in range(2):
        prompt = system + (empty_nudge if attempt else "")
        try:
            raw = _call_llm(prompt, clinical)
        except Exception as exc:
            logger.warning(
                "dictation: %s call failed (%s); safe fallback used",
                provider, _describe_llm_error(exc),
            )
            reason = f"AI provider unavailable ({_describe_llm_error(exc)})"
            return _fallback_dictation(clinical, reason, metadata, transcript)

        try:
            candidate = StructuredDictation(**_parse_llm_json(raw))
        except Exception:
            # Validation failed: return safe fallback
            logger.warning("dictation: %s output failed validation; safe fallback used", provider)
            return _fallback_dictation(
                clinical, "AI output failed validation", metadata, transcript
            )

        if not candidate.usable:
            # Belt and braces with the deterministic gate: either signal alone
            # suppresses the draft, so noise can never become a savable report.
            logger.info("dictation: %s marked the transcript unusable; no draft produced", provider)
            return _unusable_transcript_result(
                transcript,
                metadata,
                "The AI found no clinical observations in this transcript; nothing to structure",
            )
        if candidate.findings:
            structured = candidate
            break
        logger.warning(
            "dictation: %s returned an empty findings list (attempt %d of 2)",
            provider, attempt + 1,
        )

    if structured is None:
        # The model parsed cleanly but found nothing to structure in a transcript
        # that DOES contain clinical narrative. Never mislabel that as
        # "metadata only": structure it deterministically instead.
        reason = (
            "AI returned no findings for a transcript containing clinical text; "
            "used deterministic structuring"
        )
        logger.warning("dictation: %s yielded no findings; %s", provider, reason)
        return _deterministic_structuring_fallback(clinical, metadata, transcript, reason)

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

    # Empty-findings safety net: never echo raw gibberish as a "draft", and
    # never claim "metadata only" when the transcript had clinical narrative.
    if not structured.findings:
        reason = "AI output contained no usable findings after safety checks"
        logger.warning(
            "dictation: no grounded findings survived %s output; deterministic structuring used",
            provider,
        )
        return _deterministic_structuring_fallback(clinical, metadata, transcript, reason)

    logger.info("dictation: structured via %s; %d finding(s)", provider, len(structured.findings))

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
            "Not finalised. Do not save automatically."
        )
    if not structured.uncertainties:
        structured.uncertainties = ["AI-structured output: clinician verification required"]
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
        logger.info("follow-up: demo mode (no LLM call); deterministic answer for report %s", report.get("id"))
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
            "follow-up: %s call failed (%s); deterministic evidence fallback used for report %s",
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
                "follow-up: %s evidence not found in report %s; replaced with the report's own findings",
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
        logger.warning("follow-up: %s output failed validation; evidence fallback used for report %s", provider, report.get("id"))
        return {
            "answer": f"Unable to process follow-up (AI validation failed). Report impression: {report.get('impression','')}",
            "source_report_id": report.get("id"),
            "evidence": [f.get("observation", "") for f in report.get("findings", [])],
            "error": "llm_output_invalid",
        }


# ---------------------------------------------------------------------------
# General dental knowledge (educational, never patient-specific)
# ---------------------------------------------------------------------------

# Appended deterministically so the "general education, not a finding, not a
# diagnosis" framing holds even if the model's own wording drifts.
GENERAL_DENTAL_EDUCATION_NOTE = (
    "\n\n(General educational information about dental health — not a diagnosis, "
    "not personalized treatment advice, and not based on any patient's report. "
    "Only a dentist or clinician who examines you or the patient can advise on "
    "an individual case.)"
)


def _extract_answer_text(raw: str) -> Optional[str]:
    """Pull the prose answer out of a model response.

    Accepts either plain prose (what the education prompt asks for) or a JSON
    object carrying an "answer" field (what a classification-shaped response
    looks like). Returns None when the response carries no usable answer —
    e.g. another intent-classification object — so unusable model output can
    never be shown to the user as dental guidance.
    """
    if not raw or not raw.strip():
        return None
    try:
        data = _parse_llm_json(raw)
    except Exception:
        text = raw.strip()
        if text.startswith("```"):          # tolerate a stray markdown fence
            text = re.sub(r"^```[A-Za-z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip() or None
    answer = data.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    return None


def answer_general_dental(
    question: str, provider_error: Optional[str] = None
) -> str:
    """
    Answer a GENERAL dental question as educational information.

    Gemini (when configured) writes the explanation under a system prompt that
    forbids diagnoses, personalized treatment advice and any reference to
    patient records; the deterministic education note is appended here so the
    separation from patient-specific findings is explicit in every reply.

    Without a provider (demo mode, or the interpretation call already failed)
    nothing is invented: the reply says plainly that no explanation can be
    generated right now.
    """
    question = " ".join((question or "").split())[:300]

    if provider_error:
        # The provider already failed once for this request; don't make the
        # user wait for a second doomed call.
        return (
            "I couldn't reach the AI service right now, so I can't generate an "
            f"explanation for your dental question (\"{question}\"). Nothing was "
            "looked up or invented. Please try again in a moment — or ask me a "
            "question about an open report, which I answer from that report "
            "itself." + GENERAL_DENTAL_EDUCATION_NOTE
        )

    if DEMO_MODE:
        return (
            "That's a general dental question, and it would be answered as "
            f"general education only (\"{question}\") — clearly separate from any "
            "patient's findings. This session has no AI service configured, so I "
            "can't generate that explanation right now rather than make one up. "
            "For advice about your own teeth or a specific patient, please consult "
            "a dentist; I can still answer questions about an open report."
            + GENERAL_DENTAL_EDUCATION_NOTE
        )

    system = (
        "You are Viscan AI's dental education assistant. Answer the user's GENERAL "
        "dental question with accurate, plain-language educational information "
        "(one short paragraph).\n"
        "Rules:\n"
        "- This is general education, NOT a diagnosis and NOT personalized "
        "treatment advice. Never tell the user what they (or any patient) have, "
        "and never prescribe or recommend a specific treatment.\n"
        "- Do not reference, assume or invent any patient, report or clinical "
        "record; none has been provided to you.\n"
        "- Wherever there is medical uncertainty, say what is generally known and "
        "that a dentist or clinician should assess the individual case.\n"
        "- Never invent statistics, citations or study results.\n"
        "- If the question is really about a specific patient's report, say that "
        "report questions are answered from the report open in the app.\n"
        "- Plain text only: no JSON, no markdown headings."
    )

    provider = _provider_name()
    try:
        raw = _call_llm(system, question)
    except Exception as exc:
        logger.warning(
            "general-dental: %s call failed (%s); no answer generated",
            provider, _describe_llm_error(exc),
        )
        return (
            f"I couldn't reach the AI service ({_describe_llm_error(exc)}), so I "
            f"can't explain \"{question}\" right now. Nothing was invented. "
            "Please try again shortly." + GENERAL_DENTAL_EDUCATION_NOTE
        )

    answer = _extract_answer_text(raw)
    if not answer:
        logger.warning("general-dental: %s returned no usable answer", provider)
        return (
            "The AI returned an unusable response, so I don't have a reliable "
            f"explanation for \"{question}\" to give you. Nothing was invented. "
            "Please try again." + GENERAL_DENTAL_EDUCATION_NOTE
        )

    logger.info("general-dental: answered via %s", provider)
    return answer + GENERAL_DENTAL_EDUCATION_NOTE
