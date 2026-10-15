from __future__ import annotations
import json
import os
import re
import threading
from pathlib import Path
from typing import Optional
from difflib import SequenceMatcher

DATA_DIR = Path(__file__).parent.parent / "data"

# Serialises read-modify-write cycles on the JSON files (sync endpoints run
# in FastAPI's threadpool, so two saves could otherwise interleave).
_WRITE_LOCK = threading.RLock()


def _load_patients() -> list[dict]:
    with open(DATA_DIR / "patients.json", encoding="utf-8") as f:
        return json.load(f)


def _load_reports() -> list[dict]:
    with open(DATA_DIR / "reports.json", encoding="utf-8") as f:
        return json.load(f)


# ── Persistence (report / patient writes) ────────────────────────────────────

def _patient_identity(record: dict) -> tuple:
    """Normalised identity tuple of a patient record (the patients.json schema
    minus the server-generated id): whitespace collapsed, name and gender
    case-folded, absent optional contact details as ``None``. All five fields
    must match before two records are treated as the same person."""
    def _text(field: str) -> str:
        return " ".join(str(record.get(field) or "").split())

    return (
        _text("name").casefold(),
        _text("dob"),
        _text("gender").casefold(),
        _text("phone") or None,
        _text("email") or None,
    )


def _next_id(records: list[dict], prefix: str) -> str:
    """Next unused ``<prefix>NNN`` id, derived from the ids on disk."""
    highest = 0
    for record in records:
        match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", str(record.get("id", "")))
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{prefix}{highest + 1:03d}"


def _append_record(filename: str, record: dict, prefix: str) -> dict:
    """
    Assign the next unique id and append one record to a top-level JSON array.

    The data files are hand-formatted (findings kept inline), so the file is
    extended surgically instead of re-serialised, so records already on disk
    stay byte-for-byte identical. The new content is parsed back before it
    replaces the file (temp file + os.replace), so a failed write can never
    corrupt the data. Caller must hold ``_WRITE_LOCK``.
    """
    path = DATA_DIR / filename
    raw = path.read_text(encoding="utf-8")
    current = json.loads(raw)  # existing content must parse before we touch it

    fields = {k: v for k, v in record.items() if k != "id"}  # ids are assigned here, never by callers
    ordered = {"id": _next_id(current, prefix), **fields}

    body = json.dumps(ordered, indent=2, ensure_ascii=False)
    indented = "\n".join(("  " + line) if line else "" for line in body.splitlines())
    close = raw.rfind("]")
    if close == -1:
        raise ValueError(f"{filename} is not a JSON array")
    head, tail = raw[:close].rstrip(), raw[close + 1:]
    if tail.strip() or not head.startswith("["):
        raise ValueError(f"{filename} has unexpected content outside its top-level array")
    separator = "\n" if head.endswith("[") else ",\n"
    new_raw = f"{head}{separator}{indented}\n]\n"

    parsed = json.loads(new_raw)  # verify before replacing
    if len(parsed) != len(current) + 1:
        raise ValueError(f"{filename} append verification failed")

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(new_raw, encoding="utf-8")
    os.replace(tmp, path)
    return ordered


def find_identical_patient(record: dict) -> Optional[dict]:
    """
    Exact-identity duplicate guard for new-patient creation.

    A candidate whose identity fields (name, dob, gender, phone, email) all
    match a stored patient is the same person, so retries and re-entries
    reuse the id already on disk instead of appending a duplicate. The match
    is exact (whitespace/case normalised only), never fuzzy: one differing
    field keeps the records distinct, so a genuinely different patient is
    never silently conflated with an existing one.
    """
    wanted = _patient_identity(record)
    for patient in _load_patients():
        if _patient_identity(patient) == wanted:
            return patient
    return None


def add_patient(record: dict) -> dict:
    """Append a new patient (id generated server-side) to patients.json."""
    with _WRITE_LOCK:
        return _append_record("patients.json", record, "P")


def find_identical_report(patient_id: str, draft_report: str) -> Optional[dict]:
    """Exact-content duplicate guard: same patient + same draft text."""
    draft = str(draft_report).strip()
    for r in _load_reports():
        if r.get("patient_id") == patient_id and str(r.get("draft_report", "")).strip() == draft:
            return r
    return None


def add_report_if_new(record: dict) -> tuple[dict, bool]:
    """
    Append a report unless an identical one (same patient, same draft text)
    is already stored: a retry or accidental double save returns the report
    that is already on disk instead of creating a second copy.

    Returns ``(report, created)``.
    """
    with _WRITE_LOCK:
        existing = find_identical_report(str(record.get("patient_id", "")), str(record.get("draft_report", "")))
        if existing is not None:
            return existing, False
        return _append_record("reports.json", record, "R"), True


def _token_similarity(query: str, name: str) -> float:
    """
    Token-aware name similarity.
    - Exact full-name match → 1.0
    - Token-prefix matching for partial names
    - Blend with SequenceMatcher for robustness
    """
    q_lower = query.lower().strip()
    n_lower = name.lower().strip()

    # Exact match shortcut
    if q_lower == n_lower:
        return 1.0

    q_tokens = q_lower.split()
    n_tokens = n_lower.split()

    # Count how many query tokens are prefixes of any name token
    prefix_hits = 0
    for qt in q_tokens:
        if any(nt.startswith(qt) for nt in n_tokens):
            prefix_hits += 1

    prefix_score = prefix_hits / max(len(q_tokens), 1)

    # Full string similarity
    seq_score = SequenceMatcher(None, q_lower, n_lower).ratio()

    # Weighted blend: prefix matching dominates for partial queries
    return 0.65 * prefix_score + 0.35 * seq_score


def _normalise_name(value) -> str:
    """Case-folded, whitespace-collapsed name with any trailing punctuation or
    possessive stripped ("Priya's." → "Priya"), so an AI-extracted name can be
    compared directly against the stored records."""
    text = " ".join(str(value or "").split()).casefold().rstrip(".,;:!?")
    for suffix in ("'s", "’s"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip(".,;:!?")
            break
    return text


def resolve_patient_name(name: str) -> tuple[str, list[dict]]:
    """
    Resolve the interpreter-extracted patient name against real records.

    The interpreter (Gemini, or the deterministic fallback) decides WHICH name
    the user meant; this function only validates and resolves that name
    against the patient database, in priority order:

      1. exact full name
      2. exact first name (single-token query)
      3. exact last name (single-token query)
      4. fuzzy/partial match, only reached when no exact tier matched

    The first tier with a hit decides the outcome, so an exact first name
    never escalates to ambiguity just because another patient's name *starts
    with* the same letters ("Priya" ≠ "Priyanka"). Ambiguity is raised only
    when the same tier genuinely matches 2+ patients (e.g. two patients named
    Priya, or a shared last name), so the caller must ask the user.

    Returns ``(status, matches)`` where status is one of:
      - ``"match"``:     exactly one patient resolved → safe to retrieve
      - ``"ambiguous"``: 2+ patients matched at this tier → user must choose
      - ``"none"``:      no patient matched

    The decision is made here, not in ``search_patients_by_name`` (the fuzzy
    candidate generator).
    """
    query = _normalise_name(name)
    if not query:
        return "none", []

    patients = _load_patients()

    def _outcome(matches: list[dict]) -> tuple[str, list[dict]]:
        return ("match" if len(matches) == 1 else "ambiguous"), matches

    # 1. Exact (case-insensitive) full-name match
    full = [p for p in patients if _normalise_name(p.get("name")) == query]
    if full:
        return _outcome(full)

    # 2 & 3. Single-token queries: exact first name, then exact last name
    tokens = query.split()
    if len(tokens) == 1:
        token = tokens[0]

        def _word(p: dict, index: int) -> str:
            words = _normalise_name(p.get("name")).split()
            return words[index] if words else ""

        by_first = [p for p in patients if _word(p, 0) == token]
        if by_first:
            return _outcome(by_first)
        by_last = [p for p in patients if _word(p, -1) == token]
        if by_last:
            return _outcome(by_last)

    # 4. Fuzzy / partial match: the only place the score-gap rule applies
    candidates = search_patients_by_name(query)
    if not candidates:
        return "none", []
    if len(candidates) >= 2 and candidates[0]["_score"] - candidates[1]["_score"] < 0.20:
        return "ambiguous", candidates
    return "match", [candidates[0]]


def search_patients_by_name(name: str) -> list[dict]:
    """
    Return patients whose name fuzzy-matches the query.
    Always includes an exact (case-insensitive) full-name match at score 1.0,
    flagged with ``_exact = True``.

    This is the candidate generator for the fuzzy tier only, so use
    ``resolve_patient_name`` to resolve a name against the database with the
    exact-match priorities. Threshold for inclusion: 0.45
    """
    patients = _load_patients()
    threshold = 0.45
    normalised_query = " ".join(name.split()).casefold()
    matches = []
    for p in patients:
        sim = _token_similarity(name, p["name"])
        if sim >= threshold:
            is_exact = " ".join(p["name"].split()).casefold() == normalised_query
            matches.append({**p, "_score": round(sim, 3), "_exact": is_exact})
    matches.sort(key=lambda x: x["_score"], reverse=True)
    return matches


def get_patient_by_id(patient_id: str) -> Optional[dict]:
    for p in _load_patients():
        if p["id"] == patient_id:
            return p
    return None


def get_patient_by_name(name: str) -> Optional[dict]:
    """Return the patient whose name case-insensitively equals ``name``.

    Used by the patient_info intent when the query names a patient without an
    open report (the report's patient is resolved by id in that case). Returns
    None when no patient matches; never invents a record.
    """
    wanted = " ".join(str(name).split()).casefold()
    if not wanted:
        return None
    for p in _load_patients():
        if " ".join(str(p.get("name", "")).split()).casefold() == wanted:
            return p
    return None


def get_reports_for_patient(patient_id: str, report_type: Optional[str] = None) -> list[dict]:
    """Return all reports for a patient, optionally filtered by type."""
    reports = _load_reports()
    result = [r for r in reports if r["patient_id"] == patient_id]
    if report_type:
        result = [r for r in result if r["type"].upper() == report_type.upper()]
    result.sort(key=lambda r: r["date"], reverse=True)
    return result


def get_latest_report_for_patient(patient_id: str, report_type: Optional[str] = None) -> Optional[dict]:
    """Return the most recent report for a patient."""
    reports = get_reports_for_patient(patient_id, report_type)
    return reports[0] if reports else None


def get_report_by_id(report_id: str) -> Optional[dict]:
    for r in _load_reports():
        if r["id"] == report_id:
            return r
    return None
