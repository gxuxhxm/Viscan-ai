"""
Tests for persisting edited dictation drafts (POST /reports).

All writes happen against a temp copy of the data directory, so the
repository's patients.json / reports.json are never mutated.
"""
from __future__ import annotations
import sys
import os
import json
import shutil
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
# Force demo mode for all tests, never make paid API calls
os.environ["OPENAI_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""

from fastapi.testclient import TestClient
from services import retrieval
from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Copy the real data files into a temp dir so persistence can be tested."""
    src = Path(__file__).parent.parent / "data"
    for name in ("patients.json", "reports.json"):
        shutil.copy(src / name, tmp_path / name)
    monkeypatch.setattr(retrieval, "DATA_DIR", tmp_path)
    return tmp_path


def _payload(**overrides):
    """A realistic save request, as the dictation UI sends it."""
    body = {
        "draft_report": (
            "CLINICAL DICTATION DRAFT\n\n"
            "FINDINGS:\n1. Mild bone loss at 46.\n\n"
            "IMPRESSION:\nEarly periodontitis at 46."
        ),
        "patient_id": "P001",
        "report_type": "OPG",
        "findings": [
            {"site": "Lower right molar (46)", "observation": "Mild bone loss at 46.", "severity": "mild"},
        ],
        "transcript": "Mild bone loss at 46.",
        "anatomical_site": "tooth 46",
        "tooth_number": "46",
        "clinician_confirmed": True,
        "demo_mode": True,
        "fallback_reason": None,
        "dictation_metadata": {"tooth": "46"},
    }
    body.update(overrides)
    return {k: v for k, v in body.items() if v is not None}


def _report_file_records(data_dir):
    return json.loads((data_dir / "reports.json").read_text(encoding="utf-8"))


# ── Confirmed save: schema, status and persistence ──────────────────────────

def test_save_confirmed_report_to_existing_patient(isolated_data):
    resp = client.post("/reports", json=_payload())
    assert resp.status_code == 200
    data = resp.json()
    # Same schema + id sequence as the existing records
    assert data["id"] == "R013"
    assert data["patient_id"] == "P001"
    assert data["type"] == "OPG"
    assert data["date"]
    # Clinician confirmation is an explicit field; no identity invented
    assert data["clinician_confirmed"] is True
    assert data["status"] == "final"
    assert data["radiologist"] == "Unassigned"
    # Original transcript + provenance preserved
    assert data["source_transcript"] == "Mild bone loss at 46."
    assert data["provenance"]["source"] == "dictation"
    assert data["provenance"]["demo_mode"] is True
    assert data["provenance"]["fallback_reason"] is None
    assert data["provenance"]["metadata"] == {"tooth": "46"}
    # Content and additive metadata
    assert data["draft_report"] == _payload()["draft_report"]
    assert data["impression"] == "Early periodontitis at 46."
    assert data["findings"][0]["observation"] == "Mild bone loss at 46."
    assert data["tooth_number"] == "46"
    # …and it is really on disk
    stored = retrieval.get_report_by_id("R013")
    assert stored is not None
    assert stored["draft_report"] == _payload()["draft_report"]
    assert stored["clinician_confirmed"] is True
    assert len(_report_file_records(isolated_data)) == 13


def test_save_keeps_historical_current_context(isolated_data):
    """A follow-up dictation's `context` tag must survive the save, so the
    report keeps prior-study findings separate from current observations."""
    payload = _payload(findings=[
        {"site": "General alveolar bone",
         "observation": "Generalized moderate horizontal alveolar bone loss.",
         "severity": "moderate", "context": "historical"},
        {"site": "Current examination",
         "observation": "No new radiographic findings have been provided.",
         "severity": "none", "context": "current"},
    ])
    resp = client.post("/reports", json=payload)
    assert resp.status_code == 200
    contexts = [f.get("context") for f in resp.json()["findings"]]
    assert contexts == ["historical", "current"]
    stored = retrieval.get_report_by_id(resp.json()["id"])
    assert [f.get("context") for f in stored["findings"]] == ["historical", "current"]


def test_saved_report_retrievable_through_existing_workflow(isolated_data):
    """The saved report flows through the normal patient/report endpoints
    and the conversational retrieval path."""
    rid = client.post("/reports", json=_payload()).json()["id"]

    detail = client.get(f"/reports/{rid}")
    assert detail.status_code == 200
    assert detail.json()["id"] == rid

    listed = client.get("/patients/P001/reports")
    assert listed.status_code == 200
    assert rid in [r["id"] for r in listed.json()]
    assert next(r for r in listed.json() if r["id"] == rid)["clinician_confirmed"] is True

    # Newest OPG for Priya Sharma is now the saved report
    query = client.post("/query", json={"text": "Open Priya Sharma's latest OPG report"}).json()
    assert query["report"] is not None
    assert query["report"]["id"] == rid


def test_save_without_confirmation_is_not_marked_confirmed(isolated_data):
    resp = client.post("/reports", json=_payload(clinician_confirmed=False))
    assert resp.status_code == 200
    data = resp.json()
    assert data["clinician_confirmed"] is False
    assert data["status"] == "draft"


def test_dictation_response_saves_as_structured_report(isolated_data):
    """End to end: the /dictation output (structured findings + readable draft)
    persists through the same schema, endpoint and id sequence as any other
    report, linked to the patient the clinician picked."""
    d = client.post("/dictation", json={
        "transcript": "Periapical radiolucency at tooth 26. Mild horizontal bone loss at 46."
    }).json()
    assert d["usable"] is True
    assert d["requires_review"] is True

    resp = client.post("/reports", json={
        "draft_report": d["draft_report"],
        "patient_id": "P001",
        "report_type": "OPG",
        "findings": d["findings"],
        "transcript": d["transcript"],
        "anatomical_site": d["anatomical_site"],
        "clinician_confirmed": True,
        "demo_mode": d["demo_mode"],
        "fallback_reason": d["fallback_reason"],
        "dictation_metadata": d["metadata"],
    })
    assert resp.status_code == 200
    data = resp.json()

    # Each finding keeps its observation, its own anatomical site and severity
    assert [(f["site"], f["observation"], f["severity"]) for f in data["findings"]] == [
        ("tooth 26", "Periapical radiolucency at tooth 26.", "unspecified"),
        ("tooth 46", "Mild horizontal bone loss at 46.", "mild"),
    ]
    # The readable draft and the dictated transcript are stored verbatim
    assert data["draft_report"] == d["draft_report"]
    assert data["source_transcript"] == d["transcript"]
    assert data["anatomical_site"] == d["anatomical_site"]
    # Correct patient linkage, retrievable through the existing workflow
    assert data["patient_id"] == "P001"
    assert data["id"] in [r["id"] for r in client.get("/patients/P001/reports").json()]


def test_dictation_unspecified_severity_and_site_survive_save_retrieve(isolated_data):
    """A dictation finding whose severity is 'unspecified' (no grade stated) and
    its anatomical site survive the save → retrieval round-trip unchanged, and
    no tooth number is invented for a descriptive site."""
    findings = [
        {"site": "lower right third molar", "observation": "Horizontally impacted",
         "severity": "unspecified", "context": "current"},
    ]
    resp = client.post("/reports", json={
        "draft_report": (
            "CLINICAL DICTATION DRAFT\nFINDINGS:\n"
            "- Lower right third molar: horizontally impacted.\n\n"
            "REVIEW STATUS: Requires clinician confirmation."
        ),
        "patient_id": "P003",  # Arjun Mehta
        "report_type": "OPG",
        "findings": findings,
        "transcript": "lower right third molar is horizontally impacted",
        "anatomical_site": "Lower right third molar",
        "clinician_confirmed": True,
        "demo_mode": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    f = data["findings"][0]
    # observation + anatomical site + severity all preserved, nothing invented
    assert f["observation"] == "Horizontally impacted"
    assert f["site"] == "lower right third molar"
    assert f["severity"] == "unspecified"
    assert data["anatomical_site"] == "Lower right third molar"
    assert data["tooth_number"] is None
    # retrievable through the patient's report list
    assert data["id"] in [r["id"] for r in client.get("/patients/P003/reports").json()]


def test_existing_report_records_untouched_after_append(isolated_data):
    """Appending must not reformat the hand-formatted records already on disk."""
    path = isolated_data / "reports.json"
    before = path.read_text(encoding="utf-8")
    client.post("/reports", json=_payload())
    after = path.read_text(encoding="utf-8")
    assert after.startswith(before[: before.rfind("]")].rstrip())
    assert after.endswith("]\n")
    assert len(json.loads(after)) == 13


def test_identical_resubmit_does_not_duplicate(isolated_data):
    first = client.post("/reports", json=_payload()).json()
    before = _report_file_records(isolated_data)
    second = client.post("/reports", json=_payload())
    assert second.status_code == 200
    assert second.json()["id"] == first["id"]        # same report returned…
    assert len(_report_file_records(isolated_data)) == len(before)  # …nothing appended


def test_sequential_ids_for_reports_and_patients(isolated_data):
    r1 = client.post("/reports", json=_payload()).json()["id"]
    r2 = client.post("/reports", json=_payload(draft_report="CLINICAL DICTATION DRAFT\n\nFINDINGS:\n1. Something else.")).json()["id"]
    assert (r1, r2) == ("R013", "R014")

    p1 = client.post("/reports", json=_payload(
        patient_id=None,
        patient={"name": "Asha Rao", "dob": "1990-01-01", "gender": "F"},
    )).json()["patient_id"]
    p2 = client.post("/reports", json=_payload(
        patient_id=None,
        patient={"name": "Vikram Rao", "dob": "1987-06-11", "gender": "M"},
        draft_report="CLINICAL DICTATION DRAFT\n\nFINDINGS:\n1. Something else.",
    )).json()["patient_id"]
    assert (p1, p2) == ("P007", "P008")


# ── New patient: schema, unique id, saved alongside the others ──────────────

def test_save_creates_new_patient_alongside_existing(isolated_data):
    resp = client.post("/reports", json=_payload(
        patient_id=None,
        patient={"name": "Test Patient", "dob": "1995-04-02", "gender": "F", "phone": "", "email": None},
    ))
    assert resp.status_code == 200
    data = resp.json()
    assert data["patient_id"] == "P007"

    # Shows up in the patient list next to the existing ones
    patients = client.get("/patients").json()
    assert len(patients) == 7
    created = next(p for p in patients if p["id"] == "P007")
    assert created["name"] == "Test Patient"
    assert created["dob"] == "1995-04-02"
    assert created["gender"] == "F"

    # The stored record follows the existing patients.json schema
    stored = json.loads((isolated_data / "patients.json").read_text(encoding="utf-8"))
    record = next(p for p in stored if p["id"] == "P007")
    assert record["name"] == "Test Patient"
    assert record["dob"] == "1995-04-02"
    assert record["gender"] == "F"
    assert record.get("phone") is None

    # …and the new patient's report is listed for them
    assert [r["id"] for r in client.get("/patients/P007/reports").json()] == [data["id"]]


def test_new_patient_rejects_missing_or_malformed_fields(isolated_data):
    for bad in (
        {"name": "", "dob": "1995-04-02", "gender": "F"},      # empty name
        {"name": "X", "dob": "02-04-1995", "gender": "F"},     # non-ISO dob
        {"name": "X", "dob": "1995-04-02", "gender": "   "},   # blank gender
    ):
        resp = client.post("/reports", json=_payload(patient_id=None, patient=bad))
        assert resp.status_code == 422, bad
    assert len(json.loads((isolated_data / "patients.json").read_text(encoding="utf-8"))) == 6


def test_new_patient_rejects_unsupported_fields(isolated_data):
    """JSON-mode input is validated against the actual patient schema: an
    unknown field (a caller-supplied ``id`` above all) is a clear 422, never
    silently dropped data, and nothing is written."""
    for bad in (
        {"name": "X", "dob": "1995-04-02", "gender": "F", "id": "P001"},
        {"name": "X", "dob": "1995-04-02", "gender": "F", "id": "P999"},
        {"name": "X", "dob": "1995-04-02", "gender": "F", "address": "21 Main St"},
        {"name": "X", "dob": "1995-04-02", "gender": "F", "notes": None},
    ):
        resp = client.post("/reports", json=_payload(patient_id=None, patient=bad))
        assert resp.status_code == 422, bad
        assert "patient" in str(resp.json())
    assert len(json.loads((isolated_data / "patients.json").read_text(encoding="utf-8"))) == 6


def test_identical_new_patient_is_not_duplicated(isolated_data):
    """A retried save (or the same person entered twice) reuses the patient
    already on disk instead of appending a second, identical record."""
    patient = {"name": "Retry Patient", "dob": "1993-07-07", "gender": "F",
               "phone": "+91-11111-22222", "email": "retry@example.com"}

    first = client.post("/reports", json=_payload(patient_id=None, patient=patient))
    assert first.status_code == 200
    created_id = first.json()["patient_id"]

    # A different draft for the same person → new report, same patient
    second = client.post("/reports", json=_payload(
        patient_id=None, patient=patient,
        draft_report="CLINICAL DICTATION DRAFT\n\nFINDINGS:\n1. Something else entirely.",
    ))
    assert second.status_code == 200
    assert second.json()["patient_id"] == created_id

    stored = json.loads((isolated_data / "patients.json").read_text(encoding="utf-8"))
    assert len(stored) == 7                      # only the one patient was added
    assert next(p for p in stored if p["id"] == created_id)["name"] == "Retry Patient"
    assert len(_report_file_records(isolated_data)) == 14   # both reports stored

    # …and both reports are listed for that single patient
    listed = client.get(f"/patients/{created_id}/reports").json()
    assert {r["id"] for r in listed} == {first.json()["id"], second.json()["id"]}


def test_different_patient_with_similar_details_still_created(isolated_data):
    """The duplicate guard is exact: one differing field (here the dob) means
    a genuinely different person, so a new record is still appended."""
    a = {"name": "Same Name", "dob": "1990-01-01", "gender": "F"}
    b = {"name": "Same Name", "dob": "1990-01-02", "gender": "F"}
    first = client.post("/reports", json=_payload(patient_id=None, patient=a))
    second = client.post("/reports", json=_payload(
        patient_id=None, patient=b,
        draft_report="CLINICAL DICTATION DRAFT\n\nFINDINGS:\n1. Something else entirely.",
    ))
    assert first.json()["patient_id"] != second.json()["patient_id"]
    stored = json.loads((isolated_data / "patients.json").read_text(encoding="utf-8"))
    assert len(stored) == 8


# ── Validation & error handling (no partial writes, clear details) ──────────

def test_save_requires_exactly_one_patient(isolated_data):
    both = _payload()
    both["patient"] = {"name": "X", "dob": "1990-01-01", "gender": "M"}
    assert client.post("/reports", json=both).status_code == 422
    assert client.post("/reports", json=_payload(patient_id=None)).status_code == 422
    assert len(_report_file_records(isolated_data)) == 12


def test_save_unknown_patient_returns_404(isolated_data):
    resp = client.post("/reports", json=_payload(patient_id="P999"))
    assert resp.status_code == 404
    assert "P999" in resp.json()["detail"]
    # Nothing was written for the unknown patient
    assert len(_report_file_records(isolated_data)) == 12
    assert len(json.loads((isolated_data / "patients.json").read_text(encoding="utf-8"))) == 6


def test_save_empty_draft_rejected(isolated_data):
    assert client.post("/reports", json=_payload(draft_report="   ")).status_code == 422
    assert len(_report_file_records(isolated_data)) == 12


def test_save_invalid_report_type_rejected(isolated_data):
    resp = client.post("/reports", json=_payload(report_type="MRI"))
    assert resp.status_code == 422
    assert "OPG" in str(resp.json())
    assert len(_report_file_records(isolated_data)) == 12


# ── Derivation from the dictated content only ───────────────────────────────

def test_impression_derived_from_findings_when_draft_has_no_section(isolated_data):
    resp = client.post("/reports", json=_payload(
        draft_report="CLINICAL DICTATION DRAFT\n\nFINDINGS:\n1. Mild bone loss at 46.",
    ))
    assert resp.status_code == 200
    assert resp.json()["impression"] == "Mild bone loss at 46."


# ── Anatomical site: dictated region kept on the record ─────────────────────

def test_anatomical_site_persisted_and_retrieved(isolated_data):
    """The dictated primary region is stored on the report (additive field)
    and comes back through the normal report endpoints."""
    resp = client.post("/reports", json=_payload(
        anatomical_site="right maxillary sinus",
        findings=[{"site": "Right maxillary sinus", "observation": "Mucosal thickening seen.",
                   "severity": "mild"}],
    ))
    assert resp.status_code == 200
    assert resp.json()["anatomical_site"] == "right maxillary sinus"

    stored = retrieval.get_report_by_id(resp.json()["id"])
    assert stored["anatomical_site"] == "right maxillary sinus"
    assert stored["findings"][0]["site"] == "Right maxillary sinus"

    detail = client.get(f"/reports/{resp.json()['id']}").json()
    assert detail["anatomical_site"] == "right maxillary sinus"


def test_records_without_anatomical_site_stay_valid(isolated_data):
    """Reports written before the field existed still load; the field is
    simply absent/None, never an error."""
    for report_id in ("R001", "R010"):
        detail = client.get(f"/reports/{report_id}")
        assert detail.status_code == 200, report_id
        assert detail.json().get("anatomical_site") is None
        assert "anatomical_site" not in retrieval.get_report_by_id(report_id)


def test_report_type_detected_from_transcript_when_omitted(isolated_data):
    resp = client.post("/reports", json=_payload(
        report_type=None,
        transcript="Bitewing radiographs show secondary caries at 15.",
    ))
    assert resp.status_code == 200
    assert resp.json()["type"] == "BITEWING"
