"""Live LLM tests - the ONLY tests in this suite that make a real API call.

Every other test runs with the provider keys blanked and the client stubbed,
so the suite stays free. These two deliberately go the other way: the
``live_llm`` fixture (tests/conftest.py) re-installs the Gemini credential
captured from ``backend/.env``, forces the service and API layers out of demo
mode, and wraps the *original* provider client with a request counter.

Each test therefore proves three things about a genuine provider round trip:

  1. a request was actually issued (``live_llm["calls"]``),
  2. it succeeded - ``fallback_reason is None`` and ``demo_mode is False``, so
     the deterministic fallback never masked a failed or skipped call, and
  3. the guardrails still hold on real model output (right patient, review
     gate forced, no invented findings or severity grades).

Cost: two short requests to the Gemini free tier. Skip them with
``-m "not live"``, or set ``VISCAN_LIVE_TESTS=0``, or leave ``.env`` empty -
in all three cases they report as skipped, never failed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Backend import root (the real keys were already captured by conftest.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from main import app
from services import llm_service

client = TestClient(app)

pytestmark = pytest.mark.live

_DICTATION_TRANSCRIPT = (
    "Lower right third molar is horizontally impacted. "
    "Mild pericoronitis around tooth 48."
)
_SEVERITIES = {"none", "mild", "moderate", "severe", "unspecified"}


def test_live_query_retrieval_is_served_by_the_model(live_llm):
    """POST /query is answered by Gemini, not by the deterministic demo parser."""
    resp = client.post("/query", json={"text": "Open Priya's latest OPG report"})
    assert resp.status_code == 200
    data = resp.json()

    # A real provider request happened and produced this answer.
    assert live_llm["calls"] >= 1, "no provider request was issued"
    assert data["demo_mode"] is False
    assert data["fallback_reason"] is None, (
        "the live call failed and the deterministic fallback answered instead: "
        f"{data['fallback_reason']}"
    )

    # The interpretation is still correct end-to-end, guardrails included.
    assert data["intent"] == "retrieve_report"
    assert data["patient_name"] == "Priya"
    assert data["report_type"] == "OPG"
    assert data["ambiguous"] is False
    assert data["report"] is not None
    assert data["report"]["patient_id"] == "P001"
    assert data["report"]["type"] == "OPG"
    assert [p["id"] for p in data["matched_patients"]] == ["P001"]


def test_live_dictation_is_structured_and_stays_grounded(live_llm):
    """POST /dictation structures real model output without inventing findings."""
    resp = client.post("/dictation", json={"transcript": _DICTATION_TRANSCRIPT})
    assert resp.status_code == 200
    data = resp.json()

    assert live_llm["calls"] >= 1, "no provider request was issued"
    assert data["demo_mode"] is False
    assert data["fallback_reason"] is None, (
        "the live call failed and a deterministic result was produced instead: "
        f"{data['fallback_reason']}"
    )
    assert data["usable"] is True
    assert data["requires_review"] is True          # review gate never bypassed
    assert data["findings"], "no findings returned for a clinical transcript"
    assert data["draft_report"].startswith("CLINICAL DICTATION DRAFT")
    assert "REVIEW STATUS:" in data["draft_report"]

    # No invented observation: every returned finding must satisfy the same
    # deterministic token-overlap grounding check the pipeline applies.
    transcript_tokens = llm_service._tokenise(_DICTATION_TRANSCRIPT)
    transcript_norm = " ".join(_DICTATION_TRANSCRIPT.casefold().split())
    for finding in data["findings"]:
        observation = finding.get("observation", "")
        assert llm_service._is_grounded(
            observation, transcript_tokens, transcript_norm
        ), f"ungrounded finding returned by the live model: {observation!r}"
        # Only a grade the dictation literally states may survive.
        assert finding.get("severity") in _SEVERITIES, (
            f"ungrounded severity for {observation!r}: {finding.get('severity')!r}"
        )

    # ...and the dictated content itself must actually appear in the findings.
    dictated = set(llm_service._significant_tokens(_DICTATION_TRANSCRIPT))
    produced = set(
        llm_service._significant_tokens(
            " ".join(
                f"{f.get('site', '')} {f.get('observation', '')}" for f in data["findings"]
            )
        )
    )
    assert len(dictated & produced) / len(dictated) >= 0.5, (
        f"findings dropped too much of the transcript: {sorted(dictated - produced)}"
    )
