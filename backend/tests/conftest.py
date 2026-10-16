"""Shared pytest configuration for the Viscan AI backend test suite.

Responsibilities, in this order:

1. **Capture the real credentials before anything blanks them.**
   ``tests/test_api.py`` clears ``GEMINI_API_KEY`` / ``OPENAI_API_KEY`` at
   import time so the suite can never spend a paid request by accident.
   ``conftest.py`` is imported before any test module, so the real values are
   read here (process environment first, then ``backend/.env``) and kept for
   ``tests/test_live_api.py`` - the only tests in the suite allowed to call a
   provider.
2. **Force demo mode for the whole session**, whatever order pytest imports
   the modules in, so import order can never leak a live provider into the
   offline tests.
3. Register the ``live`` marker and provide the ``live_llm`` fixture.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from dotenv import dotenv_values

_FILE_ENV = dotenv_values(BACKEND_DIR / ".env")


def _setting(name: str, default: str = "") -> str:
    """Value of `name` from the process environment, falling back to `.env`."""
    return ((os.environ.get(name) or "") or (_FILE_ENV.get(name) or "")).strip() or default


LIVE_GEMINI_API_KEY = _setting("GEMINI_API_KEY")
LIVE_OPENAI_API_KEY = _setting("OPENAI_API_KEY")
LIVE_GEMINI_MODEL = _setting("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Blank both providers for the entire session: every test that does not ask
# for the `live_llm` fixture is guaranteed to be free, no matter which module
# pytest imports first.
os.environ["GEMINI_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: makes a real LLM API call (spends provider quota; skipped when no key)",
    )


@pytest.fixture
def live_llm(monkeypatch):
    """Put the app into genuine live mode for a single test.

    Re-installs the credential captured at collection time, forces the service
    *and* API layers out of demo mode, and counts provider requests with a spy
    that wraps the original client - so any request the test triggers is a
    real HTTP call to the provider, never a stub.

    Yields a state dict; ``state["calls"]`` is the number of provider requests
    issued. Skipped when no credential is configured, or when
    ``VISCAN_LIVE_TESTS=0`` is set to opt out entirely (e.g. in CI).
    """
    if _setting("VISCAN_LIVE_TESTS", "1").lower() in {"0", "false", "no", "off"}:
        pytest.skip("live LLM tests disabled (VISCAN_LIVE_TESTS=0)")
    if not (LIVE_GEMINI_API_KEY or LIVE_OPENAI_API_KEY):
        pytest.skip("no LLM credential in backend/.env; live LLM tests skipped")

    import main as main_module
    from services import llm_service

    monkeypatch.setattr(llm_service, "GEMINI_API_KEY", LIVE_GEMINI_API_KEY)
    monkeypatch.setattr(llm_service, "OPENAI_API_KEY", LIVE_OPENAI_API_KEY)
    monkeypatch.setattr(llm_service, "GEMINI_MODEL", LIVE_GEMINI_MODEL)
    monkeypatch.setattr(
        llm_service, "LLM_PROVIDER", "gemini" if LIVE_GEMINI_API_KEY else "openai"
    )
    monkeypatch.setattr(llm_service, "DEMO_MODE", False)
    monkeypatch.setattr(main_module, "DEMO_MODE", False)

    state = {"calls": 0}
    real_call_llm = llm_service._call_llm

    def _counting_call(*args, **kwargs):
        state["calls"] += 1
        return real_call_llm(*args, **kwargs)

    monkeypatch.setattr(llm_service, "_call_llm", _counting_call)
    yield state
