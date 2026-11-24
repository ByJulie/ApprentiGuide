"""Shared test setup: tests never send traces to Langfuse, even when the keys are in .env."""
import pytest


@pytest.fixture(autouse=True)
def no_langfuse(monkeypatch):
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    from app import tracing
    monkeypatch.setattr(tracing, "_CLIENT", None)
