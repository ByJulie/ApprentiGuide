"""Tracing of the agent with Langfuse (Agentic AI, between S6 and S7).

Every run of the LangGraph agent (app/agent.py) becomes one Langfuse trace: each model call with its prompt,
answer, tokens and latency, and each tool call with its arguments and result. One conversation (thread) is one
Langfuse session, so a follow-up question appears under the same session.

- Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are in the environment (.env, never committed);
  LANGFUSE_BASE_URL is the server (default https://cloud.langfuse.com). Without them the agent runs unchanged.
- Langfuse is an external service: every string sent to it goes through the input guardrail's masking first
  (email, phone, social security number, IBAN; app/guardrail.py), and the user id is replaced by a hash.
"""
import hashlib, os

from app import guardrail

_CLIENT = None


def enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def mask(*, data, **_):
    """Langfuse mask function: personal data masked in every string, at any depth, before it is sent."""
    if isinstance(data, str):
        return guardrail.mask_pii(data)[0]
    if isinstance(data, dict):
        return {k: mask(data=v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask(data=v) for v in data]
    return data


def pseudonym(user: str) -> str:
    """The user id as Langfuse sees it: a short hash, so a trace cannot be read back to a student."""
    return "u-" + hashlib.sha256(user.encode("utf-8")).hexdigest()[:12]


def client():
    """The Langfuse client, created once with the mask (None when tracing is off)."""
    global _CLIENT
    if _CLIENT is None and enabled():
        try:   # company networks that inspect TLS: trust the operating system's certificates (never verify=False)
            import truststore
            truststore.inject_into_ssl()
        except ImportError:
            pass
        from langfuse import Langfuse
        _CLIENT = Langfuse(public_key=os.environ["LANGFUSE_PUBLIC_KEY"], secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                           base_url=os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"), mask=mask)
    return _CLIENT


def run_config(thread: str, user: str, tags: list[str] | None = None) -> dict:
    """What LangGraph needs to trace one run: the callback handler and the session, user and tags metadata.
    Empty when tracing is off."""
    if client() is None:
        return {}
    from langfuse.langchain import CallbackHandler
    return {"callbacks": [CallbackHandler(public_key=os.environ["LANGFUSE_PUBLIC_KEY"])],
            "metadata": {"langfuse_session_id": thread, "langfuse_user_id": pseudonym(user),
                         "langfuse_tags": ["apprentiguide-agent"] + (tags or [])}}


def flush() -> None:
    """Send the traces still in the buffer (call it before a script ends)."""
    if _CLIENT is not None:
        _CLIENT.flush()
