"""Several API keys for one endpoint: when a key reaches its daily limit, the next one takes over.

The Groq free plan allows 200 000 tokens per day per model and per key; a full evaluation run needs more.
Keys are read from the environment: GROQ_API_KEY, then GROQ_API_KEY_2, GROQ_API_KEY_3... (empty ones
skipped). Same model on every key, so the answers of one run stay comparable. Per-minute limits are
retried by the OpenAI client itself; only daily limits switch keys.

    client = rotating_client("https://api.groq.com/openai/v1", "GROQ_API_KEY")
    client.chat.completions.create(model=..., messages=...)   # same call as an OpenAI client
"""
import os
from types import SimpleNamespace

from openai import OpenAI, RateLimitError


def keys_from_env(env: str) -> list[str]:
    names = [env] + [f"{env}_{i}" for i in range(2, 10)]
    return [os.environ[n] for n in names if os.environ.get(n)]


def is_daily_limit(error: Exception) -> bool:
    text = str(error).lower()
    return "per day" in text or "(tpd)" in text or "(rpd)" in text


class RotatingClient:
    """Duck-types the part of OpenAI we use: client.chat.completions.create(**kwargs)."""

    def __init__(self, base_url: str, keys: list[str], max_retries: int = 6):
        if not keys:
            raise ValueError("no API key")
        self._clients = [OpenAI(base_url=base_url, api_key=k, max_retries=max_retries) for k in keys]
        self.index = 0  # which key is in use (never the key itself in logs)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        while True:
            try:
                return self._clients[self.index].chat.completions.create(**kwargs)
            except RateLimitError as error:
                if is_daily_limit(error) and self.index + 1 < len(self._clients):
                    self.index += 1
                    print(f"[keys] daily limit reached, switching to key {self.index + 1} of {len(self._clients)}")
                    continue
                raise


def rotating_client(base_url: str, env: str, max_retries: int = 6):
    keys = keys_from_env(env)
    if len(keys) <= 1:
        return OpenAI(base_url=base_url, api_key=keys[0] if keys else "missing", max_retries=max_retries)
    return RotatingClient(base_url, keys, max_retries)
