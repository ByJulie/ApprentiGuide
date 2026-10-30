"""The whole product logic lives here, as small functions: easy to test, easy to turn into agent tools later."""
import os, re, sys, time, yaml
from dotenv import load_dotenv
from openai import OpenAI

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
load_dotenv(os.path.join(ROOT, ".env"))
key_env = CFG["model"].get("api_key_env", "none")
# Ollama ignores the key; cloud endpoints need the one named in config.yaml (checked on first call)
client = OpenAI(base_url=CFG["model"]["base_url"],
                api_key="ollama" if key_env == "none" else os.environ.get(key_env) or "missing",
                max_retries=6)  # the free plan answers 429 when calls come in a burst (evaluation runs)
def _client(cfg: dict) -> OpenAI:
    env = cfg.get("api_key_env", "none")
    return OpenAI(base_url=cfg["base_url"], max_retries=6,
                  api_key="ollama" if env == "none" else os.environ.get(env) or "missing")


# The rewrite can use its own (local) model: section rewrite of config.yaml, else the answer model.
REWRITE_CFG = CFG.get("rewrite") or CFG["model"]
rewrite_client = _client(REWRITE_CFG)
SYSTEM = open(os.path.join(ROOT, CFG["prompt_file"])).read()
NOT_FOUND = "NOT FOUND"
_retriever = None


def retriever():
    global _retriever
    if _retriever is None:
        from rag.retrieve import Retriever
        _retriever = Retriever(CFG["retrieval"])
    return _retriever


REWRITE = ("Translate the user's question into French. If it is already in French, return it unchanged. "
           "Keep everything it says: the person's situation (age, nationality, year, contract) and what "
           "they ask. Do not answer, do not add websites, sources or article numbers. One line, no quotes.")


REFUSAL = re.compile(r"\b(sorry|can['’]?t help|cannot help|unable to|désolée?|je ne peux pas)\b", re.I)


def rewrite_fr(question: str) -> str | None:
    """French version of the question, or None if the call fails (retrieval then uses the original)."""
    try:
        r = rewrite_client.chat.completions.create(
            model=REWRITE_CFG["name"], temperature=0, max_tokens=400,
            **({"reasoning_effort": "low"} if "gpt-oss" in REWRITE_CFG["name"] else {}),
            messages=[{"role": "system", "content": REWRITE}, {"role": "user", "content": question}])
        text = (r.choices[0].message.content or "").strip()
    except Exception:
        return None
    text = text.strip().strip('"«»“”').strip()
    # A refusal is not a query: searching with it would rank noise. Fall back to the original question.
    if REFUSAL.search(text):
        return None
    # A number the user did not write would send retrieval to the wrong article: removed in code.
    from rag.retrieve import strip_unasked_ids
    return strip_unasked_ids(question, text) or None


def retrieve(question: str):
    """Retrieval result (sources, abstain decision), or None when retrieval is disabled (baseline)."""
    if not CFG["retrieval"]["enabled"]:
        return None
    question_fr = rewrite_fr(question) if CFG["retrieval"].get("rewrite_fr") else None
    return retriever()(question, question_fr)


def generate(question: str, sources: list[dict], missing_ids: list[str] = ()) -> str:
    ctx = "\n\n".join(f"[{s['id']}] {s['text']}" for s in sources) or "(no sources)"
    note = ""
    if missing_ids:
        note = (f"\n\nNote: {', '.join(missing_ids)} is not in the sources. Say so; do not answer "
                "from another article as if it were this one.")
    r = client.chat.completions.create(
        model=CFG["model"]["name"], temperature=CFG["model"]["temperature"],
        max_tokens=CFG["model"]["max_tokens"],
        **({"reasoning_effort": CFG["model"]["reasoning_effort"]} if CFG["model"].get("reasoning_effort") else {}),
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"<sources>\n{ctx}\n</sources>{note}\n\nQuestion: {question}"}])
    return (r.choices[0].message.content or "").strip()


def answer(question: str) -> dict:
    t0 = time.time()
    found = retrieve(question)
    if found is not None and found.abstain:
        # Decided in code from the reranker score: the model never sees weak sources.
        text = NOT_FOUND
        if found.missing_ids:
            text += f" ({', '.join(found.missing_ids)} is not in the texts I can search)"
        return {"answer": text, "sources": [], "seconds": round(time.time() - t0, 2),
                "best_score": found.best_score}
    sources = found.sources if found else []
    text = generate(question, sources, found.missing_ids if found else [])
    return {"answer": text, "sources": [s["id"] for s in sources], "seconds": round(time.time() - t0, 2),
            "best_score": found.best_score if found else None}
