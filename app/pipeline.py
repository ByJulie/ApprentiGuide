"""The whole product logic lives here, as small functions: easy to test, easy to turn into agent tools later."""
import os, sys, time, yaml
from dotenv import load_dotenv
from openai import OpenAI

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
load_dotenv(os.path.join(ROOT, ".env"))
key_env = CFG["model"].get("api_key_env", "none")
# Ollama ignores the key; cloud endpoints need the one named in config.yaml (checked on first call)
client = OpenAI(base_url=CFG["model"]["base_url"],
                api_key="ollama" if key_env == "none" else os.environ.get(key_env) or "missing")
SYSTEM = open(os.path.join(ROOT, CFG["prompt_file"])).read()
NOT_FOUND = "NOT FOUND"
_retriever = None


def retriever():
    global _retriever
    if _retriever is None:
        from rag.retrieve import Retriever
        _retriever = Retriever(CFG["retrieval"])
    return _retriever


REWRITE = ("Rewrite the user's question in French, as a short search query for French employment "
           "law and service-public.fr fiches. Keep article numbers and facts about the person "
           "(age, nationality, year of study). Return the query only.")


def rewrite_fr(question: str) -> str | None:
    """French version of the question, or None if the call fails (retrieval then uses the original)."""
    try:
        r = client.chat.completions.create(
            model=CFG["model"]["name"], temperature=0, max_tokens=400,
            **({"reasoning_effort": "low"} if CFG["model"].get("reasoning_effort") else {}),
            messages=[{"role": "system", "content": REWRITE}, {"role": "user", "content": question}])
        return (r.choices[0].message.content or "").strip() or None
    except Exception:
        return None


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
        return {"answer": NOT_FOUND, "sources": [], "seconds": round(time.time() - t0, 2),
                "best_score": found.best_score}
    sources = found.sources if found else []
    text = generate(question, sources, found.missing_ids if found else [])
    return {"answer": text, "sources": [s["id"] for s in sources], "seconds": round(time.time() - t0, 2),
            "best_score": found.best_score if found else None}
