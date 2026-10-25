"""The whole product logic lives here, as small functions: easy to test, easy to turn into agent tools later."""
import os, time, yaml
from dotenv import load_dotenv
from openai import OpenAI

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")))
ROOT = os.path.join(os.path.dirname(__file__), "..")
load_dotenv(os.path.join(ROOT, ".env"))
key_env = CFG["model"].get("api_key_env", "none")
# Ollama ignores the key; cloud endpoints need the one named in config.yaml
client = OpenAI(base_url=CFG["model"]["base_url"],
                api_key="ollama" if key_env == "none" else os.environ[key_env])
SYSTEM = open(os.path.join(ROOT, CFG["prompt_file"])).read()


def retrieve(question: str) -> list[dict]:
    """Return a list of {"id": ..., "text": ...}. Replace with your retriever (see the 7 Oct RAG lab)."""
    if not CFG["retrieval"]["enabled"]:
        return []
    raise NotImplementedError("plug your retriever here")


def generate(question: str, sources: list[dict]) -> str:
    ctx = "\n\n".join(f"[{s['id']}] {s['text']}" for s in sources) or "(no sources)"
    r = client.chat.completions.create(
        model=CFG["model"]["name"], temperature=CFG["model"]["temperature"],
        max_tokens=CFG["model"]["max_tokens"],
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"<sources>\n{ctx}\n</sources>\n\nQuestion: {question}"}])
    return r.choices[0].message.content.strip()


def answer(question: str) -> dict:
    t0 = time.time()
    sources = retrieve(question)
    text = generate(question, sources)
    return {"answer": text, "sources": [s["id"] for s in sources], "seconds": round(time.time() - t0, 2)}
