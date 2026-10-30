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


CITATION = re.compile(r"\[([^\[\]]+)\]")


def citations(text: str, sources: list[dict]) -> dict:
    """Sources shown under the answer, built from the database, never from the model's text.

    cited: sources whose id the answer cites, in order of first citation, with link, label, date and
    passages (a fiche can contribute several sections). consulted: retrieved but not cited.
    invented: ids cited by the model that are not among the retrieved sources (no link is made).
    """
    by_id: dict[str, dict] = {}
    for s in sources:
        entry = by_id.setdefault(s["id"].upper(), {"id": s["id"], "source_label": s["source_label"],
                                                    "url": s["url"], "version_date": s["version_date"],
                                                    "passages": []})
        entry["passages"].append({"title": s.get("title"), "text": s["text"]})
    cited_ids = []
    for group in CITATION.findall(text):
        for ref in re.split(r"[,;]\s*", group):
            ref = ref.strip().upper()
            if ref and ref not in cited_ids:
                cited_ids.append(ref)
    return {"cited": [by_id[i] for i in cited_ids if i in by_id],
            "consulted": [e for i, e in by_id.items() if i not in cited_ids],
            "invented": [i for i in cited_ids if i not in by_id]}


ENGLISH = re.compile(r"\b(the|is|are|can|my|what|how|do|does|i am|should|which|who)\b", re.I)
NOT_FOUND_MESSAGE = {
    "fr": ("Je n'ai pas trouvé la réponse dans les textes que je consulte (Code du travail, CESEDA, "
           "Code de la sécurité sociale et fiches Service-Public.fr). Essaie de reformuler ta question, ou "
           "pose-la à ton école ou à tes RH : ce sont eux qui connaissent les règles propres à ton contrat."),
    "en": ("I could not find the answer in the texts I search (French labour code, immigration code, "
           "social security code and Service-Public.fr fact sheets). Try asking differently, or ask your "
           "school or your HR team: they know the rules specific to your contract."),
}
MISSING_MESSAGE = {"fr": " L'article {ids} ne fait pas partie de ces textes.",
                   "en": " Article {ids} is not among these texts."}


def language(question: str, question_fr: str | None) -> str:
    """fr when the French rewrite left the question unchanged, else en (heuristic without a rewrite)."""
    if question_fr is not None:
        same = re.sub(r"\W+", "", question_fr.lower()) == re.sub(r"\W+", "", question.lower())
        return "fr" if same else "en"
    return "en" if len(ENGLISH.findall(question)) >= 2 else "fr"


def not_found_message(lang: str, missing_ids: list[str] = ()) -> str:
    """Fixed text written in code, in the language of the question: the user sees where we looked and
    what to do next; NOT FOUND stays an internal signal (field not_found)."""
    text = NOT_FOUND_MESSAGE[lang]
    if missing_ids:
        text += MISSING_MESSAGE[lang].format(ids=", ".join(missing_ids))
    return text


def answer(question: str) -> dict:
    t0 = time.time()
    found = retrieve(question)
    lang = language(question, found.question_fr if found else None)
    if found is not None and found.abstain:
        # Decided in code from the reranker score: the model never sees weak sources.
        reason = "missing_article" if found.missing_ids else "weak_sources"
        return {"answer": not_found_message(lang, found.missing_ids), "not_found": True,
                "not_found_reason": reason, "sources": [], "seconds": round(time.time() - t0, 2),
                "best_score": found.best_score, "citations": citations("", [])}
    sources = found.sources if found else []
    text = generate(question, sources, found.missing_ids if found else [])
    if text.strip().upper().startswith(NOT_FOUND):
        # The model judged the sources insufficient (rule 6 of the prompt): same fixed message.
        return {"answer": not_found_message(lang, found.missing_ids if found else []), "not_found": True,
                "not_found_reason": "model", "sources": [s["id"] for s in sources],
                "seconds": round(time.time() - t0, 2), "best_score": found.best_score if found else None,
                "citations": citations("", sources)}
    return {"answer": text, "not_found": False, "sources": [s["id"] for s in sources],
            "seconds": round(time.time() - t0, 2), "best_score": found.best_score if found else None,
            "citations": citations(text, sources)}
