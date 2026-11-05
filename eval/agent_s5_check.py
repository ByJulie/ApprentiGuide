"""S5 check: the LangGraph agent (app/agent.py) on the five requests of S1, one thread each, with our own tools.

For each request: the tools called in order, whether the graph paused for approval (send_email), the answer, the
model calls and the time. The email of request 5 is refused by the check (nothing is sent; outbox in a temp folder).
Today is fixed (Wednesday 4 November 2026), like the S2 and S4 checks, and the memory store is a temp folder.

search_texts:
  --search real     the RAG retrieval of the Generative AI product (needs the PostgreSQL database, README section 3)
  --search corpus   no database: the 5 chunks of the real corpus (data/raw and the regulations in data/private) that
                    share the most words with the query. Same texts and ids, a much simpler ranking than the RAG.
python -m eval.agent_s5_check --search corpus [--model openai/gpt-oss-20b]   →   eval/runs/<date>_s5_langgraph_check_<model>.json"""
import argparse, json, os, re, sys, tempfile, time, unicodedata
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from langchain_core.messages import AIMessage, ToolMessage

from app import agent, memory, pipeline as P
from eval.agent_s2_check import REQUESTS, TODAY

STOP = set("avec dans pour quel quelle quels quelles est-ce mon mes une des les que qui est pas plus mais peux puis "
           "suis vais ai the and for what can how combien".split())
_CHUNKS = None


def words(text: str) -> set[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return {w[:6] for w in re.findall(r"[a-z0-9]{3,}", t) if w not in STOP}      # 6 letters: a crude stem


def corpus_search(query: str) -> str:
    """No database: the 5 best chunks of the real corpus by shared words, one per document, as SearchTextsResult."""
    global _CHUNKS
    if _CHUNKS is None:
        from ingest.load import build_chunks, regulation_chunks
        _CHUNKS = [(c, words(c.title + " " + c.text)) for c in build_chunks() + regulation_chunks()]
    q = words(query)
    ranked = sorted(_CHUNKS, key=lambda x: -len(q & x[1]))
    chosen, seen = [], set()
    for c, w in ranked:
        if c.citation_id not in seen and len(q & w) >= 2:
            chosen.append(c)
            seen.add(c.citation_id)
        if len(chosen) == 5:
            break
    return json.dumps({"not_found": not chosen, "missing_ids": [], "passages": [
        {"id": c.citation_id, "title": c.title, "source": c.source_label, "version_date": c.version_date,
         "url": c.url, "text": c.text} for c in chosen]}, ensure_ascii=False, default=str)


def trace(messages) -> list[dict]:
    out = []
    for m in messages:
        if isinstance(m, AIMessage):
            out += [{"tool": c["name"], "args": c["args"]} for c in m.tool_calls or []]
        elif isinstance(m, ToolMessage):
            r = m.content
            if m.name == "search_texts":   # the school regulations are private: keep the ids, not the texts
                r = "passage ids: " + ", ".join(re.findall(r'"id":\s*"([^"]+)"', r))
            out.append({"result_of": m.name, "result": r[:300]})
        elif getattr(m, "type", "") == "human" and str(m.content).startswith("[system check]"):
            out.append({"check_in_code": m.content})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", choices=["real", "corpus"], required=True)
    ap.add_argument("--model", help="answer model, default the one of config.yaml (gpt-oss-120b)")
    a = ap.parse_args()
    if a.model:
        P.use_model(a.model)
    memory.STORE = tempfile.mkdtemp(prefix="s5_memory_")
    outbox = os.path.join(tempfile.mkdtemp(prefix="s5_outbox_"), "outbox.json")
    graph = agent.build(search=corpus_search if a.search == "corpus" else None)
    rows, t0 = [], time.time()
    for i, q in enumerate(REQUESTS, 1):
        t = time.time()
        out = agent.ask(graph, q, thread=f"request-{i}", user="ines", today=TODAY, outbox=outbox)
        row = {"request": q, "paused_for_approval": out["paused"]}
        if out["paused"]:
            row["approval_asked"] = {k: v for k, v in out["approval"].items() if k != "body"} | {"body": out["approval"]["body"][:400]}
            out = agent.decide(graph, False, thread=f"request-{i}", user="ines", today=TODAY, outbox=outbox)
            row["decision"] = "refused by the check (nothing sent)"
        row.update(answer=out["answer"], steps=trace(out["messages"]),
                   model_calls=sum(isinstance(m, AIMessage) for m in out["messages"]), seconds=round(time.time() - t, 1))
        rows.append(row)
        print(f"{i}. {[s.get('tool') for s in row['steps'] if 'tool' in s]} paused={row['paused_for_approval']} "
              f"{row['seconds']} s\n   {row['answer'][:160]!r}", flush=True)
    res = {"date": str(date.today()), "today_fixed": str(TODAY), "model": P.CFG["model"]["name"], "search": a.search,
           "framework": "LangGraph (app/agent.py)", "requests": rows,
           "memory_after": [f.model_dump(mode="json") for f in memory.load("ines", TODAY)],
           "outbox_after": json.load(open(outbox, encoding="utf-8")) if os.path.exists(outbox) else [], "seconds": round(time.time() - t0)}
    path = os.path.join(os.path.dirname(__file__), "runs",
                        f"{date.today()}_s5_langgraph_check_{a.search}_{P.CFG['model']['name'].split('/')[-1]}.json")
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("->", path)


if __name__ == "__main__":
    main()
