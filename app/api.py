"""OpenAI-compatible API in front of the pipeline, so any chat interface (Open WebUI) can use ApprentiGuide.

Run from the repository root:
    uvicorn api:app --app-dir app --host 0.0.0.0 --port 8001
then add http://localhost:8001/v1 (or http://host.docker.internal:8001/v1 from Docker) as an OpenAI
connection in Open WebUI: the model "apprentiguide" appears in the list.

Each request runs the same pipeline as eval.py and ui.py. Follow-up questions are made complete from the
conversation first. Nothing is stored: the conversation lives in the chat interface only.
"""
import json
import re
import time
import uuid

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

import pipeline

MODEL_ID = "apprentiguide"
app = FastAPI(title="ApprentiGuide")


def text_of(content) -> str:
    """Message content as text (Open WebUI may send a list of parts)."""
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def interface_task(question: str) -> str | None:
    """Open WebUI also asks the model for a chat title, tags or follow-up suggestions ("### Task:").
    Those must not go through retrieval and the answer model: answered here, at no cost."""
    if "### Task:" not in question:
        return None
    lower = question.lower()
    if "title" in lower:
        return json.dumps({"title": "ApprentiGuide"})
    if "follow" in lower:
        return json.dumps({"follow_ups": []})
    if "tags" in lower:
        return json.dumps({"tags": [{"name": "apprentissage"}]})
    return ""


MODEL_SOURCES_LINE = re.compile(r"\n\s*(\*\*)?sources?\s*(\*\*)?\s*:.*\Z", re.I | re.S)
# The model's own sources line: "Sources :", "**Sources**" or "Sources" alone on its line.
SOURCES_START = re.compile(r"\n\s*(\*\*)?sources?\s*(\*\*)?\s*(:|\n|$)", re.I)


def held_back(text: str) -> str:
    """Text safe to show while the model writes: a last line that may still become "Sources" waits."""
    last = text[text.rfind("\n") + 1:]
    word = last.strip().strip("*").lower()
    if "\n" in text and word and "sources".startswith(word[:7]):
        return text[:text.rfind("\n")]
    return text


def footer() -> str:
    return (f"\n\n---\n*Textes : Légifrance et Service-Public.fr/DILA (Licence Ouverte), fichiers du "
            f"{pipeline.CFG.get('corpus_snapshot', '')}. Vérifie toujours avec ton école ou tes RH.*")


def render(out: dict) -> str:
    """Plain Markdown answer for clients that do not stream: the answer, then its sources from the
    database. The model's own "Sources:" line is dropped: the block built from the database replaces it."""
    parts = [MODEL_SOURCES_LINE.sub("", out["answer"]).rstrip()]
    refs = out.get("citations", {})
    if refs.get("cited"):
        parts.append("\n**Sources**")
        for s in refs["cited"]:
            date = f", version du {s['version_date']}" if s.get("version_date") else ""
            parts.append(f"- [{s['id']}]({s['url']}) · {s['source_label']}{date}")
    if refs.get("invented"):
        parts.append(f"\n⚠️ Citation non vérifiée (absente des textes trouvés) : {', '.join(refs['invented'])}")
    return "\n".join(parts) + footer()


def source_card(s: dict) -> dict:
    """A source as Open WebUI shows it: a card under the answer, the passage in a panel, the link."""
    date = f", version du {s['version_date']}" if s.get("version_date") else ""
    return {"source": {"id": s["url"], "name": s["id"], "url": s["url"]},
            "document": [p["text"] for p in s["passages"]],
            "metadata": [{"source": s["url"], "name": f"{s['id']} · {s['source_label']}{date}"} for _ in s["passages"]]}


def status(text: str, done: bool = False) -> dict:
    return {"event": {"type": "status", "data": {"description": text, "done": done}}}


def answer_events(question: str, history: list[dict]):
    """The pipeline step by step for a chat interface: status lines while searching, the model's reasoning
    and answer as they are written, then the sources as cards. Same decisions as pipeline.answer()."""
    if history:
        yield status("Je relis la conversation…")
        question = pipeline.standalone(question, history)
    yield status("Je cherche dans le Code du travail, le CESEDA et les fiches Service-Public…")
    found = pipeline.retrieve(question)
    lang = pipeline.language(question, found.question_fr if found else None)
    if found is not None and found.abstain:
        yield {"content": pipeline.not_found_message(lang, found.missing_ids) + footer()}
        yield status("Aucun texte ne répond à cette question", done=True)
        return
    sources = found.sources if found else []
    yield status(f"{len({s['id'] for s in sources})} textes trouvés, je rédige la réponse…")
    text, shown, not_found = "", 0, False
    for kind, piece in pipeline.generate_stream(question, sources, found.missing_ids if found else [], lang):
        if kind == "reasoning":
            yield {"reasoning": piece}
            continue
        text += piece.replace("【", "[").replace("】", "]")  # the model sometimes writes 【F2918】
        start = text.lstrip()
        if len(start) < 12 and pipeline.NOT_FOUND.startswith(start.upper()[:len(pipeline.NOT_FOUND)]):
            continue  # might be the beginning of NOT FOUND: hold it back
        if start.upper().startswith(pipeline.NOT_FOUND):
            not_found = True
            break
        end = SOURCES_START.search(text)
        visible = text[:end.start()] if end else held_back(text)
        visible = visible.rstrip("\n") + ("\n" if visible.endswith("\n") else "")  # no runs of blank lines
        if len(visible) > shown:
            yield {"content": visible[shown:]}
            shown = len(visible)
    if not_found or text.strip().upper().startswith(pipeline.NOT_FOUND):  # also when the stream ends on it
        yield {"content": pipeline.not_found_message(lang, found.missing_ids if found else []) + footer()}
        yield status("Les textes trouvés ne répondent pas à cette question", done=True)
        return
    end = SOURCES_START.search(text)
    visible = (text[:end.start()] if end else text).rstrip()
    if len(visible) > shown:
        yield {"content": visible[shown:]}
    refs = pipeline.citations(text, sources)
    for s in refs["cited"]:
        yield {"event": {"type": "source", "data": source_card(s)}}
    if refs["invented"]:
        yield {"content": f"\n\n⚠️ Citation non vérifiée (absente des textes trouvés) : {', '.join(refs['invented'])}"}
    yield {"content": footer()}
    yield status(f"Réponse fondée sur {len(refs['cited'])} texte(s)", done=True)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "apprentiguide"}]}


@app.post("/v1/chat/completions")
def chat(body: dict, authorization: str | None = Header(default=None)):
    messages = [{"role": m.get("role"), "content": text_of(m.get("content"))} for m in body.get("messages", [])]
    users = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if not users:
        raise HTTPException(400, "no user message")
    question = messages[users[-1]]["content"]
    content = interface_task(question)
    history = [m for m in messages[:users[-1]] if m["role"] in ("user", "assistant")]
    # Previous answers carry sources and the licence line: only the answer text is useful as context.
    history = [{**m, "content": re.split(r"\n\*\*Sources\*\*|\n\n\*Textes :", m["content"])[0]} for m in history]
    reply_id, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
    if not body.get("stream"):
        if content is None:
            content = render(pipeline.answer(pipeline.standalone(question, history)))
        return {"id": reply_id, "object": "chat.completion", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}]}
    base = {"id": reply_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID}

    def sse(delta: dict | None = None, event: dict | None = None, finish: str | None = None) -> str:
        data = {**base, "choices": [] if event else [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
        if event:
            data["event"] = event
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def events():
        if content is not None:  # interface task
            yield sse({"role": "assistant", "content": content})
        else:
            for item in answer_events(question, history):
                if "event" in item:
                    yield sse(event=item["event"])
                elif "reasoning" in item:
                    yield sse({"reasoning_content": item["reasoning"]})
                else:
                    yield sse({"content": item["content"]})
        yield sse(finish="stop")
        yield "data: [DONE]\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")
