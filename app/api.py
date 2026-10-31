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


def render(out: dict) -> str:
    """The answer, then its sources from the database (links, label, version date, passage), as Markdown.
    The model's own "Sources:" line is dropped: the block built from the database replaces it."""
    parts = [MODEL_SOURCES_LINE.sub("", out["answer"]).rstrip()]
    refs = out.get("citations", {})
    if refs.get("cited"):
        parts.append("\n**Sources**")
        for s in refs["cited"]:
            date = f", version du {s['version_date']}" if s.get("version_date") else ""
            passage = "\n\n".join(p["text"] for p in s["passages"]).replace("\n", " ")
            parts.append(f"- [{s['id']}]({s['url']}) · {s['source_label']}{date}\n"
                         f"  <details><summary>Voir le passage</summary>\n\n  {passage}\n\n  </details>")
    if refs.get("invented"):
        parts.append(f"\n⚠️ Citation non vérifiée (absente des textes trouvés) : {', '.join(refs['invented'])}")
    if refs.get("consulted"):
        links = ", ".join(f"[{s['id']}]({s['url']})" for s in refs["consulted"])
        parts.append(f"\n<sub>Autres textes consultés : {links}</sub>")
    parts.append(f"\n<sub>Textes : Légifrance et Service-Public.fr/DILA (Licence Ouverte), fichiers du "
                 f"{pipeline.CFG.get('corpus_snapshot', '')}. Vérifie toujours avec ton école ou tes RH.</sub>")
    return "\n".join(parts)


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
    if content is None:
        history = [m for m in messages[:users[-1]] if m["role"] in ("user", "assistant")]
        # Previous answers carry their sources block: only the answer text is useful as context.
        history = [{**m, "content": m["content"].split("\n**Sources**")[0]} for m in history]
        content = render(pipeline.answer(pipeline.standalone(question, history)))
    reply_id, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
    if not body.get("stream"):
        return {"id": reply_id, "object": "chat.completion", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}]}

    def events():
        chunk = {"id": reply_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID}
        yield f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': content}, 'finish_reason': None}]})}\n\n"
        yield f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")
