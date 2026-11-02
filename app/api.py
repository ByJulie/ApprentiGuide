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

import guardrail
import pipeline
from pipeline import source_link

MODEL_ID = "apprentiguide"
app = FastAPI(title="ApprentiGuide")


def text_of(content) -> str:
    """Message content as text (Open WebUI may send a list of parts)."""
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def chat_title(task: str, limit: int = 50) -> str:
    """Title of a conversation in the sidebar: the start of the user's first question, cut on a word.
    No model call: titles must not use the answer quota."""
    match = re.search(r"USER:\s*(.+)", task)
    first = re.sub(r"\s+", " ", match.group(1)).strip().rstrip("?").strip() if match else ""
    if not first:
        return "ApprentiGuide"
    if len(first) > limit:
        first = first[:limit].rsplit(" ", 1)[0] + "…"
    return first[0].upper() + first[1:]


def interface_task(question: str) -> str | None:
    """Open WebUI also asks the model for a chat title, tags or follow-up suggestions ("### Task:").
    Those must not go through retrieval and the answer model: answered here, at no cost."""
    if "### Task:" not in question:
        return None
    lower = question.lower()
    if "title" in lower:
        return json.dumps({"title": chat_title(question)}, ensure_ascii=False)
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
    return (f"\n\n---\n*Textes : Légifrance et Service-Public.fr/DILA (Licence Ouverte), règlement de l'école, fichiers du "
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
            title = "" if s.get("url") else f"\n  {s['passages'][0]['title']}"
            parts.append(f"- {source_link(s)} · {s['source_label']}{date}{title}")
    if refs.get("invented"):
        parts.append(f"\n⚠️ Citation non vérifiée (absente des textes trouvés) : {', '.join(refs['invented'])}")
    return "\n".join(parts) + footer()


SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


def clean_passage(text: str, citation_id: str) -> tuple[str, str]:
    """(section title, text) of a stored passage: the "Article L6222-35" heading and the entry-into-force
    note ("Nota : ...") removed, spaces normalised."""
    title, _, body = text.partition("\n")
    if not body:
        title, body = "", title
    body = re.split(r"\bNota\s*:", body)[0]
    body = re.sub(r"\s+", " ", body).strip()
    title = "" if title.strip().lower() == f"article {citation_id}".lower() else title.strip()
    return title, body


def excerpt(source: dict, answer: str, keep: int = 2) -> str:
    """What the user reads when opening a source: where it comes from, the sentences closest to the
    sentences of the answer that cite it (bge-m3, local), then the full cleaned text for context."""
    from rag import models
    title, body = clean_passage(" ".join(p["text"] for p in source["passages"]), source["id"])
    sentences = [s for s in SENTENCE_END.split(body) if len(s) > 20]
    claims = [s for s in SENTENCE_END.split(answer) if source["id"].upper() in s.upper()] or [answer]
    if len(sentences) > keep:
        vectors = models.embed([" ".join(claims)] + sentences)
        scores = [sum(a * b for a, b in zip(vectors[0], v)) for v in vectors[1:]]
        best = sorted(sorted(range(len(sentences)), key=lambda i: -scores[i])[:keep])
        key = ("… " if best[0] > 0 else "") + " … ".join(sentences[i] for i in best)
    else:
        key = " ".join(sentences) or body
    kind = ("Fiche" if source["id"].upper().startswith("F")
            else "Extrait" if not source.get("url") else "Article")
    date = f", version du {source['version_date']}" if source.get("version_date") else ""
    head = f"{kind} {source['id']} · {source['source_label']}{date}"
    if title:
        head += f"\n{title}"
    full = f"\n\nTexte complet :\n{body}" if key != body else ""
    return f"{head}\n\nCe qui répond à ta question :\n« {key} »{full}"


def source_card(s: dict, answer: str) -> dict:
    """A source as Open WebUI shows it: a numbered card under the answer; opened, the excerpt that supports
    the answer, the label, version date and the link to the full text."""
    link = s["url"] or s["id"]
    return {"source": {"id": link, "name": s["id"], "url": s["url"]},
            "document": [excerpt(s, answer)],
            "metadata": [{"source": link, "name": s["id"]}]}


def numbered(text: str, known: set[str], final: bool = False) -> str:
    """Citations as numbers, [L6222-35] -> [1], in order of first appearance (Open WebUI links [1] to the
    first source card). Ids that were not retrieved stay as written. While streaming, an unclosed bracket
    is held back: "[L62" may become a citation."""
    if not final:
        cut = max(text.rfind("["), text.rfind("【"))
        if cut >= 0 and "]" not in text[cut:] and "】" not in text[cut:]:
            text = text[:cut]
    order: list[str] = []

    def number(match):
        ids = pipeline.cited_ids(match.group(0))
        if not ids or any(i not in known for i in ids):
            return match.group(0)
        for i in ids:
            if i not in order:
                order.append(i)
        return "".join(f"[{order.index(i) + 1}]" for i in ids)
    return pipeline.CITATION.sub(number, text)


def status(text: str, done: bool = False) -> dict:
    return {"event": {"type": "status", "data": {"description": text, "done": done}}}


def answer_events(question: str, history: list[dict]):
    """The pipeline step by step for a chat interface: status lines while searching, the model's reasoning
    and answer as they are written, then the sources as cards. Same decisions as pipeline.answer()."""
    check = guardrail.check(question)
    if check.blocked:
        yield {"content": check.message}
        yield status("Réponse fixe (garde-fou d'entrée)", done=True)
        return
    question = check.question
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
    known = {s["id"].upper() for s in sources}
    yield status(f"{len(known)} textes trouvés, je rédige la réponse…")
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
        visible = numbered(text[:end.start()] if end else held_back(text), known)
        visible = visible.rstrip("\n") + ("\n" if visible.endswith("\n") else "")  # no runs of blank lines
        if len(visible) > shown:
            yield {"content": visible[shown:]}
            shown = len(visible)
    if not_found or text.strip().upper().startswith(pipeline.NOT_FOUND):  # also when the stream ends on it
        yield {"content": pipeline.not_found_message(lang, found.missing_ids if found else []) + footer()}
        yield status("Les textes trouvés ne répondent pas à cette question", done=True)
        return
    end = SOURCES_START.search(text)
    answer_text = (text[:end.start()] if end else text).rstrip()
    visible = numbered(answer_text, known, final=True)
    if len(visible) > shown:
        yield {"content": visible[shown:]}
    refs = pipeline.citations(answer_text, sources)
    for s in refs["cited"]:  # same order as the numbers: order of first citation
        yield {"event": {"type": "source", "data": source_card(s, answer_text)}}
    if referral := pipeline.international_referral(refs, lang):
        yield {"content": referral}
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
    question = guardrail.mask_pii(messages[users[-1]]["content"])[0]  # before any cloud call
    content = interface_task(question)
    history = [m for m in messages[:users[-1]] if m["role"] in ("user", "assistant")]
    # Previous answers carry sources and the licence line: only the answer text is useful as context.
    history = [{**m, "content": re.split(r"\n\*\*Sources\*\*|\n\n\*Textes :", m["content"])[0]} for m in history]
    # The history goes to the cloud model too (follow-up rewrite): personal data masked first.
    history = [{**m, "content": guardrail.mask_pii(m["content"])[0]} for m in history]
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
