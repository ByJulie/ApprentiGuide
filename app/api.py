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

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

import guardrail
import pipeline
from pipeline import display_heading, display_name, source_link

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
    if "\n" in text and (last.strip() and not word or word and "sources".startswith(word[:7])):
        return text[:text.rfind("\n")]  # "**" alone may be the start of "**Sources**"
    return text


# Em dashes in the model's answer read as ", " (display only: the evaluated text is unchanged).
DASH = re.compile(r"\s*\u2014\s*")


def no_dashes(text: str) -> str:
    """Prefix-stable while streaming: trailing spaces (also no-break ones, which models put before a dash)
    wait, so "X —" then "X — Y" give "X," then "X, Y"."""
    return DASH.sub(", ", re.sub(r"[^\S\n]+\Z", "", text))


def footer(lang: str = "fr") -> str:
    """Source and date of the texts (the attribution the Licence Ouverte asks for; the licence itself is named in
    the README) and what to do before deciding, in words a student understands."""
    date = french_date(str(pipeline.CFG.get("corpus_snapshot", "")))
    if lang == "en":
        return (f"\n\n---\n*From the official texts (French Labour Code, Service-Public.fr), up to date as of {date}. "
                f"Before you decide, check with your school or HR.*")
    return (f"\n\n---\n*Tiré des textes officiels (Code du travail, Service-Public.fr), à jour au {date}. "
            f"Avant de décider, vérifie avec ton école ou tes RH.*")


WORDS = {"fr": {"sources": "**Sources** :", "school": "Extrait", "fiche": "Fiche", "article": "Article",
                "school_date": " · version du ", "fiche_date": " · mise à jour le ",
                "article_date": " · version en vigueur depuis le "},
         "en": {"sources": "**Sources:**", "school": "Extract", "fiche": "Fact sheet", "article": "Article",
                "school_date": " · version of ", "fiche_date": " · updated on ", "article_date": " · in force since "}}


def french_date(iso: str | None) -> str:
    return "-".join(reversed(iso.split("-"))).replace("-", "/") if iso else ""


def versions_line(cited: list[dict], links: bool = False, lang: str = "fr") -> str:
    """Under the answer, every cited text with its version (asked by the course): the date the article's
    version came into force (Legifrance), the last update of the fact sheet (Service-Public.fr)."""
    if not cited:
        return ""
    w, items = WORDS.get(lang, WORDS["fr"]), []
    for n, s in enumerate(cited, 1):
        fiche = s["id"].upper().startswith("F")
        school = not s.get("url")  # school regulations: private, no public page
        kind = w["school"] if school else w["fiche"] if fiche else w["article"]
        when = ""
        if s.get("version_date"):
            prefix = w["school_date"] if school else w["fiche_date"] if fiche else w["article_date"]
            when = prefix + french_date(s["version_date"])
        if school:  # "[1] Règlement pédagogique 2025-2026, section 3.3.9 · version du ... · heading"
            heading = f" · {display_heading(s)}" if display_heading(s) else ""
            items.append(f"[{n}] {display_name(s)}{when}{heading}")
            continue
        name = source_link(s) if links else s["id"]
        items.append(f"[{n}] {kind} {name} · {s['source_label']}{when}")
    return f"\n\n{w['sources']}  \n" + "  \n".join(items)


def render(out: dict) -> str:
    """Plain Markdown answer for clients that do not stream: the answer, then its sources from the
    database. The model's own "Sources:" line is dropped: the block built from the database replaces it."""
    lang = out.get("lang", "fr")
    parts = [no_dashes(MODEL_SOURCES_LINE.sub("", out["answer"]).rstrip())]
    refs = out.get("citations", {})
    if refs.get("cited"):
        parts.append(versions_line(refs["cited"], links=True, lang=lang))  # no source cards outside a chat interface
    if refs.get("invented"):
        parts.append(UNVERIFIED.get(lang, UNVERIFIED["fr"]).format(ids=", ".join(refs["invented"])))
    return "\n".join(parts) + footer(lang)


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
    date = f", version du {french_date(source['version_date'])}" if source.get("version_date") else ""
    if not source.get("url"):  # school regulations: readable name, the heading follows as title
        kind, head = "Extrait", f"{display_name(source)}{date}"
    else:
        head = f"{kind} {source['id']} · {source['source_label']}{date}"
    if title:
        head += f"\n{title}"
    full = f"\n\nTexte complet :\n{body}" if key != body else ""
    return f"{head}\n\nCe qui répond à ta question :\n« {key} »{full}"


def source_card(s: dict, answer: str) -> dict:
    """A source as Open WebUI shows it: a numbered card under the answer; opened, the excerpt that supports
    the answer, the label, version date and the link to the full text."""
    link = s["url"] or s["id"]
    name = display_name(s)
    return {"source": {"id": link, "name": name, "url": s["url"]},
            "document": [excerpt(s, answer)],
            "metadata": [{"source": link, "name": name}]}


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
        ids = [pipeline.matched_id(i, known) for i in pipeline.cited_ids(match.group(0))]
        if not ids or None in ids:
            return match.group(0)
        for i in ids:
            if i not in order:
                order.append(i)
        return "".join(f"[{order.index(i) + 1}]" for i in ids)
    return pipeline.CITATION.sub(number, text)


STATUS = {
    "fr": {"guard": "Réponse fixe (garde-fou d'entrée)", "history": "Je relis la conversation…",
           "search": "Je cherche dans le Code du travail, le CESEDA et les fiches Service-Public…",
           "none": "Aucun texte ne répond à cette question", "writing": "{n} textes trouvés, je rédige la réponse…",
           "weak": "Les textes trouvés ne répondent pas à cette question", "done": "Réponse fondée sur {n} texte(s)"},
    "en": {"guard": "Fixed answer (input guardrail)", "history": "Reading the conversation…",
           "search": "Searching the Labour Code, the CESEDA and the Service-Public fact sheets…",
           "none": "No text answers this question", "writing": "{n} texts found, writing the answer…",
           "weak": "The texts found do not answer this question", "done": "Answer based on {n} text(s)"},
}


UNVERIFIED = {"fr": "\n\n⚠️ Citation non vérifiée (absente des textes trouvés) : {ids}",
              "en": "\n\n⚠️ Unverified citation (not among the texts found): {ids}"}


def say(key: str, lang: str) -> str:
    return STATUS.get(lang, STATUS["fr"])[key]


def status(text: str, done: bool = False) -> dict:
    return {"event": {"type": "status", "data": {"description": text, "done": done}}}


def answer_events(question: str, history: list[dict]):
    """The pipeline step by step for a chat interface: status lines while searching, the model's reasoning
    and answer as they are written, then the sources as cards. Same decisions as pipeline.answer()."""
    lang = pipeline.language(question, None)  # first guess for the status lines; set again after the rewrite
    check = guardrail.check(question)
    if check.blocked:
        yield {"content": check.message}
        yield status(say("guard", lang), done=True)
        return
    question = check.question
    if history:
        yield status(say("history", lang))
        question = pipeline.standalone(question, history)
    yield status(say("search", lang))
    found = pipeline.retrieve(question)
    lang = pipeline.language(question, found.question_fr if found else None)
    if found is not None and found.abstain:
        yield {"content": pipeline.not_found_message(lang, found.missing_ids) + footer(lang)}
        yield status(say("none", lang), done=True)
        return
    sources = found.sources if found else []
    known = {s["id"].upper() for s in sources}
    yield status(say("writing", lang).format(n=len(known)))
    text, shown, not_found = "", 0, False
    for kind, piece in pipeline.generate_stream(question, sources, found.missing_ids if found else [], lang, history):
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
        visible = no_dashes(numbered(text[:end.start()] if end else held_back(text), known))
        visible = visible.rstrip("\n") + ("\n" if visible.endswith("\n") else "")  # no runs of blank lines
        if len(visible) > shown:
            yield {"content": visible[shown:]}
            shown = len(visible)
    if not_found or text.strip().upper().startswith(pipeline.NOT_FOUND):  # also when the stream ends on it
        yield {"content": pipeline.not_found_message(lang, found.missing_ids if found else []) + footer(lang)}
        yield status(say("weak", lang), done=True)
        return
    end = SOURCES_START.search(text)
    answer_text = (text[:end.start()] if end else text).rstrip()
    visible = no_dashes(numbered(answer_text, known, final=True))
    if len(visible) > shown:
        yield {"content": visible[shown:]}
    refs = pipeline.citations(answer_text, sources)
    if not refs["cited"]:
        # The model sometimes cites only in its own final "Sources:" line, which is not shown: its ids are
        # still checked against the retrieved texts, so the cards, versions and referral are not lost.
        refs = pipeline.citations(text, sources)
    for s in refs["cited"]:  # same order as the numbers: order of first citation
        yield {"event": {"type": "source", "data": source_card(s, answer_text)}}
    if referral := pipeline.international_referral(refs, lang):
        yield {"content": referral}
    if refs["invented"]:
        yield {"content": UNVERIFIED.get(lang, UNVERIFIED["fr"]).format(ids=", ".join(refs["invented"]))}
    yield {"content": versions_line(refs["cited"], links=True, lang=lang) + footer(lang)}
    yield status(say("done", lang).format(n=len(refs["cited"])), done=True)


@app.post("/v1/audio/transcriptions")
def transcriptions(file: UploadFile = File(...), model: str = Form(""), language: str | None = Form(None)):
    """Speech-to-text for the chat interface (Open WebUI microphone), OpenAI format: our local Whisper
    (app/voice.py: small model, voice activity detection, French or English, domain words), instead of
    Open WebUI's default (base model, no VAD, any language). The audio is read in memory, never stored."""
    from voice import transcribe
    heard = transcribe(file.file.read(), (language or "").lower() or None)
    return {"text": heard["text"], "language": heard["language"]}


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
    history = [{**m, "content": re.split(r"\n\*\*Sources(\*\*|:)|\n\n---\n\*(Text(es :|s:)|Information|Tiré des textes|From the official)|\n\n\*Textes :", m["content"])[0].strip()}
               for m in history]
    # The history goes to the cloud model too (follow-up rewrite): personal data masked first.
    history = [{**m, "content": guardrail.mask_pii(m["content"])[0]} for m in history]
    reply_id, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
    if not body.get("stream"):
        if content is None:
            content = render(pipeline.answer(pipeline.standalone(question, history), history))
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
