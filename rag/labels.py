"""Short labels per document (categories, keywords, concepts), extracted by the LLM then cleaned by code.

Why: a question in English or in everyday words ("my boss", "lunch vouchers") rarely shares words
with the legal text. Short English categories and concepts give the embedding a bridge between
languages and registers; the original French keywords keep the exact legal terms.
The LLM proposes, the code decides: limits, deduplication and noise filters are deterministic.
"""
import json
import re
import unicodedata

MAX_CATEGORIES = 3
MAX_KEYWORDS = 12
MAX_CONCEPTS = 8
MAX_KEYWORD_WORDS = 6
# Shown to the LLM so it reuses categories instead of inventing near-duplicates.
MAX_KNOWN_CATEGORIES = 60
DOC_CHARS_FOR_LABELS = 4000

# Words that describe the source, not the rule: they match every document, so they rank nothing.
NOISE = {
    "article", "code du travail", "legifrance", "service-public", "service-public.fr", "fiche",
    "fiche pratique", "loi", "décret", "decret", "law", "labour code", "labor code", "france",
    "french law", "regulation", "réglementation", "dispositions", "disposition", "salarié",
    "employee", "employer", "employeur",
}

PROMPT = """You index French employment-law documents for a search engine used by apprentices and students.
Return JSON only, with this shape:
{{"categories": [...], "keywords": [...], "concepts": [{{"concept": "...", "keyword": "..."}}]}}

Rules:
- categories: 1 to 3, in English, Title Case, abstract (e.g. "Paid Leave", not "congés payés en apprentissage"). Reuse a known category when one fits.
- keywords: up to 12, copied in the document's original French wording and case, concrete (amounts, durations, conditions, named schemes), at most 6 words each.
- concepts: up to 8, in English, Title Case, each one the stable abstraction of one keyword from your list (give that keyword).
- Never use the name of the source, the code, or generic words such as "article", "employer", "employee".

Known categories: {known}

Document {citation_id} ({title}):
{text}"""


def build_prompt(doc: dict, known_categories: list[str]) -> str:
    """Prompt for one document; known categories are the most used ones so far."""
    known = ", ".join(known_categories[:MAX_KNOWN_CATEGORIES]) or "(none yet)"
    return PROMPT.format(known=known, citation_id=doc["citation_id"], title=doc["title"] or "",
                         text=doc["text"][:DOC_CHARS_FOR_LABELS])


def parse(raw: str) -> dict:
    """JSON object from the model output, tolerant of a code fence or text around it."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip(" .,;:\"'") if value else ""


def _fold(text: str) -> str:
    """Lowercase without accents, so "Légifrance" and "legifrance" are the same label."""
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


NOISE = {_fold(n) for n in NOISE}


def _keep(label: str, seen: set[str]) -> bool:
    key = _fold(label)
    if not label or key in NOISE or key in seen:
        return False
    seen.add(key)
    return True


def postprocess(data: dict) -> list[tuple[str, str]]:
    """(kind, label) pairs after the deterministic rules.

    Order matters: categories first, so a term proposed as both category and keyword stays a
    category only; deduplication is case-insensitive across all kinds.
    """
    seen: set[str] = set()
    out = []
    categories = [_clean(c) for c in data.get("categories") or [] if isinstance(c, str)]
    for label in [c for c in categories if _keep(c, seen)][:MAX_CATEGORIES]:
        out.append(("category", label))
    keywords = [_clean(k) for k in data.get("keywords") or [] if isinstance(k, str)]
    keywords = [k for k in keywords if len(k.split()) <= MAX_KEYWORD_WORDS]
    for label in [k for k in keywords if _keep(k, seen)][:MAX_KEYWORDS]:
        out.append(("keyword", label))
    concepts = []
    for item in data.get("concepts") or []:
        concept = _clean(item.get("concept")) if isinstance(item, dict) else _clean(item)
        if concept:
            concepts.append(concept)
    for label in [c for c in concepts if _keep(c, seen)][:MAX_CONCEPTS]:
        out.append(("concept", label))
    return out


def extract(client, model: str, doc: dict, known_categories: list[str]) -> list[tuple[str, str]]:
    """Labels of one document; an unreadable answer gives no labels (the document stays searchable by content)."""
    r = client.chat.completions.create(
        model=model, temperature=0, max_tokens=2000,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": build_prompt(doc, known_categories)}])
    return postprocess(parse(r.choices[0].message.content or ""))
