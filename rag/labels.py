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
DOC_CHARS_FOR_LABELS = 3000

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
- categories: 1 to 3, in English, Title Case: the topic of the rule as a person would name it, read from the place of the document in the code (its section titles). Never a procedure or form ("Notification", "Procedure", "Deadline"). Reuse a known category only if it names this exact topic; otherwise create a precise one.
- keywords: up to 12, copied in the document's original French wording and case, concrete (amounts, durations, conditions, named schemes), at most 6 words each.
- concepts: up to 8, in English, Title Case, each one the stable abstraction of one keyword from your list (give that keyword).
- Never use the name of the source, the code, or generic words such as "article", "employer", "employee".
- Ignore the legal plumbing: entry-into-force notes ("Nota"), decree or law numbers, dates of entry into force, references to other articles. Label what the rule says, not where it comes from.

Known categories: {known}

Document {citation_id} ({title})
Place in the code: {place}
{text}"""


def build_prompt(doc: dict, known_categories: list[str]) -> str:
    """Prompt for one document; known categories are the most used ones so far."""
    # Alphabetical, not by frequency: a frequency order pulls every document into the top category.
    known = ", ".join(sorted(known_categories[:MAX_KNOWN_CATEGORIES])) or "(none yet)"
    return PROMPT.format(known=known, citation_id=doc["citation_id"], title=doc["title"] or "",
                         place=place(doc.get("section_path")), text=doc["text"][:DOC_CHARS_FOR_LABELS])


def place(section_path: str | None, levels: int = 4) -> str:
    """The last section titles: "Caisses de congés payés > ... bâtiment" says what a short article is about."""
    if not section_path:
        return "(unknown)"
    return " > ".join(section_path.split(" > ")[-levels:])


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
# Procedures, not topics: as categories they would join unrelated rules.
GENERIC_CATEGORIES = {_fold(c) for c in ("Notification", "Procedure", "Procedures", "Deadline", "Deadlines",
                                         "Formalities", "Administrative Procedure", "Obligations", "Rights")}

# Legal plumbing, deterministic: references to other texts and entry-into-force notes.
# Amounts, durations and ages stay ("5 jours ouvrables", "moins de 18 ans").
PLUMBING = [
    re.compile(r"\b(articles?|art\.)\s+[LRD]\b", re.I),                    # "article R. 422-9"
    re.compile(r"\b[LRD]\.?\s?\*?\d{3,4}-\d"),                            # "L. 422-10", "R5221-7"
    re.compile(r"\b(d[ée]cret|decree|loi|law|ordonnance|ordinance)\b.*\d", re.I),  # "décret n° 2020-1734"
    re.compile(r"\b\d{4}-\d{2,4}\b"),                                       # bare text numbers "2020-1734"
    re.compile(r"\b(1er|\d{1,2})\s+(janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre|novembre|d[ée]cembre)\s+\d{4}\b", re.I),
    re.compile(r"\b(effective date|entry into force|entr(ent|e) en vigueur|in force|nota)\b", re.I),
]


def is_plumbing(label: str) -> bool:
    return any(p.search(label) for p in PLUMBING)


def _keep(label: str, seen: set[str]) -> bool:
    key = _fold(label)
    if not label or key in NOISE or key in seen or is_plumbing(label):
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
    # A category is a short topic: a copied section path ("Paid Leave > Section 5 : ...") is not one.
    categories = [c for c in categories if _fold(c) not in GENERIC_CATEGORIES
                  and not re.search(r"[>:]", c) and len(c.split()) <= 5]
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
    messages = [{"role": "user", "content": build_prompt(doc, known_categories)}]
    # Low reasoning: extraction needs no deliberation, and reasoning tokens eat the free rate limit.
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
    try:
        r = client.chat.completions.create(model=model, temperature=0, max_tokens=2000,
                                           response_format={"type": "json_object"}, messages=messages, **extra)
    except Exception as e:
        # Groq rejects output that is not strict JSON; the tolerant parser can often still read it.
        if "json" not in str(e).lower():
            raise
        r = client.chat.completions.create(model=model, temperature=0, max_tokens=2000, messages=messages, **extra)
    return postprocess(parse(r.choices[0].message.content or ""))
