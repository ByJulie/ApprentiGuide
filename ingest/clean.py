"""Turn the raw Legifrance and service-public records into chunks ready for the database.

Pure functions only (no file or database access), so they can be tested and reused.
"""
import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

# About 750 tokens: long enough to keep a rule with its conditions, short enough to retrieve.
MAX_CHUNK_CHARS = 3000
HEADING_MAX_CHARS = 40
HEADING_END_MARKS = (".", ":", ";", "?", "!", ")", "%")
# ABROGE_DIFF: repealed from a later date, so still applicable today.
KEPT_ARTICLE_STATES = {"VIGUEUR", "ABROGE_DIFF"}
# Menus that only lead to other fiches, not rules.
MENU_TYPES = {"Comment faire si...", "Recherche guidée"}
ARTICLE_URL = "https://www.legifrance.gouv.fr/codes/article_lc/"


@dataclass
class Chunk:
    """One row of the chunks table."""

    source: str
    corpus: str
    citation_id: str
    part: int
    title: str
    section_path: str
    text: str
    version_date: str | None
    source_label: str
    url: str | None
    scope: dict
    lang: str = "fr"


def build_scope(theme: str | None) -> dict:
    """Scope from structure only (the theme of the scope entry that selected the document)."""
    scope = {}
    if theme:
        scope["theme"] = theme
    if theme == "etranger":
        scope["audience"] = "international"
    return scope


def theme_for(article_number: str, patterns: list[dict]) -> str | None:
    """Theme of the first scope pattern matching an article number, or None."""
    for pattern in patterns:
        if re.match(pattern["regex"], article_number):
            return pattern["theme"]
    return None


def epoch_ms_to_date(value) -> str | None:
    """ISO date from a Legifrance timestamp in milliseconds, None when missing."""
    if not value:
        return None
    return datetime.fromtimestamp(int(value) / 1000, UTC).date().isoformat()


def legifrance_chunks(articles: list[dict], corpus: str, patterns: list[dict],
                      label: str) -> list[Chunk]:
    """One chunk per in-force article; the article number is the citation."""
    seen, chunks = set(), []
    for article in articles:
        number = (article.get("num") or "").strip()
        text = "\n".join(line.strip() for line in (article.get("texte") or "").splitlines())
        if article.get("etat") not in KEPT_ARTICLE_STATES or not text.strip() or not number:
            continue
        if number in seen:
            continue
        seen.add(number)
        # The nota can limit when the article applies (date, court ruling): keep it with the text.
        nota = " ".join((article.get("nota") or "").split())
        body = text.strip() + (f"\n\nNota : {nota}" if nota else "")
        chunks.append(Chunk(
            source="legifrance", corpus=corpus, citation_id=number, part=0,
            title=f"Article {number}", section_path=article.get("sectionPath", ""),
            text=body, version_date=epoch_ms_to_date(article.get("dateDebut")),
            source_label=label, url=ARTICLE_URL + article["id"],
            scope=build_scope(theme_for(number, patterns))))
    return chunks


def is_heading(line: str) -> bool:
    """A short line with no closing punctuation, not a list item or a table row."""
    return (len(line) < HEADING_MAX_CHARS and not line.endswith(HEADING_END_MARKS)
            and not line.startswith("- ") and "|" not in line)


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split on line boundaries into pieces of at most max_chars (a longer line stays whole).

    A heading never ends a piece: it moves to the next one, with the text it introduces.
    """
    pieces, current = [], ""
    for line in text.split("\n"):
        if current and len(current) + len(line) + 1 > max_chars:
            head, _, last = current.rpartition("\n")
            if head and is_heading(last):
                pieces.append(head)
                current = f"{last}\n{line}"
            else:
                pieces.append(current)
                current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        pieces.append(current)
    return pieces


def service_public_chunks(fiches: list[dict]) -> list[Chunk]:
    """Chunks of the fiches: menus dropped, identical fiches kept once, long sections split."""
    seen, chunks = set(), []
    for fiche in fiches:
        if fiche["type"] in MENU_TYPES:
            continue
        body = "\n".join(s["text"] for s in fiche["sections"])
        key = (fiche["id"], hashlib.sha256(body.encode()).hexdigest())
        if key in seen:
            continue
        seen.add(key)
        part = 0
        for section in fiche["sections"]:
            for index, piece in enumerate(split_text(section["text"])):
                label = section["title"] + (f" (suite {index})" if index else "")
                chunks.append(Chunk(
                    source="service-public", corpus=fiche["corpus"], citation_id=fiche["id"],
                    part=part, title=f"{fiche['title']} : {label}",
                    section_path=fiche["sectionPath"], text=piece,
                    version_date=fiche["versionDate"], source_label=fiche["sourceLabel"],
                    url=fiche["url"], scope=build_scope(fiche["theme"])))
                part += 1
    return chunks


def chunk_rows(chunks: list[Chunk]) -> list[dict]:
    """Chunks as plain dicts, without the NUL characters PostgreSQL refuses in text."""
    rows = []
    for chunk in chunks:
        row = asdict(chunk)
        rows.append({k: v.replace("\x00", "") if isinstance(v, str) else v
                     for k, v in row.items()})
    return rows
