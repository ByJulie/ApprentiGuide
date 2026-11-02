"""Chunks of a school's regulations (PDF files kept in data/private, never committed).

One chunk per article (internal rules) or per numbered section (study rules), cited as
`<school>-<document>-<number>`, for example `aivancity-RP-3.3.4`. Pure functions on the text,
except `read_pdf`.
"""
import re
import unicodedata

from ingest import clean

# Dotted leaders and underscores of the table of contents, and lines made of a page number only.
LEADER = re.compile(r"(?:\.\s*){4,}|(?:_\s*){4,}|\s\.\.\s*\d+$")
PAGE_NUMBER = re.compile(r"^\d{1,3}$")
TOC_SECTION = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*?)\s*(?:\.\s*){2,}\d+$")
ARTICLE = re.compile(r"^ARTICLE\s+(\d+)\s+(\S.*)$")
RI_SUBSECTION = re.compile(r"^(\d+\.\d+)\.?\s+(\S.*)$")


def read_pdf(path: str) -> str:
    """Text of a PDF (layout mode: the default mode inserts spaces inside words)."""
    from pypdf import PdfReader
    return "\n".join(page.extract_text(extraction_mode="layout") for page in PdfReader(path).pages)


def clean_lines(text: str) -> list[str]:
    """Lines without indentation, page numbers or the bullet glyph; ligatures expanded (NFKC)."""
    text = unicodedata.normalize("NFKC", text).replace("", "-")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return [line for line in lines if not PAGE_NUMBER.match(line)]


def toc_numbers(lines: list[str]) -> dict[str, str]:
    """Number -> title of the sections listed in the table of contents."""
    return {m.group(1): m.group(2) for m in map(TOC_SECTION.match, lines) if m}


def sentence_case(title: str) -> str:
    """Capitalised title when it is written in capitals."""
    return title.capitalize() if title.isupper() else title


def split_sections(lines: list[str], kind: str) -> list[tuple[str, str, list[str]]]:
    """(number, title, body lines) in reading order; kind is "RI" (articles) or "RP" (numbered).

    Text before the first heading is dropped. RI: an article is cut at its "7.3." subsections.
    RP: a heading is a line starting with a number that the table of contents lists.
    """
    known = toc_numbers(lines) if kind == "RP" else {}
    lines = [line for line in lines if not LEADER.search(line)]
    sections, article, wrapped = [], None, ""
    for line in lines:
        if wrapped and line == wrapped:
            wrapped = ""
            continue
        number = title = None
        if kind == "RI":
            if m := ARTICLE.match(line):
                article, number, title = m.group(1), m.group(1), m.group(2)
            elif (article and (m := RI_SUBSECTION.match(line))
                  and m.group(1).split(".")[0] == article):
                number, title = m.group(1), m.group(2)
        elif ((m := re.match(r"^(\d+(?:\.\d+)*)\s+(\S.*)$", line)) and m.group(1) in known
              and ("." in m.group(1) or m.group(2).isupper())):
            number, title = m.group(1), known[m.group(1)]
            # A long heading wraps onto a second line, which is not part of the body.
            wrapped = title[len(m.group(2)):].strip() if title.startswith(m.group(2)) else ""
        if number:
            sections.append((number, sentence_case(title.strip()), []))
        elif sections:
            sections[-1][2].append(line)
    return sections


def regulation_chunks(text: str, *, school: str, kind: str, doc_label: str, corpus: str,
                      source_label: str, version_date: str | None,
                      articles: list[str] | None = None) -> list[clean.Chunk]:
    """Chunks of one document; `articles` keeps only those article numbers (RI)."""
    chunks, titles = [], {}
    for number, title, body in split_sections(clean_lines(text), kind):
        titles[number] = title
        if articles and number.split(".")[0] not in articles:
            continue
        body_text = "\n".join(line for line in body).strip()
        if not body_text:
            continue
        ancestors = [".".join(number.split(".")[:i]) for i in range(1, number.count(".") + 1)]
        parents = [titles[a] for a in ancestors if a in titles]
        for part, piece in enumerate(clean.split_text(body_text)):
            chunks.append(clean.Chunk(
                source="reglement", corpus=corpus, citation_id=f"{school}-{kind}-{number}",
                part=part, title=f"{doc_label} {number} : {title}",
                section_path=" > ".join(parents), text=piece, version_date=version_date,
                source_label=source_label, url=None,
                scope=clean.build_scope("ecole", school)))
    return chunks
