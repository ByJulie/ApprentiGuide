"""Extract the service-public.fr fiches of the scope (ingest/scope.yaml) from the DILA open data.

Usage: python -I ingest/service_public.py
Downloads each zip once (data/raw/sp_<audience>/), reads only the fiches in scope straight from
the archive, and writes data/raw/service_public_<audience>.jsonl, one fiche per line with its
sections as plain text. Source: Service-Public.fr/DILA, Licence Ouverte.
"""
import email.utils
import json
import os
import re
import zipfile

import requests
import yaml
from lxml import etree

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))["service_public"]
SCOPE = yaml.safe_load(open(os.path.join(ROOT, "ingest", "scope.yaml"), encoding="utf-8"))
RAW = os.path.join(ROOT, "data", "raw")
SOURCE_LABEL = "Service-Public.fr/DILA"
DC = "{http://purl.org/dc/elements/1.1/}"
MAX_XML_BYTES = 5_000_000
# Dropped on purpose: layout and images (TitreRiche, Infographie), cross links (VoirAussi) and
# legal references (Reference). Also not read: Definition, PourEnSavoirPlus and QuiPeutMAider
# (glossary, further reading, contacts), which carry no rule.
SKIPPED_TAGS = {"TitreRiche", "Infographie", "Reference", "VoirAussi"}
PARSER = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)


def squash(text: str) -> str:
    """Collapse whitespace, including the non-breaking spaces of French typography."""
    return " ".join(text.split())


def render(node) -> list[str]:
    """Plain-text lines of an XML node: paragraphs, bullet items, table rows."""
    tag = etree.QName(node).localname
    if tag in SKIPPED_TAGS:
        return []
    if tag in ("Paragraphe", "Titre") and not any(
            etree.QName(c).localname in ("Liste", "Paragraphe") for c in node):
        text = squash("".join(node.itertext()))
        return [text] if text else []
    if tag == "Item":
        inner = " ".join(line for c in node for line in render(c))
        text = squash(inner or "".join(node.itertext()))
        return ["- " + text] if text else []
    if tag == "Tableau":
        return table_lines(node)
    if tag == "ServiceEnLigne":
        return [service_line(node)]
    return [line for child in node for line in render(child)]


def service_line(node) -> str:
    """Name of an online service (and its publisher): the sentence before it points to it."""
    title_node, source_node = node.find("Titre"), node.find("Source")
    title = squash("".join(title_node.itertext())) if title_node is not None else ""
    source = squash("".join(source_node.itertext())) if source_node is not None else ""
    return f"Service en ligne : {title}" + (f" ({source})" if source else "")


def cell_text(cell) -> str:
    """Text of a table cell, its paragraphs separated by a space so words do not stick."""
    paragraphs = [squash("".join(p.itertext())) for p in cell.iter("Paragraphe")]
    return squash(" ".join(paragraphs) or "".join(cell.itertext()))


def table_lines(table) -> list[str]:
    """Text lines of a table: one line per row, cells separated by ' | '.

    A table of 3 columns or more whose header row has the same width as the row gets one line
    per cell, "<row label>, <column header> : <value>", so a value never has to be matched to
    its column by counting separators.
    """
    headers = [cell_text(c) for c in table if etree.QName(c).localname == "Colonne"]
    lines = [" | ".join(headers)] if any(headers) else []
    header_row = None
    for row in table:
        if etree.QName(row).localname != "Rangée":
            continue
        cells = [cell_text(c) for c in row]
        if row.get("type") == "header" and header_row is None and len(cells) >= 3:
            header_row = cells
        elif header_row and len(cells) == len(header_row):
            lines += [f"{cells[0]}, {head} : {value}"
                      for head, value in zip(header_row[1:], cells[1:], strict=True) if value]
            continue
        lines.append(" | ".join(cells))
    return lines


def titled_section(node, prefix: str = "") -> dict:
    """A section from a node with a Titre child: the title (after prefix) and the rest as text."""
    title_node = node.find("Titre")
    title = squash("".join(title_node.itertext())) if title_node is not None else ""
    lines = [line for c in node if c is not title_node for line in render(c)]
    return {"title": " : ".join(t for t in (prefix, title) if t), "text": "\n".join(lines)}


def text_sections(texte, prefix: str = "") -> list[dict]:
    """Sections of a Texte node: the text outside chapters, then one section per chapter."""
    loose = [line for c in texte if etree.QName(c).localname != "Chapitre"
             for line in render(c)]
    sections = [{"title": prefix or "Hors chapitre", "text": "\n".join(loose)}] if loose else []
    return sections + [titled_section(c, prefix) for c in texte.findall("Chapitre")]


def parse_fiche(xml_bytes: bytes, corpus: str, theme: str, file_date: str) -> dict:
    """Turn one fiche XML into a record: metadata plus sections of plain text."""
    root = etree.fromstring(xml_bytes, PARSER)
    sections = []
    intro = root.find("Introduction")
    if intro is not None:
        sections.append({"title": "Introduction", "text": "\n".join(render(intro))})
    if root.find("Texte") is not None:
        sections += text_sections(root.find("Texte"))
    for situation in root.iterfind("ListeSituations/Situation"):
        title = squash("".join(situation.find("Titre").itertext()))
        sections += text_sections(situation.find("Texte"), title)
    for tag in ("Avertissement", "QuestionReponse"):
        sections += [titled_section(node) for node in root.findall(tag)]
    path = [squash("".join(n.itertext())) for n in root.iterfind("FilDAriane/Niveau")]
    modified = re.search(r"\d{4}-\d{2}-\d{2}", root.findtext(DC + "date") or "")
    return {
        "id": root.get("ID"),
        "corpus": corpus,
        "theme": theme,
        "type": root.findtext(DC + "type"),
        "title": squash(root.findtext(DC + "title") or ""),
        "description": squash(root.findtext(DC + "description") or ""),
        "audiences": [a.text for a in root.iterfind("Audience")],
        "sectionPath": " > ".join(path),
        "url": root.get("spUrl"),
        "versionDate": modified.group(0) if modified else None,
        "fileDate": file_date,
        "sourceLabel": SOURCE_LABEL,
        "sections": [s for s in sections if s["text"].strip()],
    }


def fetch_zip(audience: str, url: str) -> tuple[str, str]:
    """Download the archive once; return its path and the date the publisher gave the file."""
    folder = os.path.join(RAW, f"sp_{audience}")
    zip_path = os.path.join(folder, "vosdroits-latest.zip")
    meta_path = os.path.join(folder, "file_date.txt")
    if not (os.path.exists(zip_path) and os.path.exists(meta_path)):
        os.makedirs(folder, exist_ok=True)
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        file_date = email.utils.parsedate_to_datetime(response.headers["Last-Modified"])
        with open(zip_path, "wb") as f:
            f.write(response.content)
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(file_date.date().isoformat())
    return zip_path, open(meta_path, encoding="utf-8").read().strip()


def main():
    """Write one JSONL per audience with the fiches listed in scope.yaml."""
    for audience, themes in SCOPE["service_public"].items():
        zip_path, file_date = fetch_zip(audience, CFG[f"{audience}_zip"])
        corpus = "sp_" + audience
        out_path = os.path.join(RAW, f"service_public_{audience}.jsonl")
        archive = zipfile.ZipFile(zip_path)
        sizes = {i.filename: i.file_size for i in archive.infolist()}
        count = 0
        with open(out_path, "w", encoding="utf-8") as out:
            for theme, ids in themes.items():
                for fiche_id in ids:
                    member = f"{fiche_id}.xml"
                    if member not in sizes or sizes[member] > MAX_XML_BYTES:
                        raise ValueError(f"fiche {fiche_id} missing or too large in {zip_path}")
                    record = parse_fiche(archive.read(member), corpus, theme, file_date)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
        print(f"{audience}: {count} fiches, file date {file_date} -> {out_path}")


if __name__ == "__main__":
    main()
