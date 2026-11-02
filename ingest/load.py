"""Build the chunks from data/raw and load them into PostgreSQL (full reload, so it is idempotent).

Usage (from the repository root): python -m ingest.load
"""
import json
import os

import psycopg
import yaml
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from ingest import clean, regulations

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "data", "raw")
PRIVATE = os.path.join(ROOT, "data", "private")
CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))["database"]
SCOPE = yaml.safe_load(open(os.path.join(ROOT, "ingest", "scope.yaml"), encoding="utf-8"))
COLUMNS = ["source", "corpus", "citation_id", "part", "title", "section_path", "text",
           "version_date", "source_label", "url", "scope", "lang"]


def read_jsonl(name: str) -> list[dict]:
    """Records of a data/raw JSONL file."""
    with open(os.path.join(RAW, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_chunks() -> list[clean.Chunk]:
    """All chunks from the raw files."""
    chunks = []
    for corpus, config in SCOPE["legifrance"].items():
        chunks += clean.legifrance_chunks(
            read_jsonl(f"legifrance_{corpus}.jsonl"), corpus, config["patterns"], config["label"])
    fiches = []
    for audience in SCOPE["service_public"]:
        fiches += read_jsonl(f"service_public_{audience}.jsonl")
    return chunks + clean.service_public_chunks(fiches) + regulation_chunks()


def regulation_chunks() -> list[clean.Chunk]:
    """Chunks of the school regulations whose PDF is in data/private (others are skipped)."""
    chunks = []
    for school, config in SCOPE["regulations"].items():
        for kind, doc in config["documents"].items():
            path = os.path.join(PRIVATE, doc["file"])
            if not os.path.exists(path):
                print(f"Warning: {path} not found, {school} {kind} chunks left out")
                continue
            chunks += regulations.regulation_chunks(
                regulations.read_pdf(path), school=school, kind=kind, doc_label=doc["label"],
                corpus=doc["corpus"], source_label=doc["source_label"],
                version_date=doc["version_date"], articles=doc.get("articles"))
    return chunks


def load(chunks: list[clean.Chunk], database_url: str) -> None:
    """Replace the content of the chunks table (schema created if missing) in one transaction."""
    rows = [{**row, "scope": Jsonb(row["scope"])} for row in clean.chunk_rows(chunks)]
    with open(os.path.join(ROOT, CFG["schema_file"]), encoding="utf-8") as f:
        schema = f.read()
    placeholders = ", ".join(f"%({column})s" for column in COLUMNS)
    insert = f"INSERT INTO {CFG['table']} ({', '.join(COLUMNS)}) VALUES ({placeholders})"
    with psycopg.connect(database_url) as conn:
        conn.execute(schema)
        conn.execute(f"TRUNCATE {CFG['table']} RESTART IDENTITY")
        with conn.cursor() as cur:
            cur.executemany(insert, rows)


def main():
    """Build the chunks and load them; DATABASE_URL comes from .env."""
    load_dotenv(os.path.join(ROOT, ".env"))
    chunks = build_chunks()
    load(chunks, os.environ["DATABASE_URL"])
    print(f"{len(chunks)} chunks loaded")


if __name__ == "__main__":
    main()
