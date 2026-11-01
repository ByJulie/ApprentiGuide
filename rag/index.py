"""Build the vector index from the chunks table. Only new or changed texts are recomputed.

Usage (from the repository root, after python -m ingest.load):
    python -m rag.index            # local model, no key needed
"""
import hashlib
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from rag import models

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def chunk_text(row: dict) -> str:
    """What is embedded: the place of the chunk in the code or fiche, then its text."""
    head = " > ".join(p for p in (row["section_path"], row["title"]) if p)
    return f"{row['citation_id']} | {head}\n{row['text']}"


def connect():
    conn = psycopg.connect(os.environ["DATABASE_URL"], row_factory=psycopg.rows.dict_row)
    conn.execute(open(os.path.join(ROOT, "rag", "schema.sql"), encoding="utf-8").read())
    register_vector(conn)
    return conn


def index_chunks(conn) -> None:
    rows = conn.execute("SELECT corpus, citation_id, part, title, section_path, text FROM chunks").fetchall()
    known = {(r["corpus"], r["citation_id"], r["part"]): r["text_hash"]
             for r in conn.execute("SELECT corpus, citation_id, part, text_hash FROM chunk_vectors")}
    todo = [r for r in rows if known.get((r["corpus"], r["citation_id"], r["part"])) != text_hash(chunk_text(r))]
    # Chunks removed from the corpus (scope change) leave the index too.
    current = {(r["corpus"], r["citation_id"], r["part"]) for r in rows}
    stale = [k for k in known if k not in current]
    with conn.cursor() as cur:
        cur.executemany("DELETE FROM chunk_vectors WHERE corpus = %s AND citation_id = %s AND part = %s", stale)
    print(f"chunks: {len(rows)} in corpus, {len(todo)} to embed, {len(stale)} removed")
    if not todo:
        return
    vectors = models.embed([chunk_text(r) for r in todo])
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunk_vectors (corpus, citation_id, part, text_hash, embedding)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (corpus, citation_id, part)
               DO UPDATE SET text_hash = EXCLUDED.text_hash, embedding = EXCLUDED.embedding""",
            [(r["corpus"], r["citation_id"], r["part"], text_hash(chunk_text(r)), v)
             for r, v in zip(todo, vectors)])
    conn.commit()


def main():
    load_dotenv(os.path.join(ROOT, ".env"))
    with connect() as conn:
        index_chunks(conn)


if __name__ == "__main__":
    main()
