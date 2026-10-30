"""Build the retrieval indexes from the chunks table. Only new or changed texts are recomputed.

Usage (from the repository root, after python -m ingest.load):
    python -m rag.index            # content vectors (local model, no key needed)
    python -m rag.index --labels   # + labels per document (LLM from config.yaml, then local vectors)
"""
import argparse
import hashlib
import os
import time
from collections import Counter

import psycopg
import yaml
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from rag import labels, models

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))


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


def documents(conn) -> list[dict]:
    """One record per article or fiche: its parts joined in order."""
    return conn.execute(
        """SELECT corpus, citation_id, min(title) AS title,
                  string_agg(text, E'\n' ORDER BY part) AS text
           FROM chunks GROUP BY corpus, citation_id ORDER BY corpus, citation_id""").fetchall()


def llm_client():
    from openai import OpenAI
    key_env = CFG["model"].get("api_key_env", "none")
    key = "ollama" if key_env == "none" else os.environ.get(key_env)
    if not key:
        raise SystemExit(f"{key_env} is empty in .env: labels need the LLM from config.yaml")
    return OpenAI(base_url=CFG["model"]["base_url"], api_key=key, max_retries=8)


def index_labels(conn) -> None:
    docs = documents(conn)
    done = {(r["corpus"], r["citation_id"]): r["doc_hash"] for r in conn.execute(
        "SELECT DISTINCT corpus, citation_id, doc_hash FROM doc_labels")}
    todo = [d for d in docs if done.get((d["corpus"], d["citation_id"])) != text_hash(d["text"])]
    print(f"labels: {len(docs)} documents, {len(todo)} to label")
    category_counts = Counter(r["label"] for r in conn.execute(
        "SELECT label FROM doc_labels WHERE kind = 'category'"))
    client, model = llm_client(), CFG["model"]["name"]
    for i, doc in enumerate(todo, 1):
        known = [c for c, _ in category_counts.most_common()]
        try:
            pairs = labels.extract(client, model, doc, known)
        except Exception as e:  # one failed document must not stop a run of several hundred
            print(f"  {doc['citation_id']}: skipped ({type(e).__name__}: {str(e)[:120]})")
            time.sleep(5)
            continue
        category_counts.update(label for kind, label in pairs if kind == "category")
        vectors = models.embed([label for _, label in pairs]) if pairs else []
        with conn.cursor() as cur:
            cur.execute("DELETE FROM doc_labels WHERE corpus = %s AND citation_id = %s",
                        (doc["corpus"], doc["citation_id"]))
            cur.executemany(
                """INSERT INTO doc_labels (corpus, citation_id, kind, label, doc_hash, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                [(doc["corpus"], doc["citation_id"], kind, label, text_hash(doc["text"]), v)
                 for (kind, label), v in zip(pairs, vectors)])
        conn.commit()
        if i % 25 == 0:
            print(f"  {i}/{len(todo)} labelled")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", action="store_true", help="also extract labels with the LLM")
    args = parser.parse_args()
    load_dotenv(os.path.join(ROOT, ".env"))
    with connect() as conn:
        index_chunks(conn)
        if args.labels:
            index_labels(conn)


if __name__ == "__main__":
    main()
