-- Retrieval tables, next to the chunks table (db/schema.sql). ingest.load replaces the chunks,
-- so vectors are keyed by (corpus, citation_id, part) plus a hash of the text:
-- a reload only recomputes what changed.
CREATE EXTENSION IF NOT EXISTS vector;

-- Content index: one vector per chunk (bge-m3, 1024 dimensions, normalised).
CREATE TABLE IF NOT EXISTS chunk_vectors (
    corpus       TEXT NOT NULL,
    citation_id  TEXT NOT NULL,
    part         SMALLINT NOT NULL,
    text_hash    TEXT NOT NULL,
    embedding    VECTOR(1024) NOT NULL,
    PRIMARY KEY (corpus, citation_id, part)
);
CREATE INDEX IF NOT EXISTS chunk_vectors_hnsw ON chunk_vectors USING hnsw (embedding vector_cosine_ops);
