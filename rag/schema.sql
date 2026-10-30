-- Retrieval tables, next to the chunks table (db/schema.sql). ingest.load replaces the chunks,
-- so vectors and labels are keyed by (corpus, citation_id, part) plus a hash of the text:
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

-- Label index: short handles per document (article or fiche), separate from the content.
-- kind: category (1-3, English, abstract), keyword (up to 12, original French wording),
-- concept (up to 8, English, abstraction of a keyword).
CREATE TABLE IF NOT EXISTS doc_labels (
    corpus       TEXT NOT NULL,
    citation_id  TEXT NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('category', 'keyword', 'concept')),
    label        TEXT NOT NULL,
    doc_hash     TEXT NOT NULL,
    embedding    VECTOR(1024) NOT NULL,
    PRIMARY KEY (corpus, citation_id, kind, label)
);
CREATE INDEX IF NOT EXISTS doc_labels_hnsw ON doc_labels USING hnsw (embedding vector_cosine_ops);
