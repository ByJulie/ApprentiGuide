CREATE TABLE IF NOT EXISTS chunks (
    id            SERIAL PRIMARY KEY,
    source        TEXT NOT NULL,
    corpus        TEXT NOT NULL,
    citation_id   TEXT NOT NULL,
    part          SMALLINT NOT NULL DEFAULT 0,
    title         TEXT,
    section_path  TEXT,
    text          TEXT NOT NULL,
    version_date  DATE,
    source_label  TEXT NOT NULL,
    url           TEXT,
    scope         JSONB,
    lang          TEXT NOT NULL DEFAULT 'fr',
    text_tsv      TSVECTOR GENERATED ALWAYS AS (to_tsvector('french', coalesce(title, '') || ' ' || text)) STORED,
    UNIQUE (corpus, citation_id, part)
);
-- Tables created before regulations were added still have this CHECK on source.
ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_source_check;
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (text_tsv);
