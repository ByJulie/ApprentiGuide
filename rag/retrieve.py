"""Hybrid retrieval: three signals fused by rank, then a cross-encoder decides the order and whether to answer.

Signals (each returns chunk keys ranked best first):
  exact     an article or fiche number written in the question (L6222-35, F2918) goes straight to it
  fulltext  PostgreSQL French full text, any word may match (BM25-like: wins on legal terms and numbers)
  dense     bge-m3 on the chunks (wins on paraphrases, works across French and English)
Fusion: reciprocal rank fusion. Ranking signals are not comparable, ranks are.
Decision: the reranker scores the fused candidates; under the threshold the system abstains in code,
so NOT FOUND never depends on the model's judgement.
"""
import os
import re
from dataclasses import dataclass, field

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from rag import models

RRF_K = 60
ARTICLE_RE = re.compile(r"\b([LRD]\.?\s?\*?\s?\d{3,4}-\d+(?:-\d+)*)\b", re.IGNORECASE)
FICHE_RE = re.compile(r"\b(F\d{2,6})\b", re.IGNORECASE)


@dataclass
class Retrieval:
    sources: list[dict]
    abstain: bool
    best_score: float
    # Numbers written in the question that are not in the corpus: the answer must say so
    # instead of answering from a neighbouring article.
    missing_ids: list[str] = field(default_factory=list)
    question_fr: str | None = None


def cited_ids(question: str) -> list[str]:
    """Article and fiche numbers written in the question, normalised (L 6222-35 -> L6222-35)."""
    ids = [_norm(m) for m in ARTICLE_RE.findall(question)]
    ids += [m.upper() for m in FICHE_RE.findall(question)]
    return list(dict.fromkeys(ids))


def _norm(ref: str) -> str:
    return re.sub(r"[\s.]", "", ref).upper()


def strip_unasked_ids(question: str, text: str) -> str:
    """Remove from a rewritten question the article or fiche numbers the user did not write."""
    asked = set(cited_ids(question))
    for pattern in (ARTICLE_RE, FICHE_RE):
        text = pattern.sub(lambda m: m.group(0) if _norm(m.group(0)) in asked else "", text)
    text = re.sub(r"\(\s*(?:article|art\.|fiche)?\s*\)", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def rrf(rankings: list[list], k: int = RRF_K) -> list:
    """Keys ordered by the sum of 1 / (k + rank) over the rankings where they appear."""
    scores: dict = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def or_tsquery(conn, question: str) -> str | None:
    """French lexemes of the question joined with OR: a full sentence still matches."""
    row = conn.execute(
        "SELECT string_agg(DISTINCT quote_literal(lexeme), ' | ') AS q "
        "FROM unnest(to_tsvector('french', %s))", (question,)).fetchone()
    return row["q"] if row and row["q"] else None


KEY = "corpus, citation_id, part"


def _keys(rows) -> list[tuple]:
    return [(r["corpus"], r["citation_id"], r["part"]) for r in rows]


def exact_signal(conn, ids: list[str]) -> list[tuple]:
    if not ids:
        return []
    return _keys(conn.execute(f"SELECT {KEY} FROM chunks WHERE upper(citation_id) = ANY(%s) ORDER BY part",
                              (ids,)).fetchall())


def fulltext_signal(conn, question: str, limit: int) -> list[tuple]:
    query = or_tsquery(conn, question)
    if not query:
        return []
    return _keys(conn.execute(
        f"""SELECT {KEY} FROM chunks WHERE text_tsv @@ to_tsquery('french', %(q)s)
            ORDER BY ts_rank_cd(text_tsv, to_tsquery('french', %(q)s), 32) DESC LIMIT %(n)s""",
        {"q": query, "n": limit}).fetchall())


def dense_signal(conn, vector, limit: int) -> list[tuple]:
    return _keys(conn.execute(
        f"SELECT {KEY} FROM chunk_vectors ORDER BY embedding <=> %s LIMIT %s", (vector, limit)).fetchall())


def fetch(conn, keys: list[tuple]) -> list[dict]:
    if not keys:
        return []
    rows = conn.execute(
        f"""SELECT {KEY}, title, section_path, text, source_label, url, version_date, scope
            FROM chunks WHERE (corpus, citation_id, part) IN (SELECT * FROM unnest(%s::text[], %s::text[], %s::smallint[]))""",
        ([k[0] for k in keys], [k[1] for k in keys], [k[2] for k in keys])).fetchall()
    by_key = {(r["corpus"], r["citation_id"], r["part"]): r for r in rows}
    return [by_key[k] for k in keys if k in by_key]


PUBLIC_SECTOR_DOC = re.compile(r"secteur public|fonction publique", re.I)
PUBLIC_SECTOR_QUESTION = re.compile(r"public|mairie|fonction publique|collectivit|administration|h[ôo]pital|"
                                    r"civil service|state employer", re.I)


def is_public_sector(row: dict) -> bool:
    """Texts written for the public sector (their section title or fiche title says so)."""
    return bool(PUBLIC_SECTOR_DOC.search(f"{row.get('section_path') or ''} {row.get('title') or ''}"))


def sector_priority(question: str, ranked: list[tuple]) -> list[tuple]:
    """Our users are apprentices in private companies unless they say otherwise: public-sector texts
    go after the others, never removed (like a country priority: the user's rules first)."""
    if PUBLIC_SECTOR_QUESTION.search(question):
        return ranked
    return [x for x in ranked if not is_public_sector(x[0])] + [x for x in ranked if is_public_sector(x[0])]


def diversify(ranked: list[tuple], top_k: int, max_per_doc: int | None) -> list[tuple]:
    """Best passages first, at most max_per_doc per article or fiche: five places for one fiche's
    sections would hide the other documents from the model."""
    if not max_per_doc:
        return ranked[:top_k]
    chosen, per_doc = [], {}
    for row, score in ranked:
        doc = row["citation_id"].upper()
        if per_doc.get(doc, 0) < max_per_doc:
            chosen.append((row, score))
            per_doc[doc] = per_doc.get(doc, 0) + 1
        if len(chosen) == top_k:
            break
    return chosen


def passage(row: dict) -> str:
    """Text given to the reranker and to the LLM: where it comes from, then the rule."""
    return f"{row['title'] or row['citation_id']}\n{row['text']}"


class Retriever:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.connect()

    def connect(self) -> None:
        self.conn = psycopg.connect(os.environ["DATABASE_URL"], row_factory=psycopg.rows.dict_row,
                                    autocommit=True)
        register_vector(self.conn)

    def signals(self, question: str, question_fr: str | None = None) -> dict[str, list[tuple]]:
        n = self.cfg.get("candidates_per_signal", 20)
        vector = np.array(models.embed([question])[0], dtype=np.float32)
        ids = cited_ids(question)
        enabled = self.cfg.get("signals", ["exact", "fulltext", "dense"])
        out = {}
        if "exact" in enabled:
            out["exact"] = exact_signal(self.conn, ids)
        if "fulltext" in enabled:
            # French lexemes only match French text: use the French version when there is one.
            out["fulltext"] = fulltext_signal(self.conn, question_fr or question, n)
        if "dense" in enabled:
            out["dense"] = dense_signal(self.conn, vector, n)
        return out

    def __call__(self, question: str, question_fr: str | None = None) -> Retrieval:
        """question_fr: the question rewritten in French. The corpus is French and the cross-encoder
        scores an English question against French text far too low, so it reranks with this version.

        A connection closed by the database (Docker restarted, laptop asleep) is reopened once: otherwise
        every following question would fail until the process is restarted."""
        try:
            return self._retrieve(question, question_fr)
        except psycopg.OperationalError:
            self.connect()
            return self._retrieve(question, question_fr)

    def _retrieve(self, question: str, question_fr: str | None) -> Retrieval:
        ids = cited_ids(question)
        sig = self.signals(question, question_fr)
        fused = rrf([r for r in sig.values() if r])[: self.cfg.get("rerank_candidates", 30)]
        rows = fetch(self.conn, fused)
        top_k, threshold = self.cfg.get("top_k", 5), self.cfg.get("abstain_threshold", 0.0)
        if self.cfg.get("rerank", True):
            scores = models.rerank(question_fr or question, [passage(r) for r in rows])
            ranked = sorted(zip(rows, scores), key=lambda x: x[1], reverse=True)
        else:
            ranked = [(r, 1.0 / (RRF_K + i)) for i, r in enumerate(rows, 1)]
        if self.cfg.get("sector_priority"):
            ranked = sector_priority(question_fr or question, ranked)
        # An article the user named is always shown to the model, whatever its score.
        named = [(r, s) for r, s in ranked if r["citation_id"].upper() in ids]
        rest = [(r, s) for r, s in ranked if r["citation_id"].upper() not in ids]
        chosen = diversify(named + rest, top_k, self.cfg.get("max_per_doc"))
        best = max((s for _, s in ranked), default=0.0)
        found = {r["citation_id"].upper() for r in rows}
        sources = [{"id": r["citation_id"], "text": passage(r), "score": round(s, 3),
                    "title": r["title"], "url": r["url"], "source_label": r["source_label"],
                    "version_date": str(r["version_date"]) if r["version_date"] else None,
                    "international": (r.get("scope") or {}).get("audience") == "international"}
                   for r, s in chosen]
        missing = [i for i in ids if i not in found]
        # The user asked about an article we do not have, and none they named was found: answering
        # from the closest articles would present a neighbour as the answer. Abstain and say why.
        abstain = (not named and best < threshold) or bool(missing and not named)
        return Retrieval(sources=sources, abstain=abstain, best_score=round(best, 3), missing_ids=missing,
                         question_fr=question_fr)
