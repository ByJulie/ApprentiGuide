# Data

| Source | What | Licence / permission | Personal data? | How prepared |
|---|---|---|---|---|
| Legifrance API (PISTE), Code du travail (LEGITEXT000006072050) | 794 in-force articles: apprenticeship (L/R/D 62xx, L6332-14), paid leave and family leave (314x), transport (3261), meal vouchers (3262), foreign workers (5221), telework (L1222-9 to L1222-11), maximum working time (L3121-18 to L3121-22, L3121-27), young workers (L/R/D 3161 to 3165), weekly rest (L3132-1 to L3132-3), health check (L4624-1), favour principle (L2254-1) | Open data of the DILA, Licence Ouverte (to confirm on the PISTE terms of use). Accessed through the API, no scraping | No | Table of contents from the API, articles matching `ingest/scope.yaml`, one chunk per in-force article (`ingest/legifrance.py`, `ingest/clean.py`) |
| Legifrance API (PISTE), CESEDA (LEGITEXT000006070158) | 35 in-force articles: student residence permit (422), work authorisation (L414-10 to L414-15), keeping work after the permit expires (L433-3) | Same as above | No | Same as above |
| Legifrance API (PISTE), Code de la securite sociale (LEGITEXT000006073189) | 14 in-force articles on complementary health (mutuelle): L911-1, L911-7, L911-7-1, L911-8, D911-1 to D911-8 (with D911-1-1), R242-1-6 | Same as above | No | Same as above |
| data.gouv.fr, "Fiches pratiques et ressources de Service-Public.gouv.fr Particuliers" | 53 fiches chosen by theme in `ingest/scope.yaml` (zip of one XML per fiche, file date 2026-10-25) | Licence Ouverte (listed as `fr-lo`, version 1.0, on data.gouv.fr), source Service-Public.fr/DILA | No | Fiches read from the zip, text per chapter or situation (`ingest/service_public.py`), sections cut at 3000 characters (`ingest/clean.py`) |
| data.gouv.fr, "Fiches pratiques et ressources Entreprendre" | 11 fiches (for example F2728, work authorisation of a foreign employee) | Licence Ouverte (`fr-lo`), source Service-Public.fr/DILA | No | Same as above; fiches identical to a Particuliers one are kept once |

The database holds 1434 chunks: Code du travail 793, CESEDA 35, Code de la securite sociale 14, Particuliers 560, Entreprendre 32. Each chunk has the citation (`citation_id`: article number or fiche id), the source label, the date of the version in force, the section path, the URL and a structural `scope` (theme, and `audience: international` for foreign-worker topics). `scope` does not say that a chunk does not apply to someone when a value is missing. For Legifrance articles, the official `Nota` (limits on when the article applies) is appended to the text after "Nota :".

Not covered: the aivancity study regulations (written permission still pending), collective agreements, the rest of the Code de la securite sociale (only the mutuelle articles above).

## Reading the chunks (for `retrieve()`)
`app/pipeline.py` expects `retrieve(question)` to return `[{"id": ..., "text": ...}]`. Use `citation_id` as the `id` (article number such as `L6222-1`, or fiche id such as `F2918`): it is the citation shown in the answer. Several rows of a fiche share the same `citation_id` and are ordered by `part`. Connect with `DATABASE_URL` from `.env`.

Starting point, a full text search on the French index:
```python
import os
import psycopg

SQL = """SELECT citation_id, part, title, text, url, version_date, scope
FROM chunks
WHERE text_tsv @@ websearch_to_tsquery('french', %(question)s)
ORDER BY ts_rank(text_tsv, websearch_to_tsquery('french', %(question)s)) DESC
LIMIT 5"""

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    rows = conn.execute(SQL, {"question": "congés payés apprenti"}).fetchall()
sources = [{"id": row[0], "text": row[3]} for row in rows]
```
Tested on this database: `congés payés apprenti` returns 5 chunks (F2918, F3059, F2258, L6222-35, F31633). A whole sentence such as `Combien de jours de congés ai-je en apprentissage ?` returns 0 rows, because `websearch_to_tsquery` requires every word to match. It is only a starting point: the course recommends hybrid search (keywords plus embeddings) and a reranker on top. To favour a topic, filter or boost on `scope->>'theme'` (a missing value does not mean the chunk does not apply).

## Teammates: get the database without PISTE keys
The five extracted files `data/raw/legifrance_*.jsonl` and `data/raw/service_public_*.jsonl` (cleaned into chunks by `ingest.load`) are committed (open data, the source is kept in each chunk). Everyone builds their own local database from them: no PISTE keys, no download. Docker Desktop must be running.
```bash
git pull
pip install -r requirements.txt                      # or at least: pyyaml python-dotenv psycopg[binary]
cp .env.example .env                                 # Windows: copy .env.example .env
```
Edit your own `.env`: choose any `POSTGRES_PASSWORD` (letters and digits only) and write the same value in `DATABASE_URL` in place of `<POSTGRES_PASSWORD>`. Never commit `.env` and never share your password: each person has an independent local database.
```bash
docker compose --env-file .env -f db/docker-compose.yml up -d
python -m ingest.load                                # from the repository root; prints "1434 chunks loaded"
```
Check: `docker compose --env-file .env -f db/docker-compose.yml exec db psql -U apprentiguide -d apprentiguide -c "SELECT corpus, count(*) FROM chunks GROUP BY 1"` should list 5 corpora (ceseda 35, code_travail 793, securite_sociale 14, sp_entreprendre 32, sp_particuliers 560 rows).

When the data lead changes the cleaning or the JSONL files: `git pull`, then `python -m ingest.load` again (it replaces the whole table). If the password was changed after the first start, the database keeps the old one: run `docker compose --env-file .env -f db/docker-compose.yml down -v` first (this deletes the local data, which `ingest.load` rebuilds).

## Rebuild
```bash
docker compose --env-file .env -f db/docker-compose.yml up -d   # POSTGRES_PASSWORD and DATABASE_URL in .env
python -I ingest/legifrance.py                          # needs PISTE_CLIENT_ID / PISTE_CLIENT_SECRET (production keys)
python -I ingest/service_public.py
python -m ingest.load                                   # full reload of the chunks table
python -m pytest tests
```
Only the five `data/raw/*.jsonl` files are committed; the caches (`tm_*.json`, `sp_*/`) are not. The Legifrance table of contents is cached in `data/raw/tm_*.json`: delete it to refresh.
`config.yaml` points to the PISTE production URLs, so the keys in `.env` must be production keys: sandbox keys are refused (HTTP 400).
