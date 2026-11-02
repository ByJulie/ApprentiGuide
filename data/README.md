# Data

| Source | What | Licence / permission | Personal data? | How prepared |
|---|---|---|---|---|
| Legifrance API (PISTE), Code du travail (LEGITEXT000006072050) | 794 in-force articles: apprenticeship (L/R/D 62xx, L6332-14), paid leave and family leave (314x), transport (3261), meal vouchers (3262), foreign workers (5221), telework (L1222-9 to L1222-11), maximum working time (L3121-18 to L3121-22, L3121-27), young workers (L/R/D 3161 to 3165), weekly rest (L3132-1 to L3132-3), health check (L4624-1), favour principle (L2254-1) | Open data of the DILA, Licence Ouverte (to confirm on the PISTE terms of use). Accessed through the API, no scraping | No | Table of contents from the API, articles matching `ingest/scope.yaml`, one chunk per in-force article (`ingest/legifrance.py`, `ingest/clean.py`) |
| Legifrance API (PISTE), CESEDA (LEGITEXT000006070158) | 35 in-force articles: student residence permit (422), work authorisation (L414-10 to L414-15), keeping work after the permit expires (L433-3) | Same as above | No | Same as above |
| Legifrance API (PISTE), Code de la securite sociale (LEGITEXT000006073189) | 14 in-force articles on complementary health (mutuelle): L911-1, L911-7, L911-7-1, L911-8, D911-1 to D911-8 (with D911-1-1), R242-1-6 | Same as above | No | Same as above |
| data.gouv.fr, "Fiches pratiques et ressources de Service-Public.gouv.fr Particuliers" | 53 fiches chosen by theme in `ingest/scope.yaml` (zip of one XML per fiche, file date 2026-10-25) | Licence Ouverte (listed as `fr-lo`, version 1.0, on data.gouv.fr), source Service-Public.fr/DILA | No | Fiches read from the zip, text per chapter or situation (`ingest/service_public.py`), sections cut at 3000 characters (`ingest/clean.py`) |
| data.gouv.fr, "Fiches pratiques et ressources Entreprendre" | 11 fiches (for example F2728, work authorisation of a foreign employee) | Licence Ouverte (`fr-lo`), source Service-Public.fr/DILA | No | Same as above; fiches identical to a Particuliers one are kept once |
| aivancity, "Règlement pédagogique 2025-2026" (in full) and "Règlement intérieur 2025-2026" (articles 3, 5, 7, 9, 10 and 17) | 73 chunks: one per numbered section (`aivancity-RP-3.3.4`) or per article / sub-article (`aivancity-RI-7.3`), cut at 3000 characters. No public URL (`url` empty): the apps show the id, the label and the section title without a link | Permission received from the school by email, for use in this project. The PDF files and the chunks are never committed (the repository is public) | No | PDF files in `data/private/`, read by `ingest/regulations.py` when `python -m ingest.load` runs; skipped when the files are absent |

The database holds 1507 chunks when the two regulation PDF files are in `data/private/` (1434 without them): Code du travail 793, CESEDA 35, Code de la securite sociale 14, Particuliers 560, Entreprendre 32, aivancity rules 73 (pedagogical 56, internal 17). Each chunk has the citation (`citation_id`: article number or fiche id), the source label, the date of the version in force, the section path, the URL and a structural `scope` (theme, and `audience: international` for foreign-worker topics). `scope` does not say that a chunk does not apply to someone when a value is missing. For Legifrance articles, the official `Nota` (limits on when the article applies) is appended to the text after "Nota :".

Not covered: the aivancity charters (IT, campus, AI Clinic), the other articles of the internal rules, collective agreements, the rest of the Code de la securite sociale (only the mutuelle articles above).

## School regulations (aivancity)
Source: the school's French regulations 2025-2026, sent to the team with permission to use them in this project (the pedagogical rules in full, and articles 3, 5, 7, 9, 10 and 17 of the internal rules; the IT, campus and AI Clinic charters are not used). The repository is public: the PDF files and the chunks made from them are never committed.
- **Files:** `data/private/reglement_pedagogique.pdf` and `data/private/reglement_interieur.pdf` (git-ignored; names and options in `ingest/scope.yaml`, key `regulations`). Each person copies them there before `python -m ingest.load`. Without them the load prints a warning, loads 1434 chunks and, being a full reload, removes the regulation chunks a database already had. The load also needs `pypdf` (`pip install -r requirements.txt`).
- **Chunks (73):** one per numbered section of the pedagogical rules (56) and per article or sub-article of the internal rules (17), cut at 3000 characters like the fiches. Code: `ingest/regulations.py`.
- **Citation:** `citation_id` = `<school>-<document>-<number>`: `aivancity-RP-3.3.4` (RP: pedagogical rules, section 3.3.4), `aivancity-RI-7.3` (RI: internal rules, article 7.3). The school is in the id so a second school can be added (one entry in `scope.yaml`). The year is not in the id: next year's PDF replaces this one on reload. No page number is stored.
- **Display:** `source_label` is "Règlement pédagogique 2025-2026" or "Règlement intérieur 2025-2026". `title` is "Règlement pédagogique, section 3.3.4 : Absence pendant le stage". `scope` is `{"theme": "ecole", "school": "aivancity"}`: the retrieval can favour these chunks for questions about the school with `scope->>'school'`.
- **No URL:** these chunks have `url` empty (the PDFs are private). Before this was handled, the interface wrote `[id](None)` (a broken link) and the Open WebUI card id was `None`. Now `source_link` in `app/pipeline.py` shows the bare id; `ui.py` / `api.py` show the section title under it; the passage is introduced by "Extrait" instead of "Article"; `source_card` uses the id in place of the URL; the footers mention the school regulations. Not checked: how Open WebUI shows a card whose id is not a URL.
- **Version date:** 2025-05-02 for the pedagogical rules (adopted 2 May 2025). Empty for the internal rules: the file received says "adoptée le [o]" (date not filled in). Fill it in `scope.yaml` if the school gives one.
- **Effect on the evaluation:** some `no_answer` questions of `eval/testset.jsonl` are about the school (Q43, Q44, possibly Q11, Q12, Q41, Q52) and the regulations may now answer them. With the expected NOT FOUND kept, a correct answer counts as a failure. Check them before the final runs (details in `docs/evaluation_report.md`, section 6).
- **Titles have no accents** (as in the PDF), and an acronym in capitals such as VAE is written "Vae".

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
python -m ingest.load                                # from the repository root; prints "1434 chunks loaded" (1507 with the regulation PDFs, see Rebuild)
```
Check: `docker compose --env-file .env -f db/docker-compose.yml exec db psql -U apprentiguide -d apprentiguide -c "SELECT corpus, count(*) FROM chunks GROUP BY 1"` should list 5 corpora, 7 with the regulation PDFs (aivancity_ri 17, aivancity_rp 56; ceseda 35, code_travail 793, securite_sociale 14, sp_entreprendre 32, sp_particuliers 560 rows).

When the data lead changes the cleaning or the JSONL files: `git pull`, then `python -m ingest.load` again (it replaces the whole table). If the password was changed after the first start, the database keeps the old one: run `docker compose --env-file .env -f db/docker-compose.yml down -v` first (this deletes the local data, which `ingest.load` rebuilds).

## Rebuild
```bash
docker compose --env-file .env -f db/docker-compose.yml up -d   # POSTGRES_PASSWORD and DATABASE_URL in .env
python -I ingest/legifrance.py                          # needs PISTE_CLIENT_ID / PISTE_CLIENT_SECRET (production keys)
python -I ingest/service_public.py
python -m ingest.load                                   # full reload of the chunks table
python -m pytest tests
```
Put the two regulation PDF files in `data/private/` (`reglement_interieur.pdf`, `reglement_pedagogique.pdf`, names in `ingest/scope.yaml`) before `python -m ingest.load`; this folder is git-ignored and the chunks they give (corpora `aivancity_ri`, `aivancity_rp`, `scope` = `{"theme": "ecole", "school": "aivancity"}`) are not stored in the repository.
Only the five `data/raw/*.jsonl` files are committed; the caches (`tm_*.json`, `sp_*/`) are not. The Legifrance table of contents is cached in `data/raw/tm_*.json`: delete it to refresh.
`config.yaml` points to the PISTE production URLs, so the keys in `.env` must be production keys: sandbox keys are refused (HTTP 400).
